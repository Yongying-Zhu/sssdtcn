"""
Comprehensive experiment runner for reviewer-requested additions.

Runs the following experiments and saves results to table/:
  - CSDI (pure diffusion): 5 seeds × 2 datasets × 20%-80% missing rates
  - SSSD (explicit/S4 only): 5 seeds × 2 datasets × 20%-80% missing rates
  - SSSDTCN (full model): 5 seeds × 2 datasets × 20%-80% missing rates
  - Implicit-only (ablation): 1 seed × Debutanizer × 50% missing rate

All diffusion-based results reported as mean ± std over 5 seeds.
Output: XLSX files in /home/user/sssdtcn/table/
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Resolve paths — ROOT must come BEFORE ablation so that 'models' resolves
# to the project root models/, not ablation/models/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ablation added at the END so 'models' still resolves to ROOT/models/
_ABL = os.path.join(ROOT, 'ablation')
if _ABL not in sys.path:
    sys.path.append(_ABL)

# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length=60):
        self.data = torch.FloatTensor(data)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length + 1

    def __getitem__(self, idx):
        return self.data[idx:idx + self.sequence_length]


def load_data(data_path, sequence_length=60, batch_size=32, train_ratio=0.7, val_ratio=0.1):
    # Use numpy to load, skipping non-numeric header lines
    rows = []
    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                vals = [float(v) for v in line.split()]
                if len(vals) > 1:
                    rows.append(vals)
            except ValueError:
                pass
    data = np.array(rows)
    num_features = data.shape[1]
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    n = len(data_scaled)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_loader = DataLoader(TimeSeriesDataset(data_scaled[:train_end], sequence_length),
                              batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(TimeSeriesDataset(data_scaled[train_end:val_end], sequence_length),
                            batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(TimeSeriesDataset(data_scaled[val_end:], sequence_length),
                             batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader, scaler, num_features


def create_random_mask(data, missing_ratio):
    mask = torch.rand_like(data) > missing_ratio
    masked_data = data * mask.float()
    return mask, masked_data


# ─────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────

def build_model(model_type, num_features, device):
    """Build model by type name."""
    if model_type == 'CSDI':
        from experiments.exp_models.csdi_model import CSDIModel
        model = CSDIModel(input_dim=num_features, hidden_dim=128, embedding_dim=256,
                          num_residual_layers=8, dropout=0.15)
    elif model_type == 'SSSD':
        from ablation.models.explicit_only_diffusion import ExplicitOnlyDiffusionModel  # noqa
        model = ExplicitOnlyDiffusionModel(input_dim=num_features, hidden_dim=128,
                                           embedding_dim=256, num_residual_layers=6,
                                           dropout=0.15, num_s4_layers=4)
    elif model_type == 'SSSDTCN':
        from models.implicit_explicit_diffusion import ImplicitExplicitDiffusionModel  # noqa
        model = ImplicitExplicitDiffusionModel(input_dim=num_features, hidden_dim=128,
                                               embedding_dim=256, num_residual_layers=6,
                                               dropout=0.15)
    elif model_type == 'ImplicitOnly':
        from experiments.exp_models.implicit_only_model import ImplicitOnlyDiffusionModel
        model = ImplicitOnlyDiffusionModel(input_dim=num_features, hidden_dim=128,
                                           embedding_dim=256, num_residual_layers=6,
                                           dropout=0.15, num_conv_layers=8)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model.to(device)


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_epoch(model, train_loader, optimizer, device, missing_ratio=0.5, max_batches=60):
    model.train()
    total_loss, n = 0.0, 0
    for i, batch_data in enumerate(train_loader):
        if max_batches and i >= max_batches:
            break
        batch_data = batch_data.to(device)
        B, T, F = batch_data.shape
        mask, masked_data = create_random_mask(batch_data, missing_ratio)
        mask = mask.float()
        masked_data = masked_data
        timesteps = torch.zeros(B, dtype=torch.long, device=device)
        pred = model(masked_data, timesteps, mask, masked_data)
        mask_missing = (mask == 0)
        loss = ((pred - batch_data) ** 2 * mask_missing.float()).sum() / mask_missing.sum().clamp(min=1)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def validate(model, val_loader, device, missing_ratio=0.5, max_batches=20):
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            if max_batches and i >= max_batches:
                break
            batch_data = batch_data.to(device)
            B, T, F = batch_data.shape
            mask, masked_data = create_random_mask(batch_data, missing_ratio)
            mask = mask.float()
            timesteps = torch.zeros(B, dtype=torch.long, device=device)
            pred = model(masked_data, timesteps, mask, masked_data)
            mask_missing = (mask == 0)
            loss = ((pred - batch_data) ** 2 * mask_missing.float()).sum() / mask_missing.sum().clamp(min=1)
            total_loss += loss.item()
            n += 1
    return total_loss / max(n, 1)


def train_model(model, train_loader, val_loader, device, ckpt_path,
                num_epochs=30, lr=1e-4, weight_decay=1e-5,
                patience=6, threshold=0.005, missing_ratio=0.5):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = float('inf')
    loss_hist_train, loss_hist_val = [], []

    for epoch in range(num_epochs):
        tr_loss = train_epoch(model, train_loader, optimizer, device, missing_ratio, max_batches=30)
        val_loss = validate(model, val_loader, device, missing_ratio, max_batches=10)
        loss_hist_train.append(tr_loss)
        loss_hist_val.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({'model_state_dict': model.state_dict(), 'val_loss': val_loss}, ckpt_path)

        if epoch >= patience:
            recent_tr = loss_hist_train[-patience:]
            recent_val = loss_hist_val[-patience:]
            if (max(recent_tr) - min(recent_tr) < threshold and
                    max(recent_val) - min(recent_val) < threshold):
                print(f"    Early stop at epoch {epoch+1}")
                break

    print(f"    Best val loss: {best_val:.6f}")
    # Restore best weights
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    return model


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate_model(model, test_loader, device, missing_ratio, eval_runs=3):
    """Evaluate model at a specific missing ratio. Returns (mae, rmse)."""
    model.eval()
    mae_runs, rmse_runs = [], []

    for _ in range(eval_runs):
        mae_list, rmse_list = [], []
        with torch.no_grad():
            for batch_data in test_loader:
                batch_data = batch_data.to(device)
                B, T, F = batch_data.shape
                mask, masked_data = create_random_mask(batch_data, missing_ratio)
                mask = mask.float()
                timesteps = torch.zeros(B, dtype=torch.long, device=device)
                pred = model(masked_data, timesteps, mask, masked_data)
                mask_missing = (mask == 0)
                if mask_missing.sum() > 0:
                    err = (pred - batch_data) * mask_missing.float()
                    mae_list.append((torch.abs(err).sum() / mask_missing.sum()).item())
                    rmse_list.append((torch.sqrt((err ** 2).sum() / mask_missing.sum())).item())
        mae_runs.append(np.mean(mae_list))
        rmse_runs.append(np.mean(rmse_list))

    return np.mean(mae_runs), np.mean(rmse_runs)


# ─────────────────────────────────────────────
# Main experiment logic
# ─────────────────────────────────────────────

DATASETS = {
    'Debutanizer': os.path.join(ROOT, 'data', 'debutanizer_data.txt'),
    'SRU': os.path.join(ROOT, 'data', 'SRU_data.txt'),
}

MISSING_RATIOS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
SEEDS = [42, 123, 456, 789, 2024]
CKPT_DIR = os.path.join(ROOT, 'checkpoints', 'experiments')
TABLE_DIR = os.path.join(ROOT, 'table')


def run_multi_seed_experiment(model_type, dataset_name, data_path, device,
                              seeds=SEEDS, missing_ratios=MISSING_RATIOS):
    """
    Train model with multiple seeds and evaluate across missing ratios.
    Returns dict: {missing_ratio: {'mae': [...], 'rmse': [...]}}
    """
    os.makedirs(CKPT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Model: {model_type} | Dataset: {dataset_name}")
    print(f"Seeds: {seeds}")
    print(f"{'='*60}")

    train_loader, val_loader, test_loader, _, num_features = load_data(data_path)

    seed_results = {mr: {'mae': [], 'rmse': []} for mr in missing_ratios}

    for seed in seeds:
        print(f"\n  Seed {seed}:")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = build_model(model_type, num_features, device)
        ckpt_path = os.path.join(CKPT_DIR, f'{model_type}_{dataset_name}_seed{seed}.pth')

        model = train_model(model, train_loader, val_loader, device, ckpt_path)

        for mr in missing_ratios:
            mae, rmse = evaluate_model(model, test_loader, device, mr)
            seed_results[mr]['mae'].append(mae)
            seed_results[mr]['rmse'].append(rmse)
            print(f"    MR={int(mr*100)}%: MAE={mae:.4f}, RMSE={rmse:.4f}")

    return seed_results


def run_single_experiment(model_type, dataset_name, data_path, device,
                          seed=42, missing_ratio=0.5):
    """
    Train model once and evaluate at a single missing ratio.
    Returns (mae, rmse).
    """
    os.makedirs(CKPT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Model: {model_type} | Dataset: {dataset_name} | MR={int(missing_ratio*100)}%")
    print(f"{'='*60}")

    train_loader, val_loader, test_loader, _, num_features = load_data(data_path)

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model(model_type, num_features, device)
    ckpt_path = os.path.join(CKPT_DIR, f'{model_type}_{dataset_name}_seed{seed}.pth')
    model = train_model(model, train_loader, val_loader, device, ckpt_path)

    mae, rmse = evaluate_model(model, test_loader, device, missing_ratio, eval_runs=5)
    print(f"  Result: MAE={mae:.4f}, RMSE={rmse:.4f}")
    return mae, rmse


def format_mean_std(values_list):
    """Format list of values as 'mean±std'."""
    mean = np.mean(values_list)
    std = np.std(values_list, ddof=1) if len(values_list) > 1 else 0.0
    return f"{mean:.4f}±{std:.4f}"


def build_comparison_sheet(dataset_name, data_path, device):
    """Build DataFrame for Table 3/4 (Baselines) for one dataset."""
    model_types = ['CSDI', 'SSSD', 'SSSDTCN']
    mr_labels = [f"{int(mr*100)}%" for mr in MISSING_RATIOS]

    rows_mae = {mt: [] for mt in model_types}
    rows_rmse = {mt: [] for mt in model_types}

    for mt in model_types:
        results = run_multi_seed_experiment(mt, dataset_name, data_path, device)
        for mr in MISSING_RATIOS:
            rows_mae[mt].append(format_mean_std(results[mr]['mae']))
            rows_rmse[mt].append(format_mean_std(results[mr]['rmse']))

    # Build combined DataFrame
    data_rows = []
    for mt in model_types:
        for i, mr in enumerate(mr_labels):
            data_rows.append({
                'Model': mt,
                'Missing Rate': mr,
                'MAE (mean±std)': rows_mae[mt][i],
                'RMSE (mean±std)': rows_rmse[mt][i],
            })

    return pd.DataFrame(data_rows)


def build_ablation_sheet(device):
    """Build DataFrame for Table 5 extension (Implicit-Only on Debutanizer 50%)."""
    dataset_name = 'Debutanizer'
    data_path = DATASETS['Debutanizer']
    missing_ratio = 0.5

    mae, rmse = run_single_experiment('ImplicitOnly', dataset_name, data_path,
                                      device, seed=42, missing_ratio=missing_ratio)
    df = pd.DataFrame([{
        'Model': 'Implicit-Only (no S4)',
        'Dataset': dataset_name,
        'Missing Rate': f"{int(missing_ratio*100)}%",
        'MAE': f"{mae:.4f}",
        'RMSE': f"{rmse:.4f}",
    }])
    return df


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(TABLE_DIR, exist_ok=True)

    excel_path = os.path.join(TABLE_DIR, 'experiment_results.xlsx')

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

        # ── Baselines on Debutanizer ──────────────────────────────────────
        print("\n\n>>> Building Debutanizer baseline results...")
        df_deb = build_comparison_sheet('Debutanizer', DATASETS['Debutanizer'], device)
        df_deb.to_excel(writer, sheet_name='Baselines_Debutanizer', index=False)
        print(df_deb.to_string(index=False))

        # ── Baselines on SRU ──────────────────────────────────────────────
        print("\n\n>>> Building SRU baseline results...")
        df_sru = build_comparison_sheet('SRU', DATASETS['SRU'], device)
        df_sru.to_excel(writer, sheet_name='Baselines_SRU', index=False)
        print(df_sru.to_string(index=False))

        # ── Ablation: Implicit-only on Debutanizer 50% ───────────────────
        print("\n\n>>> Running Implicit-Only ablation (Debutanizer, 50%)...")
        df_ablation = build_ablation_sheet(device)
        df_ablation.to_excel(writer, sheet_name='Ablation_ImplicitOnly', index=False)
        print(df_ablation.to_string(index=False))

    print(f"\n\nAll results saved to: {excel_path}")


if __name__ == '__main__':
    main()
