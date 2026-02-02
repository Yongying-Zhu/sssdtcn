"""
评估消融实验模型性能

评估指标：MAE, RMSE
评估缺失率：20%, 30%, 40%, 50%, 60%, 70%, 80%
"""
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from models.single_scale_diffusion import SingleScaleDiffusionModel
from models.explicit_only_diffusion import ExplicitOnlyDiffusionModel
from data_loader import load_data, create_random_mask
from config_debutanizer import ConfigDebutanizer as Config


def evaluate_model(model, test_loader, device, missing_ratio, num_runs=5):
    """评估模型在指定缺失率下的性能"""
    model.eval()
    all_mae = []
    all_rmse = []

    for run in range(num_runs):
        mae_list = []
        rmse_list = []

        with torch.no_grad():
            for batch_data in test_loader:
                batch_data = batch_data.to(device)
                batch_size, seq_len, features = batch_data.shape

                mask, masked_data = create_random_mask(batch_data, missing_ratio)
                mask = mask.float().to(device)
                masked_data = masked_data.to(device)

                timesteps = torch.zeros(batch_size, dtype=torch.long, device=device)
                predicted_data = model(masked_data, timesteps, mask, masked_data)

                mask_missing = (mask == 0)
                if mask_missing.sum() > 0:
                    error = (predicted_data - batch_data) * mask_missing.float()
                    mae = torch.abs(error).sum() / mask_missing.sum()
                    rmse = torch.sqrt((error ** 2).sum() / mask_missing.sum())
                    mae_list.append(mae.item())
                    rmse_list.append(rmse.item())

        all_mae.append(np.mean(mae_list))
        all_rmse.append(np.mean(rmse_list))

    return np.mean(all_mae), np.std(all_mae), np.mean(all_rmse), np.std(all_rmse)


def main():
    config = Config()
    missing_ratios = config.missing_ratios

    print("Loading Debutanizer dataset...")
    train_loader, val_loader, test_loader, scaler, num_features = load_data(
        config.data_path,
        sequence_length=config.sequence_length,
        batch_size=config.batch_size,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio
    )

    results = {
        'Missing Rate': [f'{int(r*100)}%' for r in missing_ratios]
    }

    # 评估单尺度+显式模型
    print("\n" + "=" * 60)
    print("评估: 单尺度 + 显式特征模型")
    print("=" * 60)

    single_scale_path = os.path.join(config.save_dir, 'single_scale', 'single_scale_best.pth')
    if os.path.exists(single_scale_path):
        model_ss = SingleScaleDiffusionModel(
            input_dim=num_features,
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
            num_residual_layers=config.num_residual_layers,
            dropout=config.dropout
        ).to(config.device)

        checkpoint = torch.load(single_scale_path, map_location=config.device)
        model_ss.load_state_dict(checkpoint['model_state_dict'])

        ss_mae_list = []
        ss_rmse_list = []
        for mr in tqdm(missing_ratios, desc="Single-Scale"):
            mae, mae_std, rmse, rmse_std = evaluate_model(model_ss, test_loader, config.device, mr)
            ss_mae_list.append(f"{mae:.4f}±{mae_std:.4f}")
            ss_rmse_list.append(f"{rmse:.4f}±{rmse_std:.4f}")
            print(f"  Missing {int(mr*100)}%: MAE={mae:.4f}±{mae_std:.4f}, RMSE={rmse:.4f}±{rmse_std:.4f}")

        results['Single-Scale MAE'] = ss_mae_list
        results['Single-Scale RMSE'] = ss_rmse_list
    else:
        print(f"Model not found: {single_scale_path}")

    # 评估仅显式模型
    print("\n" + "=" * 60)
    print("评估: 仅显式特征模型")
    print("=" * 60)

    explicit_only_path = os.path.join(config.save_dir, 'explicit_only', 'explicit_only_best.pth')
    if os.path.exists(explicit_only_path):
        model_eo = ExplicitOnlyDiffusionModel(
            input_dim=num_features,
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
            num_residual_layers=config.num_residual_layers,
            dropout=config.dropout,
            num_s4_layers=4
        ).to(config.device)

        checkpoint = torch.load(explicit_only_path, map_location=config.device)
        model_eo.load_state_dict(checkpoint['model_state_dict'])

        eo_mae_list = []
        eo_rmse_list = []
        for mr in tqdm(missing_ratios, desc="Explicit-Only"):
            mae, mae_std, rmse, rmse_std = evaluate_model(model_eo, test_loader, config.device, mr)
            eo_mae_list.append(f"{mae:.4f}±{mae_std:.4f}")
            eo_rmse_list.append(f"{rmse:.4f}±{rmse_std:.4f}")
            print(f"  Missing {int(mr*100)}%: MAE={mae:.4f}±{mae_std:.4f}, RMSE={rmse:.4f}±{rmse_std:.4f}")

        results['Explicit-Only MAE'] = eo_mae_list
        results['Explicit-Only RMSE'] = eo_rmse_list
    else:
        print(f"Model not found: {explicit_only_path}")

    # 保存结果到Excel
    df = pd.DataFrame(results)
    output_path = os.path.join(config.save_dir, 'ablation_results.xlsx')
    df.to_excel(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # 打印表格
    print("\n" + "=" * 80)
    print("消融实验结果汇总 (Debutanizer)")
    print("=" * 80)
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
