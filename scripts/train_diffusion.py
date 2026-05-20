"""
训练扩散模型（专利改进版）

专利创新二: 以 Huber 损失函数替换原 MSE 损失函数。
    - |error| ≤ τ 时: 0.5 · error²          （与 MSE 等价，精细惩罚小误差）
    - |error| > τ 时: τ · |error| - 0.5·τ²  （与 MAE 同阶，梯度恒为 ±τ，避免梯度爆炸）
    - 阈值 τ 取训练集缺失位置误差分布的第 80 百分位数，
      使绝大多数正常样本落于 MSE 区间，少数异常样本由 MAE 稳健处理。
"""
import os
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from tqdm import tqdm
import sys

from models.implicit_explicit_diffusion import ImplicitExplicitDiffusionModel
from data_loader import load_data, create_random_mask


# ---------------------------------------------------------------------------
# Huber 损失（专利核心改进 Innovation 2）
# ---------------------------------------------------------------------------

def huber_loss_on_missing(predicted, target, mask_missing, delta=1.0):
    """
    仅在缺失位置计算 Huber 损失。

    分段定义（以阈值 τ=delta 为界，连续且可微）:
        L_δ(e) = 0.5 · e²          当 |e| ≤ δ
               = δ · |e| - 0.5·δ²  当 |e| > δ

    Args:
        predicted:    [B, L, F] 模型预测值
        target:       [B, L, F] 真实值
        mask_missing: [B, L, F] bool, True 表示该位置缺失
        delta:        阈值 τ（由训练集误差分布百分位数确定）
    Returns:
        scalar 损失值
    """
    error = (predicted - target)[mask_missing]
    abs_error = error.abs()
    loss = torch.where(
        abs_error <= delta,
        0.5 * error.pow(2),
        delta * abs_error - 0.5 * (delta ** 2)
    )
    return loss.mean()


def compute_huber_threshold(model, train_loader, device, missing_ratio=0.5, percentile=80.0):
    """
    单次遍历训练集，以缺失位置绝对误差分布的 `percentile` 百分位数作为 Huber 阈值 τ。

    阈值选取原则（专利描述）：
        取 75%~90% 分位数，使绝大多数正常样本落于 MSE 区间（精细收敛），
        仅少数异常大误差样本由 MAE 稳健处理（规避梯度爆炸）。

    Args:
        percentile: 百分位数，默认80（即训练集误差的80th percentile）
    Returns:
        threshold: float  阈值 τ（最小值 1e-4 防止退化为纯 MAE）
    """
    model.eval()
    all_abs_errors = []

    with torch.no_grad():
        for batch_data in tqdm(train_loader, desc="Computing Huber threshold τ"):
            batch_data = batch_data.to(device)
            batch_size = batch_data.shape[0]

            mask, masked_data = create_random_mask(batch_data, missing_ratio)
            mask = mask.float().to(device)
            masked_data = masked_data.to(device)

            timesteps = torch.zeros(batch_size, dtype=torch.long, device=device)
            predicted = model(masked_data, timesteps, mask, masked_data)

            mask_missing = (mask == 0)
            abs_errors = (predicted - batch_data)[mask_missing].abs().cpu()
            all_abs_errors.append(abs_errors)

    all_abs_errors = torch.cat(all_abs_errors).float()
    threshold = torch.quantile(all_abs_errors, percentile / 100.0).item()

    model.train()
    return max(threshold, 1e-4)


# ---------------------------------------------------------------------------
# 训练 / 验证
# ---------------------------------------------------------------------------

def train_epoch(model, train_loader, optimizer, device, missing_ratio=0.5, huber_delta=1.0):
    """训练一个 epoch，使用 Huber 损失函数"""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_data in tqdm(train_loader, desc="Training"):
        batch_data = batch_data.to(device)
        batch_size, seq_len, features = batch_data.shape

        mask, masked_data = create_random_mask(batch_data, missing_ratio)
        mask = mask.float().to(device)
        masked_data = masked_data.to(device)

        # 直接预测完整数据（t=0，无需扩散采样）
        timesteps = torch.zeros(batch_size, dtype=torch.long, device=device)
        predicted_data = model(masked_data, timesteps, mask, masked_data)

        # Huber 损失（仅在缺失位置计算）
        mask_missing = (mask == 0)
        loss = huber_loss_on_missing(predicted_data, batch_data, mask_missing, delta=huber_delta)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate(model, val_loader, device, missing_ratio=0.5, huber_delta=1.0):
    """验证模型，使用 Huber 损失函数"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch_data in val_loader:
            batch_data = batch_data.to(device)
            batch_size, seq_len, features = batch_data.shape

            mask, masked_data = create_random_mask(batch_data, missing_ratio)
            mask = mask.float().to(device)
            masked_data = masked_data.to(device)

            timesteps = torch.zeros(batch_size, dtype=torch.long, device=device)
            predicted_data = model(masked_data, timesteps, mask, masked_data)

            mask_missing = (mask == 0)
            loss = huber_loss_on_missing(predicted_data, batch_data, mask_missing, delta=huber_delta)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


# ---------------------------------------------------------------------------
# 主训练流程
# ---------------------------------------------------------------------------

def train_model(config):
    """训练主函数"""
    os.makedirs(config.save_dir, exist_ok=True)

    print(f"Loading {config.dataset_name} dataset...")
    train_loader, val_loader, test_loader, scaler, num_features = load_data(
        config.data_path,
        sequence_length=config.sequence_length,
        batch_size=config.batch_size,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio
    )

    print("Creating model...")
    model = ImplicitExplicitDiffusionModel(
        input_dim=num_features,
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        num_residual_layers=config.num_residual_layers,
        dropout=config.dropout
    ).to(config.device)

    # ---- 确定 Huber 阈值 τ（专利 Innovation 2 关键参数）----
    huber_delta = getattr(config, 'huber_delta', None)
    if huber_delta is None:
        print("\nComputing Huber loss threshold τ from training data (80th percentile)...")
        huber_delta = compute_huber_threshold(
            model, train_loader, config.device,
            missing_ratio=0.5, percentile=80.0
        )
        print(f"  -> Huber delta τ = {huber_delta:.6f}")
    else:
        print(f"\nUsing configured Huber delta τ = {huber_delta:.6f}")

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_val_loss = float('inf')
    loss_history = {'train': [], 'val': []}

    print(f"\nStarting training for {config.num_epochs} epochs...")
    print(f"Loss: Huber (τ={huber_delta:.4f})  |  Early stopping: patience={config.early_stop_patience}, "
          f"threshold={config.early_stop_threshold}")

    for epoch in range(config.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, config.device,
                                 huber_delta=huber_delta)
        val_loss = validate(model, val_loader, config.device,
                            huber_delta=huber_delta)

        loss_history['train'].append(train_loss)
        loss_history['val'].append(val_loss)

        print(f"Epoch {epoch+1}/{config.num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'huber_delta': huber_delta,
                'scaler': scaler
            }, os.path.join(config.save_dir, 'diffusion_best.pth'))
            print(f"  -> Saved best model (val_loss: {val_loss:.6f})")

        # 早停: 最近 patience 个 epoch 内 train 与 val loss 变化均小于阈值
        if epoch >= config.early_stop_patience:
            recent_train = loss_history['train'][-config.early_stop_patience:]
            recent_val = loss_history['val'][-config.early_stop_patience:]
            train_change = max(recent_train) - min(recent_train)
            val_change = max(recent_val) - min(recent_val)
            if train_change < config.early_stop_threshold and val_change < config.early_stop_threshold:
                print(f"\nEarly stopping triggered!")
                print(f"  Train loss change: {train_change:.6f} < {config.early_stop_threshold}")
                print(f"  Val loss change:   {val_change:.6f} < {config.early_stop_threshold}")
                break

    print(f"\nTraining completed. Best val loss: {best_val_loss:.6f}")
    print(f"Model saved to: {os.path.join(config.save_dir, 'diffusion_best.pth')}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['sru', 'debutanizer'], required=True)
    args = parser.parse_args()

    if args.dataset == 'sru':
        from configs.config_sru import ConfigSRU as Config
    else:
        from configs.config_debutanizer import ConfigDebutanizer as Config

    config = Config()
    train_model(config)
