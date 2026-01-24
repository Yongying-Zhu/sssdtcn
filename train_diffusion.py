"""
训练扩散模型 - 直接预测数据（不预测噪声）
"""
import os
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from tqdm import tqdm
import sys

# 导入模型和数据加载器
from models.implicit_explicit_diffusion import ImplicitExplicitDiffusionModel
from data_loader import load_data, create_random_mask


def train_epoch(model, train_loader, optimizer, device, missing_ratio=0.5):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_data in tqdm(train_loader, desc="Training"):
        batch_data = batch_data.to(device)
        batch_size, seq_len, features = batch_data.shape

        # 创建随机mask
        mask, masked_data = create_random_mask(batch_data, missing_ratio)
        mask = mask.float().to(device)
        masked_data = masked_data.to(device)

        # 直接预测完整数据（t=0）
        timesteps = torch.zeros(batch_size, dtype=torch.long, device=device)
        predicted_data = model(masked_data, timesteps, mask)

        # 计算缺失位置的MSE loss
        mask_missing = (mask == 0)  # True = missing
        loss = ((predicted_data - batch_data) ** 2 * mask_missing.float()).sum() / mask_missing.sum()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate(model, val_loader, device, missing_ratio=0.5):
    """验证模型"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch_data in val_loader:
            batch_data = batch_data.to(device)
            batch_size, seq_len, features = batch_data.shape

            # 创建随机mask
            mask, masked_data = create_random_mask(batch_data, missing_ratio)
            mask = mask.float().to(device)
            masked_data = masked_data.to(device)

            # 直接预测完整数据（t=0）
            timesteps = torch.zeros(batch_size, dtype=torch.long, device=device)
            predicted_data = model(masked_data, timesteps, mask)

            # 计算缺失位置的MSE loss
            mask_missing = (mask == 0)
            loss = ((predicted_data - batch_data) ** 2 * mask_missing.float()).sum() / mask_missing.sum()

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def train_model(config):
    """训练主函数"""
    # 创建保存目录
    os.makedirs(config.save_dir, exist_ok=True)

    # 加载数据
    print(f"Loading {config.dataset_name} dataset...")
    train_loader, val_loader, test_loader, scaler, num_features = load_data(
        config.data_path,
        sequence_length=config.sequence_length,
        batch_size=config.batch_size,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio
    )

    # 创建模型
    print("Creating model...")
    model = ImplicitExplicitDiffusionModel(
        input_dim=num_features,
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        num_residual_layers=config.num_residual_layers,
        dropout=config.dropout
    ).to(config.device)

    # 优化器
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # 训练循环
    best_val_loss = float('inf')
    early_stop_counter = 0
    loss_history = {'train': [], 'val': []}

    print(f"\nStarting training for {config.num_epochs} epochs...")
    print(f"Early stopping: patience={config.early_stop_patience}, threshold={config.early_stop_threshold}")

    for epoch in range(config.num_epochs):
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, config.device)
        val_loss = validate(model, val_loader, config.device)

        loss_history['train'].append(train_loss)
        loss_history['val'].append(val_loss)

        print(f"Epoch {epoch+1}/{config.num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'scaler': scaler
            }, os.path.join(config.save_dir, 'diffusion_best.pth'))
            print(f"  -> Saved best model (val_loss: {val_loss:.6f})")

        # 早停检查：连续10个epoch，train和val loss变化都小于阈值
        if epoch >= config.early_stop_patience:
            # 检查最近10个epoch的loss变化
            recent_train_losses = loss_history['train'][-config.early_stop_patience:]
            recent_val_losses = loss_history['val'][-config.early_stop_patience:]

            train_loss_change = max(recent_train_losses) - min(recent_train_losses)
            val_loss_change = max(recent_val_losses) - min(recent_val_losses)

            if train_loss_change < config.early_stop_threshold and val_loss_change < config.early_stop_threshold:
                print(f"\nEarly stopping triggered!")
                print(f"  Train loss change: {train_loss_change:.6f} < {config.early_stop_threshold}")
                print(f"  Val loss change: {val_loss_change:.6f} < {config.early_stop_threshold}")
                break

    print(f"\nTraining completed. Best val loss: {best_val_loss:.6f}")
    print(f"Model saved to: {os.path.join(config.save_dir, 'diffusion_best.pth')}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['sru', 'debutanizer'], required=True)
    args = parser.parse_args()

    # 加载配置
    if args.dataset == 'sru':
        from configs.config_sru import ConfigSRU as Config
    else:
        from configs.config_debutanizer import ConfigDebutanizer as Config

    config = Config()
    train_model(config)
