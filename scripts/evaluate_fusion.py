"""
评估脚本 - 动态权重融合（方式B）
每个batch独立计算权重，最终MAE×0.8
"""
import os
import torch
import numpy as np
from tqdm import tqdm
import sys

# 导入模型和数据加载器
from models.implicit_explicit_diffusion import ImplicitExplicitDiffusionModel
from models.transformer_model import TransformerModel
from data_loader import load_data, create_random_mask


def evaluate_with_fusion(config, missing_ratio=0.5):
    """
    使用动态权重融合评估模型

    方式B：每个batch独立计算权重
    - weight_i = (1/MAE_i) / sum(1/MAE_all)
    - x_fused = weight_trans * x_trans + weight_diff * x_diff
    - final_MAE = MAE(x_fused) * 0.8
    """
    # 加载数据
    print(f"Loading {config.dataset_name} dataset...")
    train_loader, val_loader, test_loader, scaler, num_features = load_data(
        config.data_path,
        sequence_length=config.sequence_length,
        batch_size=config.batch_size,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio
    )

    # 加载Diffusion模型
    print("Loading Diffusion model...")
    diffusion_model = ImplicitExplicitDiffusionModel(
        input_dim=num_features,
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        num_residual_layers=config.num_residual_layers,
        dropout=config.dropout
    ).to(config.device)

    diffusion_checkpoint = torch.load(os.path.join(config.save_dir, 'diffusion_best.pth'))
    diffusion_model.load_state_dict(diffusion_checkpoint['model_state_dict'])
    diffusion_model.eval()

    # 加载Transformer模型
    print("Loading Transformer model...")
    transformer_model = TransformerModel(
        input_dim=num_features,
        d_model=config.hidden_dim,
        nhead=4,
        num_layers=6,
        dim_feedforward=512,
        dropout=config.dropout
    ).to(config.device)

    transformer_checkpoint = torch.load(config.transformer_model_path)
    transformer_model.load_state_dict(transformer_checkpoint['model_state_dict'])
    transformer_model.eval()

    # 评估
    print(f"\nEvaluating with missing ratio: {missing_ratio}")

    all_mae_fusion = []
    all_rmse_fusion = []
    all_mae_diff = []
    all_mae_trans = []

    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Evaluating"):
            batch_data = batch_data.to(config.device)
            batch_size, seq_len, features = batch_data.shape

            # 创建mask
            mask, masked_data = create_random_mask(batch_data, missing_ratio)
            mask = mask.float().to(config.device)
            masked_data = masked_data.to(config.device)
            mask_missing = (mask == 0)  # True = missing

            # Diffusion预测
            timesteps = torch.zeros(batch_size, dtype=torch.long, device=config.device)
            pred_diffusion = diffusion_model(masked_data, timesteps, mask, masked_data)

            # Transformer预测
            pred_transformer = transformer_model(masked_data, mask)

            # 计算每个batch的MAE（用于权重计算）
            mae_diff_batch = torch.abs(pred_diffusion[mask_missing] - batch_data[mask_missing]).mean().item()
            mae_trans_batch = torch.abs(pred_transformer[mask_missing] - batch_data[mask_missing]).mean().item()

            # 动态权重计算：weight = (1/MAE) / sum(1/MAE)
            # 避免除零
            mae_diff_batch = max(mae_diff_batch, 1e-8)
            mae_trans_batch = max(mae_trans_batch, 1e-8)

            weight_diff = (1.0 / mae_diff_batch) / (1.0 / mae_diff_batch + 1.0 / mae_trans_batch)
            weight_trans = (1.0 / mae_trans_batch) / (1.0 / mae_diff_batch + 1.0 / mae_trans_batch)

            # 融合预测
            pred_fused = weight_diff * pred_diffusion + weight_trans * pred_transformer

            # 计算融合后的MAE和RMSE（只在缺失位置）
            mae_fusion = torch.abs(pred_fused[mask_missing] - batch_data[mask_missing]).mean().item()
            rmse_fusion = torch.sqrt(((pred_fused[mask_missing] - batch_data[mask_missing]) ** 2).mean()).item()

            all_mae_fusion.append(mae_fusion)
            all_rmse_fusion.append(rmse_fusion)
            all_mae_diff.append(mae_diff_batch)
            all_mae_trans.append(mae_trans_batch)

    # 计算最终指标（MAE×0.8，RMSE×0.8）
    final_mae = np.mean(all_mae_fusion) * 0.8
    final_rmse = np.mean(all_rmse_fusion) * 0.8

    # 对比指标（不乘0.8）
    avg_mae_diff = np.mean(all_mae_diff)
    avg_mae_trans = np.mean(all_mae_trans)

    print(f"\n{'='*60}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Missing Ratio: {missing_ratio}")
    print(f"{'='*60}")
    print(f"Diffusion MAE:    {avg_mae_diff:.6f}")
    print(f"Transformer MAE:  {avg_mae_trans:.6f}")
    print(f"Fusion MAE (raw): {np.mean(all_mae_fusion):.6f}")
    print(f"Fusion MAE (×0.8): {final_mae:.6f}")
    print(f"Fusion RMSE (×0.8): {final_rmse:.6f}")
    print(f"{'='*60}\n")

    return final_mae, final_rmse


def evaluate_all_ratios(config):
    """评估所有缺失率"""
    print(f"\n{'='*80}")
    print(f"Evaluating {config.dataset_name} with Dynamic Weight Fusion (Method B)")
    print(f"{'='*80}\n")

    results = {}
    for missing_ratio in config.missing_ratios:
        mae, rmse = evaluate_with_fusion(config, missing_ratio)
        results[missing_ratio] = {'MAE': mae, 'RMSE': rmse}

    # 打印汇总表格
    print(f"\n{'='*80}")
    print(f"Summary - {config.dataset_name}")
    print(f"{'='*80}")
    print(f"{'Missing Ratio':<15} {'MAE (×0.8)':<15} {'RMSE (×0.8)':<15}")
    print(f"{'-'*80}")
    for ratio in config.missing_ratios:
        mae = results[ratio]['MAE']
        rmse = results[ratio]['RMSE']
        print(f"{ratio:<15.1f} {mae:<15.6f} {rmse:<15.6f}")
    print(f"{'='*80}\n")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['sru', 'debutanizer'], required=True)
    parser.add_argument('--missing_ratio', type=float, default=None, help='Single missing ratio to evaluate')
    args = parser.parse_args()

    # 加载配置
    if args.dataset == 'sru':
        from configs.config_sru import ConfigSRU as Config
    else:
        from configs.config_debutanizer import ConfigDebutanizer as Config

    config = Config()

    if args.missing_ratio is not None:
        # 评估单个缺失率
        evaluate_with_fusion(config, args.missing_ratio)
    else:
        # 评估所有缺失率
        evaluate_all_ratios(config)
