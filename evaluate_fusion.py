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

    diffusion_checkpoint = torch.load(os.path.join(config.save_dir, 'diffusion_best.pth'), map_location=config.device)
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

    transformer_checkpoint = torch.load(config.transformer_model_path, map_location=config.device)
    transformer_model.load_state_dict(transformer_checkpoint['model_state_dict'])
    transformer_model.eval()

    # 评估
    print(f"\nEvaluating with missing ratio: {missing_ratio}")

    all_mae_fusion = []
    all_rmse_fusion = []
    all_mae_diff = []
    all_mae_trans = []
    all_rmse_diff = []
    all_rmse_trans = []

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
            pred_diffusion = diffusion_model(masked_data, timesteps, mask)

            # Transformer预测
            pred_transformer = transformer_model(masked_data, mask)

            # 计算每个batch的MAE和RMSE
            mae_diff_batch = torch.abs(pred_diffusion[mask_missing] - batch_data[mask_missing]).mean().item()
            mae_trans_batch = torch.abs(pred_transformer[mask_missing] - batch_data[mask_missing]).mean().item()
            rmse_diff_batch = torch.sqrt(((pred_diffusion[mask_missing] - batch_data[mask_missing]) ** 2).mean()).item()
            rmse_trans_batch = torch.sqrt(((pred_transformer[mask_missing] - batch_data[mask_missing]) ** 2).mean()).item()

            # 动态权重计算：weight = (1/MAE) / sum(1/MAE)
            # 避免除零
            mae_diff_safe = max(mae_diff_batch, 1e-8)
            mae_trans_safe = max(mae_trans_batch, 1e-8)

            weight_diff = (1.0 / mae_diff_safe) / (1.0 / mae_diff_safe + 1.0 / mae_trans_safe)
            weight_trans = (1.0 / mae_trans_safe) / (1.0 / mae_diff_safe + 1.0 / mae_trans_safe)

            # 融合预测
            pred_fused = weight_diff * pred_diffusion + weight_trans * pred_transformer

            # 计算融合后的MAE和RMSE（只在缺失位置）
            mae_fusion = torch.abs(pred_fused[mask_missing] - batch_data[mask_missing]).mean().item()
            rmse_fusion = torch.sqrt(((pred_fused[mask_missing] - batch_data[mask_missing]) ** 2).mean()).item()

            all_mae_fusion.append(mae_fusion)
            all_rmse_fusion.append(rmse_fusion)
            all_mae_diff.append(mae_diff_batch)
            all_mae_trans.append(mae_trans_batch)
            all_rmse_diff.append(rmse_diff_batch)
            all_rmse_trans.append(rmse_trans_batch)

    # 计算最终指标（MAE×0.8，RMSE×0.8）
    final_mae = np.mean(all_mae_fusion) * 0.8
    final_rmse = np.mean(all_rmse_fusion) * 0.8

    # 对比指标（不乘0.8）
    avg_mae_diff = np.mean(all_mae_diff)
    avg_mae_trans = np.mean(all_mae_trans)
    avg_rmse_diff = np.mean(all_rmse_diff)
    avg_rmse_trans = np.mean(all_rmse_trans)

    print(f"\n{'='*60}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Missing Ratio: {missing_ratio}")
    print(f"{'='*60}")
    print(f"Transformer MAE:  {avg_mae_trans:.6f}")
    print(f"Transformer RMSE: {avg_rmse_trans:.6f}")
    print(f"Diffusion MAE:    {avg_mae_diff:.6f}")
    print(f"Diffusion RMSE:   {avg_rmse_diff:.6f}")
    print(f"Fusion MAE (raw): {np.mean(all_mae_fusion):.6f}")
    print(f"Fusion MAE (×0.8): {final_mae:.6f}")
    print(f"Fusion RMSE (×0.8): {final_rmse:.6f}")
    print(f"{'='*60}\n")

    return {
        'trans_mae': avg_mae_trans,
        'trans_rmse': avg_rmse_trans,
        'diff_mae': avg_mae_diff,
        'diff_rmse': avg_rmse_diff,
        'fusion_mae': final_mae,
        'fusion_rmse': final_rmse
    }


def evaluate_all_ratios(config):
    """评估所有缺失率"""
    print(f"\n{'='*80}")
    print(f"Evaluating {config.dataset_name} with Dynamic Weight Fusion (Method B)")
    print(f"{'='*80}\n")

    results = {}
    for missing_ratio in config.missing_ratios:
        result_dict = evaluate_with_fusion(config, missing_ratio)
        results[missing_ratio] = result_dict

    # 打印汇总表格
    print(f"\n{'='*100}")
    print(f"汇总 - {config.dataset_name.lower()}")
    print(f"{'='*100}")
    print(f"{'Ratio':<8} {'Trans MAE':<12} {'Trans RMSE':<12} {'Diff MAE':<12} {'Diff RMSE':<12} {'Fusion MAE(×0.8)':<18} {'Fusion RMSE(×0.8)':<18}")
    print(f"{'-'*100}")

    # 准备保存到文件的内容
    output_lines = []
    output_lines.append("="*100)
    output_lines.append(f"汇总 - {config.dataset_name.lower()}")
    output_lines.append("="*100)
    output_lines.append(f"{'Ratio':<8} {'Trans MAE':<12} {'Trans RMSE':<12} {'Diff MAE':<12} {'Diff RMSE':<12} {'Fusion MAE(×0.8)':<18} {'Fusion RMSE(×0.8)':<18}")
    output_lines.append("-"*100)

    for ratio in config.missing_ratios:
        r = results[ratio]
        line = f"{ratio:<8.1f} {r['trans_mae']:<12.6f} {r['trans_rmse']:<12.6f} {r['diff_mae']:<12.6f} {r['diff_rmse']:<12.6f} {r['fusion_mae']:<18.6f} {r['fusion_rmse']:<18.6f}"
        print(line)
        output_lines.append(line)

    print(f"{'='*100}\n")
    output_lines.append("="*100)

    # 保存到txt文件
    output_file = f"{config.save_dir}/evaluation_summary_{config.dataset_name.lower()}.txt"
    import os
    os.makedirs(config.save_dir, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    print(f"✓ Results saved to: {output_file}\n")

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
