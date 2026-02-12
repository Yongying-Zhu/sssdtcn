"""
Transformer评估脚本
"""
import torch
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import sys
sys.path.append('/home/zhu/sssdtcn')
from universal_data_loader import get_dataloaders
from transformer_model import TransformerImputer

def evaluate_transformer(config, dataset_name):
    """评估Transformer"""
    print("="*80)
    print(f"{dataset_name} Dataset - Transformer Evaluation")
    print("="*80)
    print()
    
    # 加载数据
    _, _, test_loader = get_dataloaders(
        config['data_path'],
        config['sequence_length'],
        config['batch_size'],
        config['scaler_path']
    )
    
    with open(config['scaler_path'], 'rb') as f:
        scaler = pickle.load(f)
    
    # 加载模型
    model = TransformerImputer(
        input_dim=config['num_sensors'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout']
    ).to(config['device'])
    
    checkpoint = torch.load(f"{config['save_dir']}/best_model.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded (Epoch {checkpoint['epoch']})")
    print()
    
    # 评估不同缺失率
    mask_ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    results = []
    
    for mask_ratio in mask_ratios:
        all_mae, all_rmse = [], []
        
        with torch.no_grad():
            for x in tqdm(test_loader, desc=f'Mask {mask_ratio:.0%}', leave=False):
                x = x.to(config['device'])
                mask = torch.rand_like(x) > mask_ratio
                
                x_imputed = model.impute(x, mask)
                
                # 反标准化
                x_orig = scaler.inverse_transform(x.cpu().numpy().reshape(-1, config['num_sensors']))
                x_imp_orig = scaler.inverse_transform(x_imputed.cpu().numpy().reshape(-1, config['num_sensors']))
                mask_np = mask.cpu().numpy().reshape(-1, config['num_sensors'])
                
                missing_mask = ~mask_np
                if missing_mask.sum() > 0:
                    mae = np.abs(x_imp_orig[missing_mask] - x_orig[missing_mask]).mean()
                    rmse = np.sqrt(((x_imp_orig[missing_mask] - x_orig[missing_mask]) ** 2).mean())
                    all_mae.append(mae)
                    all_rmse.append(rmse)
        
        avg_mae = np.mean(all_mae)
        avg_rmse = np.mean(all_rmse)
        
        results.append({
            'Mask Ratio': f'{mask_ratio:.0%}',
            'MAE': avg_mae,
            'RMSE': avg_rmse
        })
        
        print(f"  {mask_ratio:.0%}: MAE={avg_mae:.4f}, RMSE={avg_rmse:.4f}")
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_file = f"{config['save_dir']}/evaluation_results.csv"
    results_df.to_csv(results_file, index=False)
    
    print()
    print("="*80)
    print("Results:")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)
    print(f"\n✓ Results saved: {results_file}")
    
    return results_df

if __name__ == "__main__":
    config = {
        'data_path': '/home/zhu/sssdtcn/hydraulic_1hz_clean.csv',
        'sequence_length': 60,
        'num_sensors': 7,
        'd_model': 128,
        'nhead': 8,
        'num_layers': 4,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'batch_size': 32,
        'scaler_path': '/home/zhu/sssdtcn/scaler_1hz.pkl',
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'save_dir': '/home/zhu/sssdtcn/baselines/transformer/checkpoints_1hz'
    }
    
    evaluate_transformer(config, "1Hz")
