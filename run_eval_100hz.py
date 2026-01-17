"""评估Transformer - 100HZ（使用A10显卡 cuda:1）"""
from evaluate_transformer import evaluate_transformer
import torch

config = {
    'data_path': '/home/zhu/sssdtcn/hydraulic_100hz_clean.csv',
    'sequence_length': 6000,
    'num_sensors': 7,
    'd_model': 256,
    'nhead': 8,
    'num_layers': 6,
    'dim_feedforward': 1024,
    'dropout': 0.1,
    'batch_size': 16,
    'scaler_path': '/home/zhu/sssdtcn/scaler_100hz.pkl',
    'device': torch.device('cuda:1'),  # 强制使用A10显卡
    'save_dir': '/home/zhu/sssdtcn/baselines/transformer/checkpoints_100hz'
}

evaluate_transformer(config, "100HZ")
