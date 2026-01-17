"""训练Transformer - 10HZ"""
from train_transformer import train_transformer
import torch

config = {
    'data_path': '/home/zhu/sssdtcn/hydraulic_10hz_clean.csv',
    'sequence_length': 600,
    'num_sensors': 2,
    'd_model': 128,
    'nhead': 8,
    'num_layers': 4,
    'dim_feedforward': 512,
    'dropout': 0.1,
    'batch_size': 32 if 600 < 1000 else 16,
    'learning_rate': 1e-4,
    'num_epochs': 80,
    'scaler_path': '/home/zhu/sssdtcn/scaler_10hz.pkl',
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'save_dir': '/home/zhu/sssdtcn/baselines/transformer/checkpoints_10hz'
}

train_transformer(config, "10HZ")
