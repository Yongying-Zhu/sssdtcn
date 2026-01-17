"""训练Transformer - 1HZ"""
from train_transformer import train_transformer
import torch

config = {
    'data_path': '/home/zhu/sssdtcn/hydraulic_1hz_clean.csv',
    'sequence_length': 60,
    'num_sensors': 7,
    'd_model': 128,
    'nhead': 8,
    'num_layers': 4,
    'dim_feedforward': 512,
    'dropout': 0.1,
    'batch_size': 32 if 60 < 1000 else 16,
    'learning_rate': 1e-4,
    'num_epochs': 80,
    'scaler_path': '/home/zhu/sssdtcn/scaler_1hz.pkl',
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'save_dir': '/home/zhu/sssdtcn/baselines/transformer/checkpoints_1hz'
}

train_transformer(config, "1HZ")
