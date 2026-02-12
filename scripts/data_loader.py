"""
Universal Data Loader for SRU and Debutanizer datasets
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pickle

class TimeSeriesDataset(Dataset):
    """时间序列数据集"""
    def __init__(self, data, sequence_length=60):
        """
        Args:
            data: numpy array [N, D]
            sequence_length: 序列长度
        """
        self.data = torch.FloatTensor(data)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length + 1

    def __getitem__(self, idx):
        """返回一个序列窗口"""
        seq = self.data[idx:idx + self.sequence_length]
        return seq

def load_data(data_path, sequence_length=60, batch_size=32, train_ratio=0.7, val_ratio=0.1):
    """
    加载TXT数据并创建DataLoader

    Args:
        data_path: 数据文件路径 (SRU_data.txt or debutanizer_data.txt)
        sequence_length: 序列长度
        batch_size: batch大小
        train_ratio: 训练集比例
        val_ratio: 验证集比例

    Returns:
        train_loader, val_loader, test_loader, scaler, num_features
    """
    # 读取数据
    data = pd.read_csv(data_path, sep='\s+', header=None).values
    print(f"Loaded data shape: {data.shape}")

    num_features = data.shape[1]

    # 标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # 划分数据集
    n_samples = len(data_scaled)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    train_data = data_scaled[:train_end]
    val_data = data_scaled[train_end:val_end]
    test_data = data_scaled[val_end:]

    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

    # 创建Dataset
    train_dataset = TimeSeriesDataset(train_data, sequence_length)
    val_dataset = TimeSeriesDataset(val_data, sequence_length)
    test_dataset = TimeSeriesDataset(test_data, sequence_length)

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, scaler, num_features

def create_random_mask(data, missing_ratio=0.5):
    """
    创建随机缺失mask

    Args:
        data: [batch, seq_len, features]
        missing_ratio: 缺失比例

    Returns:
        mask: [batch, seq_len, features] (True=observed, False=missing)
        masked_data: 观测到的数据（缺失位置为0）
    """
    batch, seq_len, features = data.shape
    mask = torch.rand_like(data) > missing_ratio  # True = observed
    masked_data = data * mask.float()
    return mask, masked_data
