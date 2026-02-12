"""
数据处理工具模块 - 与原项目保持一致
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_sru_data(data_path, selected_features=[0, 2]):
    """
    加载SRU数据集，只保留Feature 1和Feature 3
    """
    data = pd.read_csv(data_path, sep=r'\s+', header=None, skiprows=4).values
    data = data[~np.isnan(data).any(axis=1)]
    data = data[:, selected_features]
    print(f"SRU data loaded: {data.shape}")
    return data

def load_debutanizer_data(data_path):
    """
    加载Debutanizer数据集，保留所有特征
    """
    data = pd.read_csv(data_path, sep=r'\s+', header=None, skiprows=5).values
    data = data[~np.isnan(data).any(axis=1)]
    print(f"Debutanizer data loaded: {data.shape}")
    return data

def prepare_data(data, sequence_length=60, train_ratio=0.7, val_ratio=0.1):
    """
    准备数据：标准化 + 划分 + 创建序列窗口
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    n_samples = len(data_scaled)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    train_data = data_scaled[:train_end]
    val_data = data_scaled[train_end:val_end]
    test_data = data_scaled[val_end:]
    
    def create_sequences(data, seq_len):
        sequences = []
        for i in range(len(data) - seq_len + 1):
            sequences.append(data[i:i + seq_len])
        return np.array(sequences)
    
    train_sequences = create_sequences(train_data, sequence_length)
    val_sequences = create_sequences(val_data, sequence_length)
    test_sequences = create_sequences(test_data, sequence_length)
    
    print(f"Train sequences: {train_sequences.shape}")
    print(f"Val sequences: {val_sequences.shape}")
    print(f"Test sequences: {test_sequences.shape}")
    
    return train_sequences, val_sequences, test_sequences, scaler

def create_missing_mask(data, missing_ratio=0.5):
    """
    创建随机缺失mask
    True=observed, False=missing
    """
    mask = np.random.rand(*data.shape) > missing_ratio
    return mask

def apply_mask(data, mask):
    """
    应用mask，缺失位置设为NaN
    """
    masked_data = data.copy()
    masked_data[~mask] = np.nan
    return masked_data
