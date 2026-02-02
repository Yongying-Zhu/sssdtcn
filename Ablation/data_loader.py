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
    def __init__(self, data, sequence_length=60):
        self.data = torch.FloatTensor(data)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length + 1

    def __getitem__(self, idx):
        seq = self.data[idx:idx + self.sequence_length]
        return seq

def load_data(data_path, sequence_length=60, batch_size=32, train_ratio=0.7, val_ratio=0.1):
    data = pd.read_csv(data_path, sep='\s+', header=None).values
    print(f"Loaded data shape: {data.shape}")

    num_features = data.shape[1]

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    n_samples = len(data_scaled)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    train_data = data_scaled[:train_end]
    val_data = data_scaled[train_end:val_end]
    test_data = data_scaled[val_end:]

    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

    train_dataset = TimeSeriesDataset(train_data, sequence_length)
    val_dataset = TimeSeriesDataset(val_data, sequence_length)
    test_dataset = TimeSeriesDataset(test_data, sequence_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, scaler, num_features

def create_random_mask(data, missing_ratio=0.5):
    batch, seq_len, features = data.shape
    mask = torch.rand_like(data) > missing_ratio  # True = observed
    masked_data = data * mask.float()
    return mask, masked_data
