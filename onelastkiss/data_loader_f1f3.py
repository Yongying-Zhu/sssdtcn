import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = torch.FloatTensor(sequences)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]

def load_data_f1f3(data_path, sequence_length=60, batch_size=32, train_ratio=0.7, val_ratio=0.1):
    data = np.loadtxt(data_path, skiprows=1)
    print(f"Original data shape: {data.shape}")
    
    data = data[:, [0, 2]]
    print(f"Selected Feature 1 & 3, shape: {data.shape}")
    
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    print("Data normalized")
    
    n_samples = len(data)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    def create_sequences(data, seq_len):
        sequences = []
        for i in range(len(data) - seq_len + 1):
            sequences.append(data[i:i+seq_len])
        return np.array(sequences)
    
    train_seq = create_sequences(train_data, sequence_length)
    val_seq = create_sequences(val_data, sequence_length)
    test_seq = create_sequences(test_data, sequence_length)
    
    print(f"Sequences - Train: {len(train_seq)}, Val: {len(val_seq)}, Test: {len(test_seq)}")
    
    train_loader = DataLoader(TimeSeriesDataset(train_seq), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(TimeSeriesDataset(val_seq), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(TimeSeriesDataset(test_seq), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    os.makedirs('data', exist_ok=True)
    with open('data/scaler_sru_f1f3.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved: data/scaler_sru_f1f3.pkl")
    
    return train_loader, val_loader, test_loader, scaler, 2

def create_random_mask(data, missing_ratio):
    batch_size, seq_len, features = data.shape
    mask = torch.ones_like(data)
    num_missing = int(batch_size * seq_len * features * missing_ratio)
    
    indices = torch.randperm(batch_size * seq_len * features)[:num_missing]
    batch_idx = indices // (seq_len * features)
    seq_idx = (indices % (seq_len * features)) // features
    feat_idx = indices % features
    
    mask[batch_idx, seq_idx, feat_idx] = 0
    masked_data = data * mask
    
    return mask, masked_data
