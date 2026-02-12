"""
自定义M-RNN实现（当PyPOTS不支持时使用）
Multi-directional RNN for Time Series Imputation
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os
import sys
sys.path.append('/home/zhu/sssdtcn/compare')
from data_utils import load_sru_data, load_debutanizer_data, prepare_data, create_missing_mask

class MRNN(nn.Module):
    """Multi-directional RNN"""
    def __init__(self, n_features, hidden_size=128, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        
        # Forward RNN
        self.rnn_forward = nn.GRU(n_features * 2, hidden_size, batch_first=True, dropout=dropout)
        # Backward RNN
        self.rnn_backward = nn.GRU(n_features * 2, hidden_size, batch_first=True, dropout=dropout)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_features)
        )
    
    def forward(self, x, mask):
        """
        x: [batch, seq_len, features] - 带缺失的数据（缺失位置为0）
        mask: [batch, seq_len, features] - True=observed
        """
        batch, seq_len, features = x.shape
        
        # 拼接数据和mask
        x_with_mask = torch.cat([x, mask.float()], dim=-1)  # [batch, seq_len, features*2]
        
        # Forward pass
        h_forward, _ = self.rnn_forward(x_with_mask)  # [batch, seq_len, hidden]
        
        # Backward pass
        x_reversed = torch.flip(x_with_mask, dims=[1])
        h_backward, _ = self.rnn_backward(x_reversed)
        h_backward = torch.flip(h_backward, dims=[1])  # [batch, seq_len, hidden]
        
        # Concatenate
        h_combined = torch.cat([h_forward, h_backward], dim=-1)  # [batch, seq_len, hidden*2]
        
        # Output
        output = self.output_layer(h_combined)  # [batch, seq_len, features]
        
        return output

def train_mrnn_custom(dataset_name, data_path, selected_features=None, epochs=300):
    """训练自定义M-RNN"""
    print(f"\n{'='*60}")
    print(f"Training Custom M-RNN on {dataset_name.upper()}")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    if dataset_name == 'sru':
        data = load_sru_data(data_path, selected_features)
    else:
        data = load_debutanizer_data(data_path)
    
    n_features = data.shape[1]
    
    # 准备数据
    train_data, val_data, test_data, scaler = prepare_data(data)
    
    # 转换为tensor
    train_tensor = torch.FloatTensor(train_data)
    val_tensor = torch.FloatTensor(val_data)
    
    # 创建DataLoader
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor), batch_size=32, shuffle=False)
    
    # 创建模型
    model = MRNN(n_features, hidden_size=128, dropout=0.15).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            x = batch[0].to(device)
            
            # 随机生成缺失mask
            missing_ratio = np.random.uniform(0.2, 0.8)
            mask = torch.rand_like(x) > missing_ratio
            masked_x = x * mask.float()
            
            # Forward
            output = model(masked_x, mask)
            
            # Loss只在缺失位置计算
            loss_mask = ~mask
            if loss_mask.sum() > 0:
                loss = ((output - x) ** 2 * loss_mask.float()).sum() / loss_mask.sum()
            else:
                loss = torch.tensor(0.0, device=device)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                mask = torch.rand_like(x) > 0.5
                masked_x = x * mask.float()
                output = model(masked_x, mask)
                loss_mask = ~mask
                if loss_mask.sum() > 0:
                    loss = ((output - x) ** 2 * loss_mask.float()).sum() / loss_mask.sum()
                    val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            model_path = f'/home/zhu/sssdtcn/compare/models/mrnn_{dataset_name}_model.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'n_features': n_features,
                'hidden_size': 128
            }, model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    print(f"M-RNN training completed. Best val loss: {best_val_loss:.6f}")
    print(f"Model saved to /home/zhu/sssdtcn/compare/models/mrnn_{dataset_name}_model.pth")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['sru', 'debutanizer'])
    parser.add_argument('--epochs', type=int, default=300)
    args = parser.parse_args()
    
    if args.dataset == 'sru':
        train_mrnn_custom('sru', '/home/zhu/sssdtcn/SRU_data.txt', [0, 2], args.epochs)
    else:
        train_mrnn_custom('debutanizer', '/home/zhu/sssdtcn/debutanizer_data.txt', None, args.epochs)
