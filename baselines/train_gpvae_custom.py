"""
自定义GP-VAE实现（当PyPOTS不支持时使用）
Gaussian Process VAE for Time Series Imputation
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os
import sys
sys.path.append('/home/zhu/sssdtcn/compare')
from data_utils import load_sru_data, load_debutanizer_data, prepare_data, create_missing_mask

class GPVAE(nn.Module):
    """Gaussian Process VAE for time series imputation"""
    def __init__(self, n_features, seq_len=60, latent_dim=32, hidden_dim=128):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_features * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.encoder_rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        
        # Decoder
        self.decoder_rnn = nn.GRU(latent_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features)
        )
    
    def encode(self, x, mask):
        # x: [batch, seq_len, features]
        # mask: [batch, seq_len, features]
        x_with_mask = torch.cat([x, mask.float()], dim=-1)  # [batch, seq_len, features*2]
        h = self.encoder(x_with_mask)  # [batch, seq_len, hidden]
        h, _ = self.encoder_rnn(h)  # [batch, seq_len, hidden*2]
        
        # 使用最后时间步的隐状态
        h_last = h[:, -1, :]  # [batch, hidden*2]
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        # z: [batch, latent_dim]
        batch = z.shape[0]
        # 扩展到序列长度
        z_expanded = z.unsqueeze(1).repeat(1, self.seq_len, 1)  # [batch, seq_len, latent_dim]
        h, _ = self.decoder_rnn(z_expanded)  # [batch, seq_len, hidden*2]
        output = self.decoder(h)  # [batch, seq_len, features]
        return output
    
    def forward(self, x, mask):
        mu, logvar = self.encode(x, mask)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def vae_loss(recon, x, mu, logvar, mask):
    """VAE loss: reconstruction + KL divergence"""
    # 只在缺失位置计算重建误差
    loss_mask = ~mask
    if loss_mask.sum() > 0:
        recon_loss = ((recon - x) ** 2 * loss_mask.float()).sum() / loss_mask.sum()
    else:
        recon_loss = torch.tensor(0.0, device=x.device)
    
    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + 0.01 * kl_loss, recon_loss, kl_loss

def train_gpvae_custom(dataset_name, data_path, selected_features=None, epochs=300):
    """训练自定义GP-VAE"""
    print(f"\n{'='*60}")
    print(f"Training Custom GP-VAE on {dataset_name.upper()}")
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
    model = GPVAE(n_features, seq_len=60, latent_dim=32, hidden_dim=128).to(device)
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
            recon, mu, logvar = model(masked_x, mask)
            
            # Loss
            loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, mask)
            
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
                recon, mu, logvar = model(masked_x, mask)
                loss, _, _ = vae_loss(recon, x, mu, logvar, mask)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            model_path = f'/home/zhu/sssdtcn/compare/models/gpvae_{dataset_name}_model.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'n_features': n_features,
                'latent_dim': 32,
                'hidden_dim': 128
            }, model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    print(f"GP-VAE training completed. Best val loss: {best_val_loss:.6f}")
    print(f"Model saved to /home/zhu/sssdtcn/compare/models/gpvae_{dataset_name}_model.pth")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['sru', 'debutanizer'])
    parser.add_argument('--epochs', type=int, default=300)
    args = parser.parse_args()
    
    if args.dataset == 'sru':
        train_gpvae_custom('sru', '/home/zhu/sssdtcn/SRU_data.txt', [0, 2], args.epochs)
    else:
        train_gpvae_custom('debutanizer', '/home/zhu/sssdtcn/debutanizer_data.txt', None, args.epochs)
