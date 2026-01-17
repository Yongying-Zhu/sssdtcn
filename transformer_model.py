"""
Transformer for Time Series Imputation
基于注意力机制的时间序列插补
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]

class TransformerImputer(nn.Module):
    """Transformer插补模型"""
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=4, 
                 dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        # 输入投影（input_dim + 1 for mask indicator）
        self.input_proj = nn.Linear(input_dim + input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, input_dim)
        )
    
    def forward(self, x, mask):
        """
        Args:
            x: [batch, seq_len, input_dim]
            mask: [batch, seq_len, input_dim] (True = observed)
        Returns:
            x_recon: [batch, seq_len, input_dim]
        """
        # 将mask作为额外特征
        x_with_mask = torch.cat([x, mask.float()], dim=-1)  # [batch, seq_len, input_dim*2]
        
        # 输入投影
        h = self.input_proj(x_with_mask)  # [batch, seq_len, d_model]
        
        # 位置编码
        h = self.pos_encoder(h)
        
        # Transformer编码
        h = self.transformer_encoder(h)  # [batch, seq_len, d_model]
        
        # 输出投影
        x_recon = self.output_proj(h)  # [batch, seq_len, input_dim]
        
        return x_recon
    
    def compute_loss(self, x, mask):
        """
        计算重构损失
        """
        x_recon = self.forward(x, mask)
        
        # 只在观测位置计算损失
        loss = F.mse_loss(x_recon[mask], x[mask])
        
        return loss
    
    @torch.no_grad()
    def impute(self, x, mask):
        """插补缺失值"""
        x_recon = self.forward(x, mask)
        x_imputed = torch.where(mask, x, x_recon)
        return x_imputed
