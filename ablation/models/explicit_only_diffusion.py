"""
仅显式特征扩散模型 - 消融实验

与完整模型的区别：
    - 移除了扩张因果卷积（隐式特征提取模块）
    - 仅使用S4Layer进行显式建模
    - 用于验证隐式-显式特征融合的重要性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .s4_layer import S4Layer
from .mask_embedding import PositionalMaskEncoding


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        return self.layers(x) + x


class ExplicitOnlyDiffusionModel(nn.Module):
    """
    仅显式特征扩散模型

    消融设置：
        - 无隐式模块（移除扩张因果卷积）
        - 仅使用S4Layer进行显式序列建模
    """

    def __init__(self, input_dim, hidden_dim=128, embedding_dim=256,
                 num_residual_layers=6, dropout=0.15, num_s4_layers=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.mask_embedding = PositionalMaskEncoding(max_len=500, embed_dim=hidden_dim)

        # 仅使用S4Layer（增加层数以补偿移除的隐式模块）
        self.explicit_module = nn.ModuleList([
            S4Layer(hidden_dim, d_state=256, num_heads=2, dropout=dropout)
            for _ in range(num_s4_layers)
        ])

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.residual_layers = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_residual_layers)
        ])
        self.output_projection = nn.Linear(hidden_dim, input_dim)

    def _sinusoidal_embedding(self, timesteps, dim=256):
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0, device=timesteps.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device).float() * -emb)
        emb = timesteps.unsqueeze(1).float() * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    def forward(self, x, timesteps, mask, observed):
        batch, seq_len, _ = x.shape
        x_cond = x * mask + observed

        mask_embed = self.mask_embedding(mask)
        x_proj = self.input_proj(x_cond)
        x_with_mask = x_proj + mask_embed

        # 仅使用显式特征（S4Layer）
        features = x_with_mask
        for s4_layer in self.explicit_module:
            features = s4_layer(features)

        time_embed = self._sinusoidal_embedding(timesteps)
        time_embed = self.time_mlp(time_embed)
        time_embed = time_embed.unsqueeze(1).expand(-1, seq_len, -1)

        features = features + time_embed

        for residual_layer in self.residual_layers:
            features = residual_layer(features)

        return self.output_projection(features)
