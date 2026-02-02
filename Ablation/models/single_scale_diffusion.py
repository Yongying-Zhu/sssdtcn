"""
单尺度 + 显式特征扩散模型 - 消融实验

与完整模型的区别：
    - 使用单尺度因果卷积（dilation=1）代替多尺度扩张卷积
    - 保留S4Layer显式建模模块
    - 用于验证多尺度特征提取的重要性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .single_scale_conv import SingleScaleCausalConv
from .s4_layer import S4Layer
from .mask_embedding import PositionalMaskEncoding


class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())

    def forward(self, x1, x2):
        concat = torch.cat([x1, x2], dim=-1)
        gate = self.gate(concat)
        return gate * x1 + (1 - gate) * x2


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


class SingleScaleDiffusionModel(nn.Module):
    """
    单尺度 + 显式特征扩散模型

    消融设置：
        - implicit_module: 单尺度因果卷积（所有层dilation=1）
        - explicit_module: S4Layer（保持不变）
    """

    def __init__(self, input_dim, hidden_dim=128, embedding_dim=256,
                 num_residual_layers=6, dropout=0.15):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.mask_embedding = PositionalMaskEncoding(max_len=500, embed_dim=hidden_dim)

        # 单尺度因果卷积（替代多尺度扩张卷积）
        self.implicit_module = SingleScaleCausalConv(
            input_dim, hidden_dim=hidden_dim, num_layers=6, kernel_size=5, dropout=dropout
        )

        # S4Layer显式建模（保持不变）
        self.explicit_module = nn.ModuleList([
            S4Layer(hidden_dim, d_state=256, num_heads=2, dropout=dropout) for _ in range(2)
        ])

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.gated_fusion = GatedFusion(hidden_dim)
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

        # 单尺度隐式特征
        implicit_features = self.implicit_module(x_with_mask)

        # 显式特征（S4Layer）
        explicit_features = x_with_mask
        for s4_layer in self.explicit_module:
            explicit_features = s4_layer(explicit_features)

        time_embed = self._sinusoidal_embedding(timesteps)
        time_embed = self.time_mlp(time_embed)
        time_embed = time_embed.unsqueeze(1).expand(-1, seq_len, -1)

        fused = self.gated_fusion(implicit_features, explicit_features)
        fused = fused + time_embed

        for residual_layer in self.residual_layers:
            fused = residual_layer(fused)

        return self.output_projection(fused)
