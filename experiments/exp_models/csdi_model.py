"""
CSDI baseline model - Pure diffusion with simple MLP denoiser.

Represents the "pure diffusion" baseline without specialized temporal modules
(no S4 state-space layers, no dilated causal convolutions).

Architecture:
    mask embedding + input projection → time embedding + residual MLP blocks → output
"""

import torch
import torch.nn as nn
import math


class PositionalMaskEncoding(nn.Module):
    def __init__(self, max_len=500, embed_dim=128):
        super().__init__()
        self.mask_embed = nn.Embedding(2, embed_dim)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, mask):
        # mask: [B, T, D] float (1=observed, 0=missing)
        mask_int = mask.long()
        mask_emb = self.mask_embed(mask_int).mean(dim=-2)  # [B, T, embed_dim]
        return mask_emb + self.pe[:, :mask.size(1), :]


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


class CSDIModel(nn.Module):
    """
    CSDI-style pure diffusion model.

    Uses only simple MLP residual blocks as the denoising backbone,
    without any specialized temporal modules (no S4, no dilated conv).
    Represents the baseline "conditional score-based diffusion" approach.
    """

    def __init__(self, input_dim, hidden_dim=128, embedding_dim=256,
                 num_residual_layers=8, dropout=0.15):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.mask_embedding = PositionalMaskEncoding(max_len=500, embed_dim=hidden_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        # More residual blocks to compensate for no temporal module
        self.residual_layers = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_residual_layers)
        ])
        self.output_projection = nn.Linear(hidden_dim, input_dim)

    def _sinusoidal_embedding(self, timesteps, dim=256):
        half_dim = dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device).float() * -emb)
        emb = timesteps.unsqueeze(1).float() * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    def forward(self, x, timesteps, mask, observed):
        batch, seq_len, _ = x.shape
        x_cond = x * mask + observed

        mask_embed = self.mask_embedding(mask)
        x_proj = self.input_proj(x_cond)
        features = x_proj + mask_embed

        time_embed = self._sinusoidal_embedding(timesteps)
        time_embed = self.time_mlp(time_embed)
        time_embed = time_embed.unsqueeze(1).expand(-1, seq_len, -1)

        features = features + time_embed

        for residual_layer in self.residual_layers:
            features = residual_layer(features)

        return self.output_projection(features)
