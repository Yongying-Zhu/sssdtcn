"""
Implicit-only diffusion model - ablation variant.

Removes the S4 explicit module from the full SSSDTCN model.
Uses only dilated causal convolutions (implicit temporal modeling).
Used in the ablation study to evaluate the contribution of the explicit branch.
"""

import torch
import torch.nn as nn
import math
import sys
import os

# Allow importing DilatedCausalConv from the main models directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from models.dilated_causal_conv import DilatedCausalConv
from models.mask_embedding import PositionalMaskEncoding


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


class ImplicitOnlyDiffusionModel(nn.Module):
    """
    Implicit-only diffusion model (ablation variant).

    Removes the S4 explicit module from the full model.
    Only uses DilatedCausalConv for temporal feature extraction.
    Counterpart to ExplicitOnlyDiffusionModel.
    """

    def __init__(self, input_dim, hidden_dim=128, embedding_dim=256,
                 num_residual_layers=6, dropout=0.15, num_conv_layers=8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.mask_embedding = PositionalMaskEncoding(max_len=500, embed_dim=hidden_dim)

        # More conv layers to compensate for missing explicit module.
        # DilatedCausalConv receives already-projected hidden_dim features.
        self.implicit_module = DilatedCausalConv(
            hidden_dim, hidden_dim=hidden_dim,
            num_layers=num_conv_layers, kernel_size=5, dropout=dropout
        )

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
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device).float() * -emb)
        emb = timesteps.unsqueeze(1).float() * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    def forward(self, x, timesteps, mask, observed):
        batch, seq_len, _ = x.shape
        x_cond = x * mask + observed

        mask_embed = self.mask_embedding(mask)
        x_proj = self.input_proj(x_cond)
        x_with_mask = x_proj + mask_embed

        # Only implicit module (no S4)
        features = self.implicit_module(x_with_mask)

        time_embed = self._sinusoidal_embedding(timesteps)
        time_embed = self.time_mlp(time_embed)
        time_embed = time_embed.unsqueeze(1).expand(-1, seq_len, -1)

        features = features + time_embed

        for residual_layer in self.residual_layers:
            features = residual_layer(features)

        return self.output_projection(features)
