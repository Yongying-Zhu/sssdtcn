import torch
import torch.nn as nn
import numpy as np


class MaskEmbedding(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(2, embed_dim)
        self.ratio_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, mask):
        batch, seq_len, features = mask.shape
        mask_long = mask.long()
        mask_embed = self.embedding(mask_long)
        mask_embed = mask_embed.mean(dim=2)
        missing_ratio = mask.float().mean(dim=-1, keepdim=True)
        ratio_embed = self.ratio_mlp(missing_ratio)
        return mask_embed + ratio_embed


class PositionalMaskEncoding(nn.Module):
    def __init__(self, max_len=500, embed_dim=128):
        super().__init__()
        self.mask_embed = MaskEmbedding(embed_dim)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, mask):
        batch, seq_len, _ = mask.shape
        mask_embed = self.mask_embed(mask)
        pos_embed = self.pe[:seq_len, :].unsqueeze(0)
        return mask_embed + pos_embed
