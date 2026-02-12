"""
单尺度因果卷积 - 消融实验模块

与多尺度扩张卷积不同，此模块仅使用dilation=1，
用于验证多尺度特征提取的重要性。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleScaleCausalConv1d(nn.Module):
    """单层因果卷积（固定dilation=1）"""

    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.1):
        super().__init__()

        # 固定dilation=1
        self.dilation = 1
        self.padding = (kernel_size - 1) * self.dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=self.dilation,
            padding=0
        )

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        x = self.conv(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class SingleScaleCausalConv(nn.Module):
    """
    单尺度因果卷积网络 - 消融实验

    与DilatedCausalConv的区别：
        - 所有层都使用dilation=1（无扩张）
        - 无法捕捉多尺度时间模式
        - 感受野增长较慢
    """

    def __init__(self, input_dim, hidden_dim=64, num_layers=4, kernel_size=3, dropout=0.1):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # 多层单尺度卷积（所有dilation=1）
        self.conv_layers = nn.ModuleList([
            SingleScaleCausalConv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

        self.residual_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.5) for _ in range(num_layers)
        ])

    def forward(self, x):
        batch, length, input_dim = x.shape

        h = self.input_projection(x)
        h = h.transpose(1, 2)

        for i, conv_layer in enumerate(self.conv_layers):
            residual = h
            h = conv_layer(h)
            h = self.residual_weights[i] * h + (1 - self.residual_weights[i]) * residual

        h = h.transpose(1, 2)
        output = self.output_projection(h)

        return output
