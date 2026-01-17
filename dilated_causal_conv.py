"""
扩张因果卷积 - 隐式多尺度特征提取

核心思想:
    - 因果卷积: 只依赖历史数据，不泄露未来信息
    - 扩张卷积: 指数增长感受野，捕捉不同时间尺度模式
    
超参数调节:
    - dilation_rates: 扩张率列表 (如[1,2,4,8])
        - 较小值捕捉短期模式 (趋势、周期)
        - 较大值捕捉长期模式 (季节性)
    - kernel_size: 卷积核大小 (默认3，调大增加感受野)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedCausalConv1d(nn.Module):
    """单层扩张因果卷积"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        """
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小 (建议奇数，如3、5、7)
            dilation: 扩张率 (1=标准卷积，2=间隔1采样，4=间隔3采样)
            dropout: Dropout率
        """
        super().__init__()
        
        # 计算padding确保因果性 (不看未来)
        # padding = (kernel_size - 1) * dilation
        self.padding = (kernel_size - 1) * dilation
        
        # 卷积层
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=0  # 手动padding确保因果性
        )
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # GELU激活函数 (比ReLU更平滑)
        
    def forward(self, x):
        """
        前向传播
        
        输入: x [batch, channels, length]
        输出: [batch, channels, length]
        """
        # 左侧padding (只padding历史)
        x = F.pad(x, (self.padding, 0))
        
        # 卷积 + 激活 + Dropout
        x = self.conv(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x


class ImplicitExtractionModule(nn.Module):
    """
    隐式特征提取模块 - 多尺度扩张卷积网络
    
    架构设计思路:
        使用多层不同扩张率的卷积层，逐层增大感受野
        - Layer 1 (dilation=1): 捕捉邻近点关系
        - Layer 2 (dilation=2): 捕捉2步依赖
        - Layer 3 (dilation=4): 捕捉4步依赖
        - Layer 4 (dilation=8): 捕捉8步依赖
        
    超参数调优建议:
        1. 增加层数: 捕捉更长期依赖
        2. 调整扩张率序列: [1,2,4,8] → [1,3,9,27] 更激进
        3. 调整hidden_dim: 增大特征容量
    """
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=4, kernel_size=3, dropout=0.1):
        """
        参数:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度 (调大可提升表达能力)
            num_layers: 卷积层数量 (调大可捕捉更长期依赖)
            kernel_size: 卷积核大小
            dropout: Dropout率
        """
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 多层扩张卷积
        # 扩张率呈指数增长: 1, 2, 4, 8, ...
        self.conv_layers = nn.ModuleList([
            DilatedCausalConv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                dilation=2**i,  # 指数增长扩张率
                dropout=dropout
            ) for i in range(num_layers)
        ])
        
        # 输出投影
        self.output_projection = nn.Linear(hidden_dim, input_dim)
        
        # 残差连接的权重
        self.residual_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.5) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        """
        前向传播
        
        输入: x [batch, length, input_dim]
        输出: [batch, length, input_dim]
        
        处理流程:
            1. 投影到hidden_dim维度
            2. 多层扩张卷积 (带残差连接)
            3. 投影回input_dim
        """
        batch, length, input_dim = x.shape
        
        # 投影到隐藏维度
        h = self.input_projection(x)  # [batch, length, hidden_dim]
        
        # 转换为卷积格式 [batch, channels, length]
        h = h.transpose(1, 2)
        
        # 多层扩张卷积 (带加权残差)
        for i, conv_layer in enumerate(self.conv_layers):
            residual = h
            h = conv_layer(h)
            # 加权残差连接 (可学习的权重)
            h = self.residual_weights[i] * h + (1 - self.residual_weights[i]) * residual
        
        # 转换回 [batch, length, hidden_dim]
        h = h.transpose(1, 2)
        
        # 投影回输入维度
        output = self.output_projection(h)
        
        return output

