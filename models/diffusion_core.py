"""
扩散模型核心 - DDPM (Denoising Diffusion Probabilistic Models)

核心思想:
    前向过程: 逐步向数据添加高斯噪声 (固定过程)
        x_t = sqrt(alpha_t) * x_0 + sqrt(1-alpha_t) * noise
    
    反向过程: 学习逐步去噪 (模型学习)
        x_{t-1} = denoise(x_t, t)
    
超参数调节:
    - num_diffusion_steps: 扩散步数 (50-1000)
        - 越大: 生成质量越好但速度越慢
        - 越小: 速度快但质量下降
    - beta_schedule: 噪声调度策略
        - linear: 线性增长
        - cosine: 余弦调度 (通常效果更好)
"""

import torch
import torch.nn as nn
import numpy as np

class DiffusionProcess:
    """扩散过程管理 - 前向加噪 + 反向去噪"""
    
    def __init__(self, num_steps=50, beta_start=0.0001, beta_end=0.02, device='cuda:0'):
        """
        参数:
            num_steps: 扩散总步数 T (调大提升质量但变慢)
            beta_start: 起始噪声方差 (通常0.0001)
            beta_end: 结束噪声方差 (通常0.02-0.05)
            device: 计算设备
        """
        self.num_steps = num_steps
        self.device = device
        
        # ========== 噪声调度 (Beta Schedule) ==========
        # 线性调度: beta_t 从 beta_start 线性增长到 beta_end
        self.betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        
        # Alpha相关量计算
        self.alphas = 1.0 - self.betas  # alpha_t = 1 - beta_t
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)  # alpha_bar_t = prod(alpha_1...alpha_t)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0], device=device), 
            self.alphas_cumprod[:-1]
        ])
        
        # 用于采样的预计算量
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 后验方差 (用于反向采样)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
    def q_sample(self, x_start, t, noise=None):
        """
        前向加噪过程: q(x_t | x_0)
        
        公式: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        
        输入:
            x_start: 原始数据 x_0 [batch, length, channels]
            t: 时间步 [batch]
            noise: 可选的噪声 (默认为标准高斯)
        输出:
            x_t: 加噪后的数据
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # 提取对应时间步的系数
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        # 加噪
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, model, x_t, t, mask, observed_data):
        """
        反向去噪一步: p(x_{t-1} | x_t)
        
        输入:
            model: 去噪模型
            x_t: 当前时间步数据
            t: 当前时间步
            mask: 缺失mask (1=缺失, 0=观测)
            observed_data: 观测数据
        输出:
            x_{t-1}: 去噪后的数据
        """
        batch_size = x_t.shape[0]
        
        # 模型预测噪声
        predicted_noise = model(x_t, t, mask, observed_data)
        
        # 提取系数
        beta_t = self.betas[t].view(-1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1)
        alpha_bar_t = self.alphas_cumprod[t].view(-1, 1, 1)
        
        # 预测 x_0
        x_0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
        
        # 计算均值
        mean = (x_t - beta_t / torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_t)
        
        # 添加噪声 (t>0时)
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            variance = self.posterior_variance[t].view(-1, 1, 1)
            x_t_minus_1 = mean + torch.sqrt(variance) * noise
        else:
            x_t_minus_1 = mean
        
        # 保留观测值不变
        x_t_minus_1 = x_t_minus_1 * mask + observed_data * (1 - mask)
        
        return x_t_minus_1
    
    def p_sample_loop(self, model, shape, mask, observed_data):
        """
        完整反向采样循环: 从 x_T 到 x_0
        
        输入:
            model: 去噪模型
            shape: 数据形状 [batch, length, channels]
            mask: 缺失mask
            observed_data: 观测数据
        输出:
            x_0: 重建的完整数据
        """
        batch_size = shape[0]
        device = self.device
        
        # 从纯噪声开始
        x_t = torch.randn(shape, device=device)
        # 初始化时保留观测值
        x_t = x_t * mask + observed_data * (1 - mask)
        
        # 逐步去噪 T -> 0
        for i in reversed(range(self.num_steps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, t, mask, observed_data)
        
        return x_t

