"""
扩张因果卷积 - 隐式多尺度特征提取

原始版本: 固定扩张率 {1,2,4,8,16,32}
专利改进: 基于时序自注意力机制的自适应扩张率 (Patent Innovation 1)

自适应扩张率计算流程（每层每时间步）:
    1. 注意力得分与权重: Q·K^T/sqrt(d_k) → softmax → α_{t,δ}
    2. 建议扩张率: d̃_t = Σ_δ α_{t,δ} · δ  (候选集{1,2,4,8,16,32}的加权平均)
    3. Sigmoid门控调制: d_t = d̃_t · (2·σ(W_g·x_t + b_g))，系数限于(0,2)
    4. 非整数扩张线性插值: y_t = (1-frac)·x[floor] + frac·x[ceil]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 原始固定扩张率模块（保留用于消融对照）
# ---------------------------------------------------------------------------

class DilatedCausalConv1d(nn.Module):
    """单层固定扩张率因果卷积（用于消融对照）"""

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=0
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        x = self.conv(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class DilatedCausalConv(nn.Module):
    """固定扩张率卷积网络（用于消融对照）"""

    def __init__(self, input_dim, hidden_dim=64, num_layers=4, kernel_size=3, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.conv_layers = nn.ModuleList([
            DilatedCausalConv1d(hidden_dim, hidden_dim, kernel_size=kernel_size,
                                dilation=2**i, dropout=dropout)
            for i in range(num_layers)
        ])
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.residual_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.5) for _ in range(num_layers)
        ])

    def forward(self, x):
        h = self.input_projection(x)
        h = h.transpose(1, 2)
        for i, conv_layer in enumerate(self.conv_layers):
            residual = h
            h = conv_layer(h)
            h = self.residual_weights[i] * h + (1 - self.residual_weights[i]) * residual
        h = h.transpose(1, 2)
        return self.output_projection(h)


# ---------------------------------------------------------------------------
# 专利核心创新: 自适应扩张率因果卷积（Patent Innovation 1）
# ---------------------------------------------------------------------------

class AdaptiveDilatedCausalConvLayer(nn.Module):
    """
    单层自适应扩张因果卷积。

    对每个时间步 t 通过时序自注意力动态计算扩张率 d_t，
    以线性插值实现非整数位置的特征采样，确保端到端可微。

    候选扩张集合: {1, 2, 4, 8, 16, 32} — 与原模型固定扩张率一致，
    但各时间步可基于数据实际变化周期自适应加权选取。
    """

    # 候选偏移距离集合，与原模型固定扩张率对应
    _CANDIDATE_OFFSETS = [1, 2, 4, 8, 16, 32]

    def __init__(self, channels, kernel_size=3, dropout=0.1, attn_dim=None):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size

        # 注意力投影维度，默认 channels//4，至少16
        if attn_dim is None:
            attn_dim = max(channels // 4, 16)
        self.attn_dim = attn_dim

        # ---- 时序自注意力参数 ----
        # 查询 / 键投影矩阵 W_Q, W_K（无偏置，与缩放点积注意力标准做法一致）
        self.W_Q = nn.Linear(channels, attn_dim, bias=False)
        self.W_K = nn.Linear(channels, attn_dim, bias=False)

        # Sigmoid 门控: d_t = d̃_t · (2·σ(W_g·x_t + b_g))
        self.gate_fc = nn.Linear(channels, 1)

        # ---- 卷积核参数（手动管理，支持自适应非整数扩张）----
        # weight: [out_channels, in_channels, kernel_size]
        self.weight = nn.Parameter(torch.empty(channels, channels, kernel_size))
        self.bias = nn.Parameter(torch.zeros(channels))

        # Kaiming 均匀初始化（与 nn.Conv1d 默认一致）
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = channels * kernel_size
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # 候选扩张率（注册为 buffer，不参与梯度，但随模型移动设备）
        candidates = torch.tensor(self._CANDIDATE_OFFSETS, dtype=torch.float)
        self.register_buffer('candidate_offsets', candidates)

    # ------------------------------------------------------------------
    # 步骤 1-3: 注意力得分 → 建议扩张率 → 门控调制
    # ------------------------------------------------------------------
    def _compute_adaptive_dilation(self, x_t):
        """
        利用时序自注意力为每个时间步计算自适应扩张率。

        Args:
            x_t: [B, L, C]  序列特征
        Returns:
            d:   [B, L]      每时间步自适应扩张率（连续实数，支持线性插值）
        """
        Q = self.W_Q(x_t)   # [B, L, attn_dim]
        K = self.W_K(x_t)   # [B, L, attn_dim]

        B, L, attn_dim = Q.shape
        max_delta = int(self.candidate_offsets.max().item())

        # ---- 因果移位: K_shifted[b, t, cand, :] = K[b, t-δ, :] ----
        # 对 K 在时间轴左侧填零 max_delta 步，实现所有候选偏移的批量移位
        K_T = K.transpose(1, 2)                         # [B, attn_dim, L]
        K_T_padded = F.pad(K_T, (max_delta, 0))         # [B, attn_dim, L+max_delta]
        K_padded = K_T_padded.transpose(1, 2)           # [B, L+max_delta, attn_dim]

        # 按候选偏移切片，堆叠为 [B, L, num_cands, attn_dim]
        K_shifted_list = [
            K_padded[:, max_delta - int(delta.item()) : max_delta - int(delta.item()) + L, :]
            for delta in self.candidate_offsets
        ]
        K_shifted = torch.stack(K_shifted_list, dim=2)  # [B, L, num_cands, attn_dim]

        # ---- 缩放点积注意力得分（防止高维内积导致 softmax 饱和）----
        scores = (Q.unsqueeze(2) * K_shifted).sum(-1) / (attn_dim ** 0.5)  # [B, L, num_cands]
        attn_weights = F.softmax(scores, dim=-1)                             # [B, L, num_cands]

        # ---- 建议扩张率: 候选偏移距离的注意力加权平均 ----
        suggested = (attn_weights * self.candidate_offsets.view(1, 1, -1)).sum(-1)  # [B, L]

        # ---- Sigmoid 门控调制: 系数 ∈ (0, 2)，支持放大/缩小/维持 ----
        gate = 2.0 * torch.sigmoid(self.gate_fc(x_t).squeeze(-1))   # [B, L]

        return suggested * gate   # [B, L]

    # ------------------------------------------------------------------
    # 步骤 4: 非整数位置线性插值
    # ------------------------------------------------------------------
    @staticmethod
    def _gather_with_interp(x, pos):
        """
        对连续实数位置 pos 进行线性插值采样（因果边界截断到0）。

        Args:
            x:   [B, C, L]  输入特征序列
            pos: [B, L]     采样位置（可为非整数，可能为负——截断到0）
        Returns:
            out: [B, C, L]
        """
        B, C, L = x.shape
        pos = pos.clamp(0.0, float(L - 1))

        pos_lo = pos.floor().long().clamp(0, L - 1)         # [B, L]
        pos_hi = (pos_lo + 1).clamp(0, L - 1)              # [B, L]
        frac = (pos - pos_lo.float()).unsqueeze(1)          # [B, 1, L]

        idx_lo = pos_lo.unsqueeze(1).expand(B, C, L)
        idx_hi = pos_hi.unsqueeze(1).expand(B, C, L)

        x_lo = x.gather(2, idx_lo)
        x_hi = x.gather(2, idx_hi)

        return (1.0 - frac) * x_lo + frac * x_hi

    # ------------------------------------------------------------------
    # 前向传播: 自适应扩张因果卷积
    # ------------------------------------------------------------------
    def forward(self, x):
        """
        Args:
            x: [B, C, L]  卷积格式输入
        Returns:
            out: [B, C, L]
        """
        B, C, L = x.shape
        x_t = x.transpose(1, 2)   # [B, L, C]

        # 计算每时间步自适应扩张率
        d = self._compute_adaptive_dilation(x_t)   # [B, L]

        t = torch.arange(L, dtype=torch.float, device=x.device)   # [L]

        # 逐核位置采样并加权求和:
        # y[t] = Σ_{i=0}^{k-1} W[:,i] · x[t - d_t · i]
        out = torch.zeros(B, C, L, device=x.device, dtype=x.dtype)
        for i in range(self.kernel_size):
            w_i = self.weight[:, :, i]   # [C_out, C_in]

            if i == 0:
                feat = x   # x[t - 0] = x[t]
            else:
                pos = t.unsqueeze(0) - d * float(i)   # [B, L]
                feat = self._gather_with_interp(x, pos)   # [B, C, L]

            # feat: [B, C_in, L]  →  [B, L, C_in] @ [C_in, C_out]  →  [B, C_out, L]
            out = out + torch.matmul(feat.transpose(1, 2), w_i.T).transpose(1, 2)

        out = out + self.bias.view(1, -1, 1)
        out = self.activation(out)
        out = self.dropout(out)

        return out


class AdaptiveDilatedCausalConv(nn.Module):
    """
    自适应扩张因果卷积网络 — 专利核心创新模块（隐式特征提取）。

    将原隐式特征提取模块中固定扩张率 {1,2,4,8,16,32} 替换为
    基于时序自注意力机制的自适应扩张率，使卷积感受野与各工业
    过程变量的实际变化周期动态匹配。

    接口与原 DilatedCausalConv 完全兼容，可直接替换。
    """

    def __init__(self, input_dim, hidden_dim=64, num_layers=6, kernel_size=3, dropout=0.1):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # 每层独立的自适应扩张卷积（均从全局候选集 {1,2,4,8,16,32} 自适应选取）
        self.conv_layers = nn.ModuleList([
            AdaptiveDilatedCausalConvLayer(
                channels=hidden_dim,
                kernel_size=kernel_size,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

        # 可学习残差连接权重（与原模型保持一致）
        self.residual_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.5) for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        Args:
            x: [B, L, input_dim]
        Returns:
            output: [B, L, hidden_dim]
        """
        h = self.input_projection(x)       # [B, L, hidden_dim]
        h = h.transpose(1, 2)              # [B, hidden_dim, L] for conv layers

        for i, conv_layer in enumerate(self.conv_layers):
            residual = h
            h = conv_layer(h)
            h = self.residual_weights[i] * h + (1 - self.residual_weights[i]) * residual

        h = h.transpose(1, 2)             # [B, L, hidden_dim]
        return self.output_projection(h)
