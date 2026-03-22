import torch
import torch.nn as nn
import torch.nn.functional as F


class S4Layer(nn.Module):
    """
    Structured State Space layer with diagonal state matrix.

    Diagonal recurrence:  h_t = lambda * h_{t-1} + B * x_t  (element-wise)
    Output:               y_t = C @ h_t

    Implemented efficiently via causal depthwise convolution:
      h_t[i] = sum_{k=0}^{t} lambda_i^(t-k) * B_i * x_k[i]
    The exponential decay kernels are built on-the-fly and applied
    with F.conv1d (grouped), eliminating the Python time-step loop.
    """

    def __init__(self, d_model, d_state=256, dropout=0.1, num_heads=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.num_heads = num_heads
        self.d_head = d_state // num_heads

        # Learnable diagonal eigenvalues (mapped to [0,1) via sigmoid)
        self.log_neg_lambdas = nn.Parameter(torch.randn(num_heads, self.d_head))
        # Input scaling B
        self.B = nn.Parameter(torch.randn(num_heads, self.d_head) * 0.1)
        # Output projection C per head: [num_heads, d_head, d_head]
        self.C = nn.Parameter(torch.randn(num_heads, self.d_head, self.d_head) * 0.1)

        self.D = nn.Parameter(torch.ones(d_model))
        self.input_proj = nn.Linear(d_model, d_state)
        self.output_proj = nn.Linear(d_state, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def _apply_ssm(self, x_head, lambdas, B_h, C_h):
        """
        Vectorized SSM via causal depthwise convolution.

        x_head: [batch, seq_len, d_head]
        lambdas: [d_head]  in (0, 1)
        B_h:    [d_head]
        C_h:    [d_head, d_head]
        Returns: [batch, seq_len, d_head]
        """
        batch, seq_len, d_head = x_head.shape
        device = x_head.device

        # Build exponential-decay kernels K[t, d] = lambda[d]^t * B[d]
        t_idx = torch.arange(seq_len, dtype=torch.float32, device=device)
        K = lambdas.unsqueeze(0) ** t_idx.unsqueeze(1) * B_h.unsqueeze(0)
        # K: [seq_len, d_head]

        # We need causal convolution: h[t] = sum_{k=0}^{t} K[t-k] * x[k]
        # Flip K to turn convolution into correlation with conv1d
        K_flip = K.flip(0)  # [seq_len, d_head]

        # Reshape for grouped conv1d:
        x_in = x_head.permute(0, 2, 1)          # [B, d_head, T]
        x_pad = F.pad(x_in, (seq_len - 1, 0))   # causal padding
        w = K_flip.T.unsqueeze(1)                # [d_head, 1, T]

        h = F.conv1d(x_pad, w, groups=d_head)   # [B, d_head, T]
        h = h.permute(0, 2, 1)                   # [B, T, d_head]

        y = h @ C_h.T                            # [B, T, d_head]
        return y

    def forward(self, x):
        batch, seq_len, _ = x.shape
        residual = x
        x_proj = self.input_proj(x)  # [batch, seq_len, d_state]

        outputs = []
        for head in range(self.num_heads):
            lambdas = torch.sigmoid(self.log_neg_lambdas[head])   # [d_head]
            B_h = self.B[head]                                      # [d_head]
            C_h = self.C[head]                                      # [d_head, d_head]
            x_head = x_proj[:, :, head * self.d_head:(head + 1) * self.d_head]

            head_out = self._apply_ssm(x_head, lambdas, B_h, C_h)
            outputs.append(head_out)

        out = torch.cat(outputs, dim=-1)
        out = self.output_proj(out)
        out = self.dropout(out)
        out = self.norm(out + residual)
        return out
