import torch
import torch.nn as nn
import numpy as np


class S4Layer(nn.Module):
    def __init__(self, d_model, d_state=256, dropout=0.1, num_heads=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.num_heads = num_heads
        self.d_head = d_state // num_heads
        
        self.A_list = nn.ParameterList([
            nn.Parameter(self._init_hippo_matrix(self.d_head), requires_grad=False)
            for _ in range(num_heads)
        ])
        self.B = nn.Parameter(torch.randn(num_heads, self.d_head, 1))
        self.C = nn.Parameter(torch.randn(num_heads, 1, self.d_head))
        self.D = nn.Parameter(torch.ones(d_model))
        self.log_dt = nn.Parameter(torch.ones(num_heads) * (-2.0))
        self.scale = nn.Parameter(torch.ones(num_heads) * 0.02)
        
        self.input_proj = nn.Linear(d_model, d_state)
        self.output_proj = nn.Linear(d_state, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def _init_hippo_matrix(self, N):
        A = np.zeros((N, N))
        for n in range(N):
            for k in range(N):
                if n > k:
                    A[n, k] = -np.sqrt((2*n+1) * (2*k+1))
                elif n == k:
                    A[n, n] = n + 1
        A = A / (np.max(np.abs(A)) + 1e-8)
        return torch.tensor(A, dtype=torch.float32)
    
    def _discretize(self, A, B, dt):
        I = torch.eye(A.size(0), device=A.device)
        A_discrete = torch.linalg.solve(I + dt/2 * A, I - dt/2 * A)
        B_discrete = torch.linalg.solve(I + dt/2 * A, dt * B)
        return A_discrete, B_discrete
    
    def forward(self, x):
        batch, seq_len, _ = x.shape
        residual = x
        x_proj = self.input_proj(x)
        outputs = []
        
        for head in range(self.num_heads):
            A = self.A_list[head]
            B = self.B[head]
            C = self.C[head]
            dt = torch.exp(self.log_dt[head])
            scale = self.scale[head]
            
            A_scaled = A * scale
            A_d, B_d = self._discretize(A_scaled, B, dt)
            
            state = torch.zeros(batch, self.d_head, device=x.device)
            head_outputs = []
            x_head = x_proj[:, :, head*self.d_head:(head+1)*self.d_head]
            
            for t in range(seq_len):
                x_t = x_head[:, t, :]
                state = torch.matmul(state, A_d.T) + x_t.unsqueeze(2) * B_d.squeeze()
                state = torch.clamp(state, -10, 10)
                y_t = torch.matmul(state, C.T)
                head_outputs.append(y_t)
            
            head_out = torch.stack(head_outputs, dim=1)
            outputs.append(head_out)
        
        out = torch.cat(outputs, dim=-1)
        out = self.output_proj(out.expand(-1, -1, self.d_state))
        out = self.dropout(out)
        out = self.norm(out + residual)
        return out
