import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM = 0

batch_size = 256

class S6(nn.Module):
    """处理离散化过程和前向传播
    """
    def __init__(self, seq_len, d_model, state_size) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model
        self.state_size = state_size

        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, state_size)
        self.fc3 = nn.Linear(d_model, state_size)

        self.A = nn.Parameter(F.normalize(torch.ones(d_model, state_size), p=2, dim=-1))
        nn.init.xavier_uniform_(self.A)

        self.B = torch.zeros(batch_size, seq_len, state_size)
        self.C = torch.zeros(batch_size, seq_len, state_size)

        self.delta = torch.zeros(batch_size, seq_len, d_model)
        self.dA = torch.zeros(batch_size, seq_len, d_model, state_size)
        self.dB = torch.zeros(batch_size, seq_len, d_model, state_size)

        self.h = torch.zeros(batch_size, seq_len, d_model, state_size)
        self.y = torch.zeros(batch_size, seq_len, d_model)

    def discretize(self):
        self.dB = torch.einsum("bld, bln->bldn", self.delta, self.B)
        self.dA = torch.exp(torch.einsum("bld, dn->bldn", self.delta, self.A))
        return self.dA, self.dB

    def forward(self, x):
        self.B = self.fc2(x)
        self.C = self.fc3(x)
        self.delta = F.softplus(self.fc1(x))

        self.discretize()

        h = torch.zeros(x.size(0), self.seq_len, self.d_model, self.state_size, device=x.device)
        y = torch.zeros_like(x)

        h = torch.einsum("bldn, bldn->bldn", self.dA, h) + rearrange(x, "b l d -> b l d 1") * self.dB

        y = torch.einsum("bln, bldn->bld", self.C, h)
        return y

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class MambaBlock(nn.Module):
    def __init__(self, seq_len, d_model, state_size) -> None:
        super().__init__()

        self.inp_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(2 * d_model, d_model)

        # For residual skip connection
        self.D = nn.Linear(d_model, 2 * d_model)

        self.out_proj.bias._no_weight_decay = True

        nn.init.constant_(self.out_proj.bias, 1.0)

        self.S6 = S6(seq_len, 2 * d_model, state_size)

        self.conv = nn.Conv1d(seq_len, seq_len, kernel_size=3, padding=1)

        self.conv_linear = nn.Linear(2 * d_model, 2 * d_model)

        self.norm = RMSNorm(d_model)

    def forward(self, x):
        """
        x_proj.shape = (batch_size, seq_len, 2 * d_model)
        x_conv.shape = (batch_size, seq_len, 2 * d_model)
        x_conv_act.shape = (batch_size, seq_len, 2 * d_model)
        """
        x = self.norm(x)
        x_proj = self.inp_proj(x)
        x_conv = self.conv(x_proj)
        x_conv_act = F.silu(x_conv)
        x_conv_out = self.conv_linear(x_conv_act)
        x_ssm = self.S6(x_conv_out)
        x_act = F.silu(x_ssm)
        x_residual = F.silu(self.D(x))
        x_combined = x_residual * x_act
        x_out = self.out_proj(x_combined)
        return x_out

class Mamba(nn.Module):
    def __init__(self, seq_len, d_model, state_size):
        super().__init__()

        self.mamba_block1 = MambaBlock(seq_len, d_model, state_size)
        self.mamba_block2 = MambaBlock(seq_len, d_model, state_size)
        self.mamba_block3 = MambaBlock(seq_len, d_model, state_size)

    def forward(self, x):
        x = self.mamba_block1(x)
        x = self.mamba_block2(x)
        x = self.mamba_block3(x)
        return x
