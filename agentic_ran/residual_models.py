from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


class ResidualMLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        block_count = max(3, min(6, num_layers))
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([ResidualBlock(hidden_size, dropout=dropout) for _ in range(block_count)])
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x[:, -1, :]
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h).squeeze(-1)


class _TemporalConvStack(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList()
        for dilation in (1, 2, 4):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=dilation, dilation=dilation),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
                    nn.Dropout(dropout),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = h + layer(h)
        return h


class ResidualTCNRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        block_count = max(3, min(6, num_layers))
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_size, dropout=dropout) for _ in range(block_count)])
        self.temporal = _TemporalConvStack(hidden_size, dropout=dropout)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.temporal(h.transpose(1, 2)).transpose(1, 2)
        return self.head(h[:, -1, :]).squeeze(-1)


class ResidualLiquidTCNRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        block_count = max(3, min(6, num_layers))
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_size, dropout=dropout) for _ in range(block_count)])
        self.temporal = _TemporalConvStack(hidden_size, dropout=dropout)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.state_proj = nn.Linear(hidden_size, hidden_size)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        h_seq = self.input_proj(x)
        for block in self.blocks:
            h_seq = block(h_seq)

        temporal_seq = self.temporal(h_seq.transpose(1, 2)).transpose(1, 2)

        bsz, seq_len, hidden = h_seq.shape
        h = torch.zeros((bsz, hidden), device=h_seq.device, dtype=h_seq.dtype)
        for t in range(seq_len):
            temporal_state = torch.tanh(self.state_proj(temporal_seq[:, t, :]))
            gate = torch.sigmoid(self.gate(h_seq[:, t, :]))
            h = h + gate * temporal_state

        return self.head(h).squeeze(-1)
