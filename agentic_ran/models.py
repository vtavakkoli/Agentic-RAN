from __future__ import annotations

import torch
from torch import nn

from agentic_ran.scenarios import ScenarioConfig


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        layers = []
        current = input_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(current, hidden_size), nn.ReLU(), nn.Dropout(dropout)])
            current = hidden_size
        layers.append(nn.Linear(current, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class AttentionRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_size)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        return self.head(pooled).squeeze(-1)


class LiquidCell(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_size)
        self.state_proj = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(input_dim + hidden_size, hidden_size)

    def forward(self, x_t: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([x_t, h_t], dim=-1)))
        candidate = torch.tanh(self.in_proj(x_t) + self.state_proj(h_t))
        return (1 - g) * h_t + g * candidate


class LiquidRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = LiquidCell(input_dim, hidden_size)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.size()
        h = torch.zeros((bsz, self.hidden_size), device=x.device)
        for step in range(seq_len):
            h = self.cell(x[:, step, :], h)
        return self.head(h).squeeze(-1)


class XLSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def create_model(config: ScenarioConfig, input_dim: int) -> nn.Module:
    if config.model_type == "mlp":
        return MLPRegressor(input_dim=input_dim, hidden_size=config.hidden_size, num_layers=config.num_layers, dropout=config.dropout)
    if config.model_type == "attention":
        return AttentionRegressor(input_dim=input_dim, hidden_size=config.hidden_size, num_layers=config.num_layers, dropout=config.dropout)
    if config.model_type == "liquid":
        return LiquidRegressor(input_dim=input_dim, hidden_size=config.hidden_size)
    if config.model_type == "xlstm":
        return XLSTMRegressor(input_dim=input_dim, hidden_size=config.hidden_size, num_layers=config.num_layers, dropout=config.dropout)
    raise ValueError(f"Unsupported model type: {config.model_type}")
