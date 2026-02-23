from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from oran_sim.config import SCENARIOS


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: list[int]):
        super().__init__()
        self.layers = nn.ModuleList()
        in_dim = input_dim
        for h in hidden_sizes:
            self.layers.append(nn.LSTM(input_size=in_dim, hidden_size=h, batch_first=True))
            in_dim = h
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out, _ = layer(out)
        return self.head(out[:, -1, :]).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class AttentionRegressor(nn.Module):
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos(self.in_proj(x))
        x = self.encoder(x)
        x = self.norm(x[:, -1, :])
        return self.head(x).squeeze(-1)


class LiquidCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dt: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dt = dt
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.rec_proj = nn.Linear(hidden_dim, hidden_dim)
        self.tau = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        drive = torch.tanh(self.in_proj(x_t) + self.rec_proj(h_prev))
        tau = torch.clamp(self.tau, min=0.05)
        dh = -h_prev / tau + drive
        return h_prev + self.dt * dh


class LiquidRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, dt: float):
        super().__init__()
        self.cell = LiquidCell(input_dim, hidden_size, dt)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.zeros(x.size(0), self.cell.hidden_dim, device=x.device)
        for t in range(x.size(1)):
            h = self.cell(x[:, t, :], h)
        return self.head(h).squeeze(-1)


@dataclass
class Complexity:
    model_params: int
    model_size_mb: float
    lstm_complexity: float
    general_complexity: float


def build_model(model_type: str, input_dim: int) -> nn.Module:
    cfg = SCENARIOS[model_type]
    if cfg.kind == "lstm":
        return LSTMRegressor(input_dim, cfg.hidden_sizes or [32])
    if cfg.kind == "attention":
        return AttentionRegressor(input_dim, cfg.d_model or 64, cfg.nhead or 4, cfg.num_layers or 2, cfg.dim_feedforward or 128, cfg.dropout or 0.1)
    if cfg.kind == "liquid":
        return LiquidRegressor(input_dim, cfg.hidden_size or 64, cfg.dt or 0.1)
    raise ValueError(f"Unsupported model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_model_size_mb(model: nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)


def lstm_layer_complexity(d_x: int, d_h: int) -> int:
    return 4 * (d_x * d_h + d_h * d_h + d_h)


def compute_complexity(model_type: str, input_dim: int, seq_len: int) -> Complexity:
    cfg = SCENARIOS[model_type]
    model = build_model(model_type, input_dim)
    params = count_parameters(model)
    size_mb = estimate_model_size_mb(model)

    if cfg.kind == "lstm":
        total = 0
        prev = input_dim
        for h in cfg.hidden_sizes or []:
            total += lstm_layer_complexity(prev, h)
            prev = h
        return Complexity(params, size_mb, float(total), float(total * seq_len))

    if cfg.kind == "attention":
        d_model = cfg.d_model or 64
        n_layers = cfg.num_layers or 2
        ff = cfg.dim_feedforward or 128
        generalized = n_layers * (seq_len**2 * d_model + seq_len * d_model * ff)
        return Complexity(params, size_mb, 0.0, float(generalized))

    h = cfg.hidden_size or 64
    generalized = (input_dim * h + h * h + h) * seq_len
    return Complexity(params, size_mb, 0.0, float(generalized))
