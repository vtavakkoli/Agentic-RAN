#!/usr/bin/env python3
"""Model Zoo for O-RAN NAS simulation scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


SCENARIO_CONFIGS: Dict[str, Dict] = {
    "lightweight-32": {"kind": "lstm", "hidden_sizes": [32], "features": 6},
    "lightweight-64": {"kind": "lstm", "hidden_sizes": [64], "features": 6},
    "balanced-small": {"kind": "lstm", "hidden_sizes": [64, 32], "features": 8},
    "balanced-medium": {"kind": "lstm", "hidden_sizes": [100, 50], "features": 8},
    "deep-performance": {"kind": "lstm", "hidden_sizes": [128, 100, 64], "features": 10},
    "ultra-performance": {"kind": "lstm", "hidden_sizes": [512, 256, 128], "features": 16},
    # Transformer tuned to be in same ballpark as balanced-medium.
    "attention-baseline": {
        "kind": "attention",
        "features": 8,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 128,
        "dropout": 0.1,
    },
    # Lightweight continuous-time RNN proxy for Liquid networks.
    "liquid-baseline": {
        "kind": "liquid",
        "features": 6,
        "hidden_size": 64,
        "dt": 0.1,
    },
}


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: List[int]):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.layers = nn.ModuleList()

        in_dim = input_dim
        for h in hidden_sizes:
            self.layers.append(nn.LSTM(input_size=in_dim, hidden_size=h, batch_first=True))
            in_dim = h
        self.head = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for lstm in self.layers:
            out, _ = lstm(out)
        return self.head(out[:, -1, :]).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class AttentionRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = self.norm(x[:, -1, :])
        return self.head(x).squeeze(-1)


class LiquidCell(nn.Module):
    """Simple continuous-time recurrent cell: h(t+1)=h+dt*(-h/tau+tanh(Wxh+Whh+b))."""

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
    def __init__(self, input_dim: int, hidden_size: int, dt: float = 0.1):
        super().__init__()
        self.cell = LiquidCell(input_dim, hidden_size, dt)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        h = torch.zeros(batch, self.cell.hidden_dim, device=x.device)
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
    cfg = SCENARIO_CONFIGS[model_type]
    kind = cfg["kind"]

    if kind == "lstm":
        return LSTMRegressor(input_dim=input_dim, hidden_sizes=cfg["hidden_sizes"])
    if kind == "attention":
        return AttentionRegressor(
            input_dim=input_dim,
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
        )
    if kind == "liquid":
        return LiquidRegressor(input_dim=input_dim, hidden_size=cfg["hidden_size"], dt=cfg["dt"])

    raise ValueError(f"Unsupported model_type={model_type}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_model_size_mb(model: nn.Module) -> float:
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / (1024**2)


def lstm_layer_complexity(d_x: int, d_h: int) -> int:
    """Exact requested formula: C_LSTM = 4 * (d_x*d_h + d_h^2 + d_h)."""
    return 4 * (d_x * d_h + d_h**2 + d_h)


def compute_complexity(model_type: str, input_dim: int, seq_len: int) -> Complexity:
    cfg = SCENARIO_CONFIGS[model_type]
    model = build_model(model_type, input_dim)
    params = count_parameters(model)
    size_mb = estimate_model_size_mb(model)

    if cfg["kind"] == "lstm":
        total = 0
        prev_dim = input_dim
        for h in cfg["hidden_sizes"]:
            total += lstm_layer_complexity(prev_dim, h)
            prev_dim = h
        lstm_comp = float(total)
        general = lstm_comp * seq_len
    elif cfg["kind"] == "attention":
        d_model = cfg["d_model"]
        n_layers = cfg["num_layers"]
        ff = cfg["dim_feedforward"]
        # Generalized operation count proxy per sequence:
        # per layer ~ O(S^2*d_model + S*d_model*ff)
        general = float(n_layers * (seq_len**2 * d_model + seq_len * d_model * ff))
        lstm_comp = 0.0
    else:
        h = cfg["hidden_size"]
        # Generalized liquid complexity per step approx O(d_x*h + h^2 + h)
        per_step = input_dim * h + h * h + h
        general = float(per_step * seq_len)
        lstm_comp = 0.0

    return Complexity(model_params=params, model_size_mb=size_mb, lstm_complexity=lstm_comp, general_complexity=general)


def scenario_feature_count(model_type: str) -> int:
    return int(SCENARIO_CONFIGS[model_type]["features"])


def supported_scenarios() -> Tuple[str, ...]:
    return tuple(SCENARIO_CONFIGS.keys())
