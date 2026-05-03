from __future__ import annotations

import torch
from torch import nn

from agentic_ran.agentic_models import AgenticLiquidModel, AgenticResidualMLP, AgenticSequenceModel
from agentic_ran.residual_models import ResidualLiquidTCNRegressor, ResidualMLPRegressor, ResidualTCNRegressor
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




class PatchTSTRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float, patch_len: int = 4):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(input_dim * patch_len, hidden_size)
        enc = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=hidden_size * 2, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=max(1, num_layers))
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        b, t, f = x.shape
        p = min(self.patch_len, t)
        n = t // p
        x = x[:, : n * p, :].reshape(b, n, p * f)
        h = self.encoder(self.proj(x))
        return self.head(h.mean(dim=1)).squeeze(-1)


class TSMixerRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_size)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Dropout(dropout)) for _ in range(max(1, num_layers))])
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.in_proj(x)
        for b in self.blocks:
            h = h + b(h)
        return self.head(h[:, -1, :]).squeeze(-1)


class KANRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_size), nn.SiLU()]
        for _ in range(max(1, num_layers - 1)):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x[:, -1, :]
        return self.net(x).squeeze(-1)


class AgenticPatchKANMixer(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.patch = PatchTSTRegressor(input_dim, hidden_size, max(1, num_layers // 2), dropout)
        self.kan = KANRegressor(input_dim, hidden_size, max(1, num_layers // 2), dropout)
        self.mix = nn.Sequential(nn.Linear(2, hidden_size), nn.GELU(), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.patch(x)
        k = self.kan(x)
        return self.mix(torch.stack([p, k], dim=-1)).squeeze(-1)

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
    if config.model_type == "patchtst":
        return PatchTSTRegressor(input_dim=input_dim, hidden_size=config.hidden_size, num_layers=config.num_layers, dropout=config.dropout)
    if config.model_type == "tsmixer":
        return TSMixerRegressor(input_dim=input_dim, hidden_size=config.hidden_size, num_layers=config.num_layers, dropout=config.dropout)
    if config.model_type == "kan":
        return KANRegressor(input_dim=input_dim, hidden_size=config.hidden_size, num_layers=config.num_layers, dropout=config.dropout)
    if config.model_type == "agentic_patch_kan_mixer":
        return AgenticPatchKANMixer(input_dim=input_dim, hidden_size=config.hidden_size, num_layers=config.num_layers, dropout=config.dropout)
    if config.model_type == "residual_mlp":
        return ResidualMLPRegressor(input_dim=input_dim, hidden_size=config.hidden_size, num_layers=config.num_layers, dropout=config.dropout)
    if config.model_type == "residual_tcn":
        return ResidualTCNRegressor(input_dim=input_dim, hidden_size=config.hidden_size, num_layers=config.num_layers, dropout=config.dropout)
    if config.model_type == "residual_liquid_tcn":
        return ResidualLiquidTCNRegressor(input_dim=input_dim, hidden_size=config.hidden_size, num_layers=config.num_layers, dropout=config.dropout)
    if config.model_type in {"agentic_mlp", "agentic_residual_mlp"}:
        return AgenticResidualMLP(input_dim=input_dim, hidden_size=config.hidden_size, num_layers=config.num_layers, dropout=config.dropout)
    if config.model_type in {"agentic_sequence_model", "agentic_sequence_attention"}:
        return AgenticSequenceModel(input_dim=input_dim, hidden_size=config.hidden_size, num_layers=config.num_layers, dropout=config.dropout)
    if config.model_type in {"agentic_liquid_model", "agentic_liquid_residual", "liquid_agentic", "liquid_agentic_residual"}:
        return AgenticLiquidModel(input_dim=input_dim, hidden_size=config.hidden_size, num_layers=config.num_layers, dropout=config.dropout)
    raise ValueError(f"Unsupported model type: {config.model_type}")
