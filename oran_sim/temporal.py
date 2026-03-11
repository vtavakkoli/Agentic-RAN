from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from oran_sim.models import AttentionRegressor, LSTMRegressor, LiquidRegressor


@dataclass(frozen=True)
class TemporalSpec:
    architecture: str
    seq_len: int
    hidden_sizes: list[int] | None = None
    d_model: int | None = None
    nhead: int | None = None
    num_layers: int | None = None
    dim_feedforward: int | None = None
    dropout: float | None = None
    hidden_size: int | None = None
    dt: float | None = None


def build_temporal_model(input_dim: int, spec: TemporalSpec) -> nn.Module:
    if spec.architecture == "lstm":
        return LSTMRegressor(input_dim=input_dim, hidden_sizes=spec.hidden_sizes or [64, 32])
    if spec.architecture == "attention":
        return AttentionRegressor(
            input_dim=input_dim,
            d_model=spec.d_model or 64,
            nhead=spec.nhead or 4,
            num_layers=spec.num_layers or 2,
            dim_feedforward=spec.dim_feedforward or 128,
            dropout=spec.dropout if spec.dropout is not None else 0.1,
        )
    if spec.architecture == "liquid":
        return LiquidRegressor(input_dim=input_dim, hidden_size=spec.hidden_size or 64, dt=spec.dt or 0.1)
    raise ValueError(f"Unsupported temporal architecture: {spec.architecture}")


def train_temporal_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epochs: int,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> nn.Module:
    device = torch.device("cpu")
    model.to(device)
    ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(max(1, epochs)):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def predict_temporal_model(model: nn.Module, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        out = model(torch.from_numpy(x).to(torch.device("cpu")))
    return out.detach().cpu().numpy().astype(float)
