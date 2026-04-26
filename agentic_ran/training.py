from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _to_tensor(x, y):
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def weighted_huber_loss(pred: torch.Tensor, target: torch.Tensor, peak_weight: float, quantile_85: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    abs_err = torch.abs(pred - target)
    huber = torch.where(abs_err <= delta, 0.5 * abs_err**2, delta * (abs_err - 0.5 * delta))
    peak_mask = (target > quantile_85).to(target.dtype)
    weight = 1.0 + peak_weight * peak_mask
    return torch.mean(weight * huber)


def _build_criterion(loss_name: str, y_train: torch.Tensor, peak_weight: float):
    if loss_name == "mse":
        return nn.MSELoss(), None
    if loss_name == "huber":
        return nn.SmoothL1Loss(), None
    if loss_name == "weighted_huber":
        q85 = torch.quantile(y_train, 0.85)

        def _criterion(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return weighted_huber_loss(pred, target, peak_weight=peak_weight, quantile_85=q85.to(target.device))

        return _criterion, float(q85.item())
    raise ValueError(f"Unsupported loss: {loss_name}")


def train_model(
    model,
    train_set,
    val_set,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    log_path: Path,
    loss_name: str = "mse",
    peak_weight: float = 2.0,
):
    x_train, y_train = _to_tensor(*train_set)
    x_val, y_val = _to_tensor(*val_set)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion, q85 = _build_criterion(loss_name=loss_name, y_train=y_train, peak_weight=peak_weight)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * len(xb)

        train_loss /= max(1, len(train_loader.dataset))
        val_loss /= max(1, len(val_loader.dataset))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    with log_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)

    return history, q85
