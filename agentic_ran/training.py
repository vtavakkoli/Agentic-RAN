from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _to_tensor(x, y, action_labels=None):
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    if action_labels is None:
        return x_t, y_t, None
    return x_t, y_t, torch.tensor(action_labels, dtype=torch.long)


def weighted_huber_loss(pred: torch.Tensor, target: torch.Tensor, peak_weight: float, quantile_85: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    abs_err = torch.abs(pred - target)
    huber = torch.where(abs_err <= delta, 0.5 * abs_err**2, delta * (abs_err - 0.5 * delta))
    peak_mask = (target > quantile_85).to(target.dtype)
    weight = 1.0 + peak_weight * peak_mask
    return torch.mean(weight * huber)


def _build_regression_criterion(loss_name: str, y_train: torch.Tensor, peak_weight: float):
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
    action_loss_alpha: float = 0.5,
    early_stopping_patience: int = 5,
):
    x_train, y_train = train_set[0], train_set[1]
    x_val, y_val = val_set[0], val_set[1]
    train_actions = train_set[2] if len(train_set) > 2 else None
    val_actions = val_set[2] if len(val_set) > 2 else None

    x_train, y_train, a_train = _to_tensor(x_train, y_train, train_actions)
    x_val, y_val, a_val = _to_tensor(x_val, y_val, val_actions)

    if a_train is None:
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    else:
        train_loader = DataLoader(TensorDataset(x_train, y_train, a_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(x_val, y_val, a_val), batch_size=batch_size, shuffle=False)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    reg_criterion, q85 = _build_regression_criterion(loss_name=loss_name, y_train=y_train, peak_weight=peak_weight)
    action_criterion = nn.CrossEntropyLoss()

    best_val = float("inf")
    best_state = None
    no_improve = 0

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            xb, yb = batch[0].to(device), batch[1].to(device)
            ab = batch[2].to(device) if len(batch) > 2 else None
            optimizer.zero_grad()
            out = model(xb)
            if isinstance(out, tuple):
                pred, logits, _ = out
                loss = reg_criterion(pred, yb)
                if ab is not None:
                    loss = loss + action_loss_alpha * action_criterion(logits, ab)
            else:
                pred = out
                loss = reg_criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                xb, yb = batch[0].to(device), batch[1].to(device)
                ab = batch[2].to(device) if len(batch) > 2 else None
                out = model(xb)
                if isinstance(out, tuple):
                    pred, logits, _ = out
                    loss = reg_criterion(pred, yb)
                    if ab is not None:
                        loss = loss + action_loss_alpha * action_criterion(logits, ab)
                else:
                    loss = reg_criterion(out, yb)
                val_loss += loss.item() * len(xb)

        train_loss /= max(1, len(train_loader.dataset))
        val_loss /= max(1, len(val_loader.dataset))
        scheduler.step(val_loss)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]})
        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    with log_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["epoch", "train_loss", "val_loss", "lr"])
        writer.writeheader()
        writer.writerows(history)

    return history, q85
