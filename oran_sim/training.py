from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, epochs: int, lr: float) -> Dict[str, List[float]]:
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": []}

    for ep in range(epochs):
        model.train()
        tr_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            tr_losses.append(float(loss.item()))

        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                va_losses.append(float(criterion(model(xb), yb).item()))

        history["train_loss"].append(float(np.mean(tr_losses)))
        history["val_loss"].append(float(np.mean(va_losses)))
        print(f"Epoch {ep+1:02d}/{epochs} train={history['train_loss'][-1]:.5f} val={history['val_loss'][-1]:.5f}")

    return history
