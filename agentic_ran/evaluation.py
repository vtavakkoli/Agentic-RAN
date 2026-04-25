from __future__ import annotations

import numpy as np
import torch


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    y_mean = np.mean(y_true)
    sst = float(np.sum((y_true - y_mean) ** 2) + 1e-9)
    sse = float(np.sum(err**2))
    r2 = 1 - sse / sst

    mape = float(np.mean(np.abs(err) / (np.abs(y_true) + 1e-6)) * 100)
    smape = float(np.mean(2 * np.abs(err) / (np.abs(y_true) + np.abs(y_pred) + 1e-6)) * 100)
    wmape = float(np.sum(np.abs(err)) / (np.sum(np.abs(y_true)) + 1e-6) * 100)
    composite_score = float(max(0.0, r2) * 100 - (0.15 * rmse + 0.2 * mae + 0.1 * mape + 0.1 * smape + 0.1 * wmape))

    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "smape": smape,
        "wmape": wmape,
        "composite_score": composite_score,
    }


def predict(model, x_test: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    x_tensor = torch.tensor(x_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(x_tensor).detach().cpu().numpy()
    return pred
