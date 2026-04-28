from __future__ import annotations

import numpy as np
import torch


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    if not labels:
        return 0.0
    f1s: list[float] = []
    for label in labels:
        tp = float(np.sum((y_true == label) & (y_pred == label)))
        fp = float(np.sum((y_true != label) & (y_pred == label)))
        fn = float(np.sum((y_true == label) & (y_pred != label)))
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        f1s.append(f1)
    return float(np.mean(f1s))


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    action_true: np.ndarray | None = None,
    action_pred: np.ndarray | None = None,
    is_agentic_model: bool = False,
) -> dict:
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
    composite_score = float(max(0.0, r2) * 100 - (0.2 * rmse + 0.25 * mae + 0.15 * smape + 0.15 * wmape))

    peak_q85 = float(np.quantile(y_true, 0.85))
    peak_mask = y_true > peak_q85
    normal_mask = ~peak_mask
    peak_mae = float(np.mean(np.abs(err[peak_mask]))) if np.any(peak_mask) else float("nan")
    normal_mae = float(np.mean(np.abs(err[normal_mask]))) if np.any(normal_mask) else float("nan")
    peak_rmse = float(np.sqrt(np.mean(err[peak_mask] ** 2))) if np.any(peak_mask) else float("nan")
    if np.any(peak_mask):
        y_peak = y_true[peak_mask]
        peak_sst = float(np.sum((y_peak - np.mean(y_peak)) ** 2) + 1e-9)
        peak_sse = float(np.sum(err[peak_mask] ** 2))
        peak_r2 = 1 - peak_sse / peak_sst
    else:
        peak_r2 = float("nan")

    payload = {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "smape": smape,
        "wmape": wmape,
        "composite_score": composite_score,
        "peak_q85": peak_q85,
        "peak_mae": peak_mae,
        "normal_mae": normal_mae,
        "peak_rmse": peak_rmse,
        "peak_r2": peak_r2,
    }

    if is_agentic_model and action_true is not None and action_pred is not None and len(action_true) == len(action_pred):
        payload["action_accuracy"] = float((action_true == action_pred).mean())
        payload["action_macro_f1"] = _macro_f1(action_true, action_pred)
        unique, counts = np.unique(action_true, return_counts=True)
        payload["per_action_support"] = {int(k): int(v) for k, v in zip(unique.tolist(), counts.tolist())}
        pred_u, pred_c = np.unique(action_pred, return_counts=True)
        payload["action_distribution"] = {int(k): int(v) for k, v in zip(pred_u.tolist(), pred_c.tolist())}
    else:
        payload["action_accuracy"] = None
        payload["action_macro_f1"] = None

    return payload


def predict(model, x_test: np.ndarray, device: str):
    model.eval()
    x_tensor = torch.tensor(x_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(x_tensor)
        if isinstance(out, tuple):
            pred, logits, confidence = out
            return (
                pred.detach().cpu().numpy(),
                logits.argmax(dim=-1).detach().cpu().numpy(),
                confidence.detach().cpu().numpy(),
            )
        pred = out.detach().cpu().numpy()
    return pred
