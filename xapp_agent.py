#!/usr/bin/env python3
"""Near-RT RIC xApp simulation with ReAct loop for O-RAN traffic prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from filelock import FileLock
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from model_zoo import (
    compute_complexity,
    scenario_feature_count,
    supported_scenarios,
    build_model,
)

# Feature priority order for slicing scenario feature subset.
FEATURE_ORDER = [
    "rsrp",
    "rsrq",
    "sinr",
    "prb_utilization",
    "ue_count",
    "handover_rate",
    "packet_loss",
    "latency_ms",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "is_weekend",
    "mobility_index",
    "video_demand",
    "gaming_demand",
    "iot_demand",
]


def build_sequences(features: np.ndarray, target: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i in range(seq_len, len(features)):
        x.append(features[i - seq_len : i])
        y.append(target[i])
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> Dict[str, float]:
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}
    for ep in range(epochs):
        model.train()
        tr_losses: List[float] = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            tr_losses.append(float(loss.item()))

        model.eval()
        va_losses: List[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                va_losses.append(float(loss.item()))

        history["train_loss"].append(float(np.mean(tr_losses)))
        history["val_loss"].append(float(np.mean(va_losses)))
        print(
            f"Epoch {ep + 1:02d}/{epochs} | train_loss={history['train_loss'][-1]:.6f} "
            f"| val_loss={history['val_loss'][-1]:.6f}"
        )

    return history


def react_loop(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.72) -> Dict[str, float]:
    """Mock ReAct loop (Thought, Action, Monitor) across test predictions."""
    reward = 0.0
    penalties = 0.0
    tx_power = 1.0
    rb_alloc = 1.0

    for yt, yp in zip(y_true, y_pred):
        # Thought: infer congestion risk and prediction confidence.
        error = abs(yt - yp)
        congestion_risk = yp
        confidence = max(0.0, 1.0 - error)

        # Action: adapt TX power and RB allocation heuristically.
        if congestion_risk > threshold:
            tx_power = min(1.25, tx_power + 0.02)
            rb_alloc = min(1.40, rb_alloc + 0.03)
        else:
            tx_power = max(0.85, tx_power - 0.01)
            rb_alloc = max(0.85, rb_alloc - 0.01)

        # Monitor: reward favors low error and stable actuation.
        control_cost = abs(tx_power - 1.0) * 0.05 + abs(rb_alloc - 1.0) * 0.05
        step_reward = (1.0 - error) * confidence - control_cost
        reward += step_reward
        penalties += control_cost

    avg_reward = reward / len(y_true)
    avg_penalty = penalties / len(y_true)
    return {
        "react_avg_reward": float(avg_reward),
        "react_avg_penalty": float(avg_penalty),
        "react_final_tx_power": float(tx_power),
        "react_final_rb_alloc": float(rb_alloc),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="xApp agent runner for one NAS scenario")
    parser.add_argument("--model_type", required=True, choices=supported_scenarios())
    parser.add_argument("--features", type=int, default=None, help="Override number of input features")
    parser.add_argument("--data", type=str, default="shared_data/traffic_data.csv")
    parser.add_argument("--output", type=str, default="shared_data/results.csv")
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    feature_count = args.features or scenario_feature_count(args.model_type)
    selected_features = FEATURE_ORDER[:feature_count]

    df = pd.read_csv(args.data)
    x_raw = df[selected_features].values
    y_raw = df["traffic_load"].values.astype(np.float32)

    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x_scaled = x_scaler.fit_transform(x_raw)
    y_scaled = y_scaler.fit_transform(y_raw.reshape(-1, 1)).reshape(-1)

    x_seq, y_seq = build_sequences(x_scaled, y_scaled, seq_len=args.seq_len)
    n = len(x_seq)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    x_train, y_train = x_seq[:train_end], y_seq[:train_end]
    x_val, y_val = x_seq[train_end:val_end], y_seq[train_end:val_end]
    x_test, y_test = x_seq[val_end:], y_seq[val_end:]

    train_loader = DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(y_train)), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(x_val), torch.tensor(y_val)), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model_type, input_dim=feature_count)

    complexity = compute_complexity(args.model_type, input_dim=feature_count, seq_len=args.seq_len)

    history = train_one_model(model, train_loader, val_loader, device=device, epochs=args.epochs, lr=args.lr)

    model.eval()
    with torch.no_grad():
        preds_scaled = model(torch.tensor(x_test, device=device)).cpu().numpy()

    y_test_real = y_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
    preds_real = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(-1)

    mae = mean_absolute_error(y_test_real, preds_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, preds_real))
    mape = np.mean(np.abs((y_test_real - preds_real) / np.clip(np.abs(y_test_real), 1e-6, None))) * 100
    r2 = r2_score(y_test_real, preds_real)

    react_metrics = react_loop(y_test_real, preds_real)

    result = {
        "model_type": args.model_type,
        "features": feature_count,
        "seq_len": args.seq_len,
        "epochs": args.epochs,
        "params": complexity.model_params,
        "model_size_mb": complexity.model_size_mb,
        "lstm_complexity": complexity.lstm_complexity,
        "general_complexity": complexity.general_complexity,
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "r2": float(r2),
        "final_train_loss": float(history["train_loss"][-1]),
        "final_val_loss": float(history["val_loss"][-1]),
        **react_metrics,
        "selected_features": json.dumps(selected_features),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lock = FileLock(str(out_path) + ".lock")
    with lock:
        if out_path.exists():
            existing = pd.read_csv(out_path)
            existing = existing[existing["model_type"] != args.model_type]
            updated = pd.concat([existing, pd.DataFrame([result])], ignore_index=True)
        else:
            updated = pd.DataFrame([result])
        updated.to_csv(out_path, index=False)

    print(f"Completed {args.model_type} -> metrics appended to {out_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
