#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from oran_sim.agent import react_loop
from oran_sim.config import SCENARIOS, supported_scenarios
from oran_sim.data import prepare_dataset
from oran_sim.models import build_model, compute_complexity
from oran_sim.training import train_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Near-RT RIC xApp runner")
    p.add_argument("--model_type", required=True, choices=supported_scenarios())
    p.add_argument("--features", type=int, default=None)
    p.add_argument("--data", type=str, default="shared_data/traffic_data.csv")
    p.add_argument("--output", type=str, default="shared_data/results.csv")
    p.add_argument("--seq_len", type=int, default=24)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    default_features = SCENARIOS[args.model_type].features
    feature_count = args.features or default_features

    x_seq, y_seq, _, y_scaler, selected = prepare_dataset(args.data, feature_count, args.seq_len)

    n = len(x_seq)
    tr_end, va_end = int(0.7 * n), int(0.85 * n)
    x_train, y_train = x_seq[:tr_end], y_seq[:tr_end]
    x_val, y_val = x_seq[tr_end:va_end], y_seq[tr_end:va_end]
    x_test, y_test = x_seq[va_end:], y_seq[va_end:]

    train_loader = DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(y_train)), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(x_val), torch.tensor(y_val)), batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model_type, feature_count)
    complexity = compute_complexity(args.model_type, feature_count, args.seq_len)
    history = train_model(model, train_loader, val_loader, device, args.epochs, args.lr)

    model.eval()
    with torch.no_grad():
        preds_scaled = model(torch.tensor(x_test, device=device)).cpu().numpy()

    y_test_real = y_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
    preds_real = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(-1)

    result = {
        "model_type": args.model_type,
        "features": feature_count,
        "seq_len": args.seq_len,
        "epochs": args.epochs,
        "params": complexity.model_params,
        "model_size_mb": complexity.model_size_mb,
        "lstm_complexity": complexity.lstm_complexity,
        "general_complexity": complexity.general_complexity,
        "mae": float(mean_absolute_error(y_test_real, preds_real)),
        "rmse": float(np.sqrt(mean_squared_error(y_test_real, preds_real))),
        "mape": float(np.mean(np.abs((y_test_real - preds_real) / np.clip(np.abs(y_test_real), 1e-6, None))) * 100),
        "r2": float(r2_score(y_test_real, preds_real)),
        "final_train_loss": float(history["train_loss"][-1]),
        "final_val_loss": float(history["val_loss"][-1]),
        **react_loop(y_test_real, preds_real),
        "selected_features": json.dumps(selected),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(out) + ".lock"):
        if out.exists():
            prev = pd.read_csv(out)
            prev = prev[prev["model_type"] != args.model_type]
            merged = pd.concat([prev, pd.DataFrame([result])], ignore_index=True)
        else:
            merged = pd.DataFrame([result])
        merged.to_csv(out, index=False)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
