from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from oran_sim.models import AttentionRegressor, LSTMRegressor, LiquidRegressor
from oran_sim.sequence_data import make_sequences, sort_by_time


def compute_pct_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    denom = np.abs(y_true)
    pct = np.full(err.shape, np.nan, dtype=float)
    np.divide(err * 100.0, denom, out=pct, where=denom > 1e-6)
    return pct


def _build_temporal_model(spec: dict, input_dim: int):
    arch = spec.get("architecture", "lstm")
    if arch == "attention":
        return AttentionRegressor(
            input_dim=input_dim,
            d_model=spec.get("d_model") or 64,
            nhead=spec.get("nhead") or 4,
            num_layers=spec.get("num_layers") or 2,
            dim_feedforward=spec.get("dim_feedforward") or 128,
            dropout=spec.get("dropout") if spec.get("dropout") is not None else 0.1,
        )
    if arch == "liquid":
        return LiquidRegressor(input_dim=input_dim, hidden_size=spec.get("hidden_size") or 64, dt=spec.get("dt") or 0.1)
    return LSTMRegressor(input_dim=input_dim, hidden_sizes=spec.get("hidden_sizes") or [64, 32])


def _predict_temporal(model_dir: Path, df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = json.loads((model_dir / "temporal_spec.json").read_text(encoding="utf-8"))
    seq_len = int(spec.get("seq_len", 16))
    xdf = df[features].copy()
    for c in xdf.columns:
        if not pd.api.types.is_numeric_dtype(xdf[c]):
            xdf[c] = xdf[c].astype("category").cat.codes
    xdf = xdf.fillna(0.0)
    scaler = joblib.load(model_dir / "temporal_scaler.joblib")
    x = scaler.transform(xdf.to_numpy(dtype=float))
    y = df["target"].to_numpy(dtype=float)
    x_seq, y_seq = make_sequences(x, y, seq_len)

    model = _build_temporal_model(spec, input_dim=x_seq.shape[-1])
    model.load_state_dict(torch.load(model_dir / "temporal_model.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.from_numpy(x_seq.astype(np.float32))).detach().cpu().numpy()

    start_idx = seq_len - 1
    time_vals = df.get("time_ms", pd.Series(np.arange(len(df)))).to_numpy()[start_idx:]
    return y_seq, y_pred, time_vals


def main() -> None:
    p = argparse.ArgumentParser(description="Predict using trained model artifacts")
    p.add_argument("--model_dir", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    features = json.loads((model_dir / "features.json").read_text(encoding="utf-8"))
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))

    df = pd.read_csv(args.csv)

    if cfg.get("temporal", False):
        df = sort_by_time(df)
        y_true, y_pred, time_vals = _predict_temporal(model_dir, df, features)
    else:
        model = joblib.load(model_dir / "model.joblib")
        x = df[features]
        y_true = df["target"].to_numpy()
        y_pred = model.predict(x)
        time_vals = df.get("time_ms", pd.Series(np.arange(len(df)))).to_numpy()

    err = y_true - y_pred
    pct = compute_pct_error(y_true, y_pred)

    out_df = pd.DataFrame(
        {
            "index": np.arange(len(y_true)),
            "time_ms": time_vals,
            "y_true": y_true,
            "y_pred": y_pred,
            "error": err,
            "abs_error": np.abs(err),
            "pct_error": pct,
        }
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"wrote predictions to {out} rows={len(out_df)}")


if __name__ == "__main__":
    main()
