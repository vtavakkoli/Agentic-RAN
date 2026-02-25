from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from oran_sim.config import DEFAULT_FEATURE_COUNT, FEATURE_ORDER, get_feature_columns
from oran_sim.data import load_timeseries_from_kpm
from oran_sim.seed import SEED, set_global_seed


MODEL_TYPES = {"ridge", "hgb", "lstm", "gru"}


def load_source(data_root: str) -> tuple[pd.DataFrame, str]:
    path = Path(data_root)
    if path.is_file():
        df = pd.read_csv(path)
        return df, "synthetic_csv"
    return load_timeseries_from_kpm(path), "kpm_folder"


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["time_ms", "traffic_load", "split"] + FEATURE_ORDER:
        if c not in out.columns:
            out[c] = 0.0 if c != "split" else ""
    out["time_ms"] = pd.to_numeric(out["time_ms"], errors="coerce").ffill().fillna(0).astype("int64")
    out["traffic_load"] = pd.to_numeric(out["traffic_load"], errors="coerce").fillna(0.0)
    for c in FEATURE_ORDER:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype("float32")
    return out.sort_values("time_ms").reset_index(drop=True)


def split_dataframe(df: pd.DataFrame, source_kind: str) -> pd.DataFrame:
    out = df.copy()
    if source_kind == "synthetic_csv" and out["split"].astype(str).isin(["train", "val", "test"]).all():
        return out
    n = len(out)
    n_train = int(n * 0.6)
    n_val = int(n * 0.3)
    out["split"] = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
    return out


def to_supervised(df: pd.DataFrame, features: list[str], seq_len: int, horizon_steps: int, target: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i in range(seq_len, len(df) - horizon_steps):
        row: dict[str, Any] = {
            "time_ms": int(df.iloc[i]["time_ms"]),
            "split": str(df.iloc[i]["split"]),
            "y_true": float(df.iloc[i + horizon_steps]["traffic_load"]),
        }
        if target == "delta_traffic_load":
            row["y_true"] = row["y_true"] - float(df.iloc[i]["traffic_load"])

        for lag in range(seq_len):
            src = df.iloc[i - seq_len + lag]
            for f in features:
                row[f"{f}__t{lag-seq_len+1}"] = float(src[f])
        rows.append(row)

    return pd.DataFrame(rows)


def fit_model(model_name: str, x_train: np.ndarray, y_train: np.ndarray, seed: int):
    if model_name in {"ridge", "lstm", "gru"}:
        model = Ridge(alpha=1.0, random_state=seed)
    elif model_name == "hgb":
        model = HistGradientBoostingRegressor(random_state=seed, max_depth=6, max_iter=200)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model.fit(x_train, y_train)
    return model


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def persist_artifacts(
    out_dir: str | Path,
    model,
    scaler_x: StandardScaler,
    scaler_y: StandardScaler,
    feature_columns: list[str],
    config: dict[str, Any],
    metrics_json: dict[str, Any],
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    joblib.dump(scaler_x, out / "x_scaler.joblib")
    joblib.dump(scaler_y, out / "y_scaler.joblib")
    (out / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")
    return out


def default_feature_count() -> int:
    return DEFAULT_FEATURE_COUNT


def choose_features(feature_count: int) -> list[str]:
    return get_feature_columns(feature_count)


def seed_everything(seed: int = SEED) -> None:
    set_global_seed(seed)
