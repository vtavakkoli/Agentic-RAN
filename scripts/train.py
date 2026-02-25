from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from oran_sim.config import FEATURE_ORDER


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mape_denom = np.clip(np.abs(y_true), 1e-6, None)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / mape_denom)) * 100.0),
        "R2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Train CPU-only baseline model")
    p.add_argument("--csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", choices=["ridge", "hgb"], default="hgb")
    args = p.parse_args()

    csv = Path(args.csv)
    root = csv.parent
    stem = csv.stem
    train_path = root / f"{stem}_train.csv"
    val_path = root / f"{stem}_val.csv"
    test_path = root / f"{stem}_test.csv"

    if train_path.exists() and val_path.exists() and test_path.exists():
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
    else:
        full = pd.read_csv(csv)
        full = full.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        n = len(full)
        a, b = int(0.6 * n), int(0.9 * n)
        train_df, val_df, test_df = full.iloc[:a].copy(), full.iloc[a:b].copy(), full.iloc[b:].copy()

    features = [c for c in FEATURE_ORDER if c in train_df.columns]
    if "scheduling_policy" not in features and "scheduling_policy" in train_df.columns:
        features.append("scheduling_policy")
    num_features = [c for c in features if c != "scheduling_policy"]
    cat_features = [c for c in features if c == "scheduling_policy"]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
        ],
        remainder="drop",
    )

    model = Ridge(random_state=args.seed) if args.model == "ridge" else HistGradientBoostingRegressor(random_state=args.seed)
    pipe = Pipeline([("pre", pre), ("model", model)])

    x_train, y_train = train_df[features], train_df["target"].to_numpy()
    x_val, y_val = val_df[features], val_df["target"].to_numpy()
    x_test, y_test = test_df[features], test_df["target"].to_numpy()

    pipe.fit(x_train, y_train)
    val_pred = pipe.predict(x_val)
    test_pred = pipe.predict(x_test)

    metrics = {"val": _metrics(y_val, val_pred), "test": _metrics(y_test, test_pred)}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_dir / "model.joblib")
    (out_dir / "features.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(
        json.dumps({"seed": args.seed, "model_type": args.model, "target": "target", "horizon": 1, "features": features}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"train_rows={len(train_df)} val_rows={len(val_df)} test_rows={len(test_df)}")
    print(f"metrics.val={metrics['val']}")
    print(f"metrics.test={metrics['test']}")


if __name__ == "__main__":
    main()
