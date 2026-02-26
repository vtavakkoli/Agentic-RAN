from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def compute_pct_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return signed percentage error using absolute true-value denominator.

    This keeps sign from (y_true - y_pred) while avoiding sign flips from
    negative targets in the denominator.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    denom = np.abs(y_true)
    pct = np.full(err.shape, np.nan, dtype=float)
    np.divide(err * 100.0, denom, out=pct, where=denom > 1e-6)
    return pct


def main() -> None:
    p = argparse.ArgumentParser(description="Predict using trained model artifacts")
    p.add_argument("--model_dir", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    model = joblib.load(model_dir / "model.joblib")
    features = json.loads((model_dir / "features.json").read_text(encoding="utf-8"))

    df = pd.read_csv(args.csv)
    x = df[features]
    y_true = df["target"].to_numpy()
    y_pred = model.predict(x)
    err = y_true - y_pred
    pct = compute_pct_error(y_true, y_pred)

    out_df = pd.DataFrame(
        {
            "index": np.arange(len(df)),
            "time_ms": df.get("time_ms", pd.Series(np.arange(len(df)))),
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
