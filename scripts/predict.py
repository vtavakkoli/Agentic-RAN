from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from scripts.pipeline_utils import ensure_columns, load_source, split_dataframe, to_supervised


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prediction using saved model artifacts")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--reservation", default=None)
    parser.add_argument("--output", default="results/predictions/preds.csv")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    x_cols = json.loads((model_dir / "feature_columns.json").read_text(encoding="utf-8"))

    source = args.reservation if args.reservation else args.input
    raw_df, kind = load_source(source)
    df = split_dataframe(ensure_columns(raw_df), kind)

    base_features = sorted({c.split("__t")[0] for c in x_cols})
    sup = to_supervised(df, base_features, int(config["seq_len"]), int(config["horizon_steps"]), str(config["target"]))

    model = joblib.load(model_dir / "model.joblib")
    x_scaler = joblib.load(model_dir / "x_scaler.joblib")
    y_scaler = joblib.load(model_dir / "y_scaler.joblib")

    x_scaled = x_scaler.transform(sup[x_cols])
    pred_scaled = model.predict(x_scaled).reshape(-1, 1)
    y_pred = y_scaler.inverse_transform(pred_scaled).reshape(-1)

    out_df = pd.DataFrame(
        {
            "time_ms": sup["time_ms"].astype("int64"),
            "y_true": sup["y_true"].astype(float),
            "y_pred": y_pred,
        }
    )
    out_df["error"] = out_df["y_true"] - out_df["y_pred"]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"Saved predictions: {out} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
