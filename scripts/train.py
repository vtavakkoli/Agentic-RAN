from __future__ import annotations

import argparse
import json

from sklearn.preprocessing import StandardScaler

from scripts.pipeline_utils import (
    choose_features,
    default_feature_count,
    fit_model,
    load_source,
    metrics,
    persist_artifacts,
    seed_everything,
    split_dataframe,
    to_supervised,
    ensure_columns,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train model on KPM or synthetic data")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", default="results/model")
    parser.add_argument("--target", choices=["traffic_load", "delta_traffic_load"], default="traffic_load")
    parser.add_argument("--horizon_steps", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--feature_count", type=int, default=default_feature_count())
    parser.add_argument("--model", choices=["gru", "lstm", "ridge", "hgb"], default="ridge")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    raw_df, kind = load_source(args.data_root)
    df = split_dataframe(ensure_columns(raw_df), kind)
    features = choose_features(args.feature_count)
    sup = to_supervised(df, features, args.seq_len, args.horizon_steps, args.target)

    x_cols = [c for c in sup.columns if "__t" in c]
    train = sup[sup["split"] == "train"]
    val = sup[sup["split"] == "val"]
    test = sup[sup["split"] == "test"]

    x_scaler = StandardScaler().fit(train[x_cols])
    y_scaler = StandardScaler().fit(train[["y_true"]])

    x_train = x_scaler.transform(train[x_cols])
    y_train = y_scaler.transform(train[["y_true"]]).reshape(-1)

    model = fit_model(args.model, x_train, y_train, args.seed)

    split_metrics: dict[str, dict[str, float]] = {}
    for name, chunk in [("train", train), ("val", val), ("test", test)]:
        x_scaled = x_scaler.transform(chunk[x_cols])
        pred_scaled = model.predict(x_scaled).reshape(-1, 1)
        y_pred = y_scaler.inverse_transform(pred_scaled).reshape(-1)
        y_true = chunk["y_true"].to_numpy()
        split_metrics[name] = metrics(y_true, y_pred)

    metrics_json = {
        "source_kind": kind,
        "rows": len(df),
        "samples": len(sup),
        "model": args.model,
        "target": args.target,
        "horizon_steps": args.horizon_steps,
        "seq_len": args.seq_len,
        "features": features,
        "metrics": split_metrics,
    }

    persist_artifacts(
        args.out_dir,
        model,
        x_scaler,
        y_scaler,
        x_cols,
        {
            "data_root": args.data_root,
            "source_kind": kind,
            "target": args.target,
            "horizon_steps": args.horizon_steps,
            "seq_len": args.seq_len,
            "feature_count": args.feature_count,
            "model": args.model,
            "seed": args.seed,
        },
        metrics_json,
    )

    print(json.dumps(split_metrics, indent=2))


if __name__ == "__main__":
    main()
