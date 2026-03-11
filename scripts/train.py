from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from oran_sim.config import FEATURE_ORDER
from oran_sim.feature_selection import (
    rank_features_by_importance,
    select_top_k_features,
    write_feature_importance_artifacts,
)
from oran_sim.model import build_model, get_model_metadata


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mape_denom = np.clip(np.abs(y_true), 1e-6, None)
    smape_denom = np.clip(np.abs(y_true) + np.abs(y_pred), 1e-6, None)
    wmape_denom = np.clip(np.sum(np.abs(y_true)), 1e-6, None)
    abs_err = np.abs(y_true - y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / mape_denom)) * 100.0),
        "sMAPE": float(np.mean(2.0 * abs_err / smape_denom) * 100.0),
        "wMAPE": float(np.sum(abs_err) / wmape_denom * 100.0),
        "R2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Train CPU-only baseline model")
    p.add_argument("--csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="hgb")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--feature_count", type=int, default=None)
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

    candidate_features = [c for c in FEATURE_ORDER if c in train_df.columns]
    importance_df = rank_features_by_importance(train_df, candidate_features, random_state=args.seed)

    feature_count = args.feature_count if args.feature_count is not None else len(candidate_features)
    feature_count = min(feature_count, len(candidate_features))
    features = select_top_k_features(importance_df, feature_count)

    if "scheduling_policy" not in features and "scheduling_policy" in train_df.columns and args.feature_count is None:
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

    model = build_model(args.model, args.seed)
    model_meta = get_model_metadata(args.model)
    pipe = Pipeline([("pre", pre), ("model", model)])

    x_train, y_train = train_df[features], train_df["target"].to_numpy()
    x_val, y_val = val_df[features], val_df["target"].to_numpy()
    x_test, y_test = test_df[features], test_df["target"].to_numpy()

    epochs = int(max(1, args.epochs))
    epoch_rows: list[dict] = []
    final_metrics = None
    for epoch in range(1, epochs + 1):
        pipe.fit(x_train, y_train)
        train_pred = pipe.predict(x_train)
        val_pred = pipe.predict(x_val)
        test_pred = pipe.predict(x_test)

        train_m = _metrics(y_train, train_pred)
        val_m = _metrics(y_val, val_pred)
        test_m = _metrics(y_test, test_pred)
        final_metrics = {"train": train_m, "val": val_m, "test": test_m}

        row = {
            "epoch": epoch,
            "train_MAE": train_m["MAE"],
            "train_RMSE": train_m["RMSE"],
            "train_R2": train_m["R2"],
            "val_MAE": val_m["MAE"],
            "val_RMSE": val_m["RMSE"],
            "val_R2": val_m["R2"],
            "test_MAE": test_m["MAE"],
            "test_RMSE": test_m["RMSE"],
            "test_R2": test_m["R2"],
        }
        epoch_rows.append(row)
        print(
            f"epoch={epoch}/{epochs} val_MAE={val_m['MAE']:.6f} "
            f"val_RMSE={val_m['RMSE']:.6f} val_R2={val_m['R2']:.6f}",
            flush=True,
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_feature_importance_artifacts(
        importance_df,
        json_path=Path("results") / "feature_importance.json",
        csv_path=Path("results") / "feature_importance.csv",
    )
    write_feature_importance_artifacts(
        importance_df,
        json_path=out_dir / "feature_importance.json",
        csv_path=out_dir / "feature_importance.csv",
    )
    joblib.dump(pipe, out_dir / "model.joblib")
    (out_dir / "features.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "model_type": args.model,
                "model_backend": model_meta["backend"],
                "logical_profile": model_meta["logical_profile"],
                "profile_note": model_meta["profile_note"],
                "target": "target",
                "horizon": 1,
                "features": features,
                "feature_selection": "random_forest_importance_top_k",
                "epochs": epochs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    pd.DataFrame(epoch_rows).to_csv(out_dir / "epoch_metrics.csv", index=False)

    print(f"train_rows={len(train_df)} val_rows={len(val_df)} test_rows={len(test_df)}")
    print(f"metrics.val={final_metrics['val']}")
    print(f"metrics.test={final_metrics['test']}")


if __name__ == "__main__":
    main()
