from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from oran_sim.config import FEATURE_ORDER, SCENARIOS
from oran_sim.feature_selection import rank_features_by_importance, select_top_k_features, write_feature_importance_artifacts
from oran_sim.model import build_model, get_model_metadata
from oran_sim.sequence_data import make_sequences, sort_by_time
from oran_sim.temporal import TemporalSpec, build_temporal_model, predict_temporal_model, train_temporal_model


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


def _load_splits(csv: Path, seed: int, temporal: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        if temporal:
            full = sort_by_time(full)
        else:
            full = full.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(full)
        a, b = int(0.6 * n), int(0.9 * n)
        train_df, val_df, test_df = full.iloc[:a].copy(), full.iloc[a:b].copy(), full.iloc[b:].copy()

    if temporal:
        return sort_by_time(train_df), sort_by_time(val_df), sort_by_time(test_df)
    return train_df, val_df, test_df


def _prepare_tabular_preprocessor(features: list[str]) -> ColumnTransformer:
    num_features = [c for c in features if c != "scheduling_policy"]
    cat_features = [c for c in features if c == "scheduling_policy"]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
        ],
        remainder="drop",
    )



def main() -> None:
    p = argparse.ArgumentParser(description="Train scenario model")
    p.add_argument("--csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="hgb")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--feature_count", type=int, default=None)
    args = p.parse_args()

    scenario_cfg = SCENARIOS.get(args.model)
    is_temporal = bool(scenario_cfg and scenario_cfg.kind == "temporal")

    csv = Path(args.csv)
    train_df, val_df, test_df = _load_splits(csv, args.seed, is_temporal)

    candidate_features = [c for c in FEATURE_ORDER if c in train_df.columns]
    importance_df = rank_features_by_importance(train_df, candidate_features, random_state=args.seed)
    feature_count = args.feature_count if args.feature_count is not None else len(candidate_features)
    features = select_top_k_features(importance_df, min(feature_count, len(candidate_features)))

    model_meta = get_model_metadata(args.model)
    epochs = int(max(1, args.epochs))
    epoch_rows: list[dict] = []

    if is_temporal:
        requested_seq_len = scenario_cfg.seq_len or 16
        seq_len = max(2, min(requested_seq_len, len(train_df), len(val_df), len(test_df)))

        def encode(df: pd.DataFrame) -> pd.DataFrame:
            x = df[features].copy()
            for c in x.columns:
                if not pd.api.types.is_numeric_dtype(x[c]):
                    x[c] = x[c].astype("category").cat.codes
            return x.fillna(0.0)

        x_train_df = encode(train_df)
        x_val_df = encode(val_df)
        x_test_df = encode(test_df)

        scaler = StandardScaler()
        x_train_np = scaler.fit_transform(x_train_df.to_numpy(dtype=float))
        x_val_np = scaler.transform(x_val_df.to_numpy(dtype=float))
        x_test_np = scaler.transform(x_test_df.to_numpy(dtype=float))

        y_train = train_df["target"].to_numpy(dtype=float)
        y_val = val_df["target"].to_numpy(dtype=float)
        y_test = test_df["target"].to_numpy(dtype=float)

        x_train_seq, y_train_seq = make_sequences(x_train_np, y_train, seq_len)
        x_val_seq, y_val_seq = make_sequences(x_val_np, y_val, seq_len)
        x_test_seq, y_test_seq = make_sequences(x_test_np, y_test, seq_len)

        spec = TemporalSpec(architecture=scenario_cfg.architecture or "lstm", seq_len=seq_len)
        if args.model == "deep-performance":
            spec = TemporalSpec(architecture="lstm", seq_len=seq_len, hidden_sizes=[64, 32])
        elif args.model == "ultra-performance":
            spec = TemporalSpec(architecture="lstm", seq_len=seq_len, hidden_sizes=[96, 64, 32])
        elif args.model == "xlstm-baseline":
            spec = TemporalSpec(architecture="lstm", seq_len=seq_len, hidden_sizes=[64, 64])
        elif args.model == "attention-baseline":
            spec = TemporalSpec(architecture="attention", seq_len=seq_len, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1)
        elif args.model == "liquid-baseline":
            spec = TemporalSpec(architecture="liquid", seq_len=seq_len, hidden_size=64, dt=0.1)

        model = build_temporal_model(input_dim=x_train_seq.shape[-1], spec=spec)
        final_metrics = None
        for epoch in range(1, epochs + 1):
            model = train_temporal_model(model, x_train_seq, y_train_seq, epochs=1)
            train_pred = predict_temporal_model(model, x_train_seq)
            val_pred = predict_temporal_model(model, x_val_seq)
            test_pred = predict_temporal_model(model, x_test_seq)
            train_m = _metrics(y_train_seq, train_pred)
            val_m = _metrics(y_val_seq, val_pred)
            test_m = _metrics(y_test_seq, test_pred)
            final_metrics = {"train": train_m, "val": val_m, "test": test_m}
            epoch_rows.append(
                {
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
            )
            print(f"epoch={epoch}/{epochs} val_MAE={val_m['MAE']:.6f} val_RMSE={val_m['RMSE']:.6f} val_R2={val_m['R2']:.6f}", flush=True)

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_feature_importance_artifacts(importance_df, json_path=Path("results") / "feature_importance.json", csv_path=Path("results") / "feature_importance.csv")
        write_feature_importance_artifacts(importance_df, json_path=out_dir / "feature_importance.json", csv_path=out_dir / "feature_importance.csv")

        torch.save(model.state_dict(), out_dir / "temporal_model.pt")
        joblib.dump(scaler, out_dir / "temporal_scaler.joblib")
        (out_dir / "temporal_spec.json").write_text(json.dumps(spec.__dict__, indent=2), encoding="utf-8")
        (out_dir / "model.joblib").write_text("temporal-model", encoding="utf-8")

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
                    "seq_len": seq_len,
                    "requested_seq_len": requested_seq_len,
                    "temporal": True,
                    "input_shape": [int(seq_len), int(len(features))],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (out_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
        pd.DataFrame(epoch_rows).to_csv(out_dir / "epoch_metrics.csv", index=False)
    else:
        pre = _prepare_tabular_preprocessor(features)
        model = build_model(args.model, args.seed)
        pipe = Pipeline([("pre", pre), ("model", model)])
        x_train, y_train = train_df[features], train_df["target"].to_numpy()
        x_val, y_val = val_df[features], val_df["target"].to_numpy()
        x_test, y_test = test_df[features], test_df["target"].to_numpy()

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
            epoch_rows.append(
                {
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
            )
            print(f"epoch={epoch}/{epochs} val_MAE={val_m['MAE']:.6f} val_RMSE={val_m['RMSE']:.6f} val_R2={val_m['R2']:.6f}", flush=True)

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_feature_importance_artifacts(importance_df, json_path=Path("results") / "feature_importance.json", csv_path=Path("results") / "feature_importance.csv")
        write_feature_importance_artifacts(importance_df, json_path=out_dir / "feature_importance.json", csv_path=out_dir / "feature_importance.csv")
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
                    "temporal": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (out_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
        pd.DataFrame(epoch_rows).to_csv(out_dir / "epoch_metrics.csv", index=False)

    print(f"train_rows={len(train_df)} val_rows={len(val_df)} test_rows={len(test_df)}")


if __name__ == "__main__":
    main()
