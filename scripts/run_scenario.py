from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from agentic_ran.agentic_policy import ACTION_SPACE, recommend_action
from agentic_ran.data_loading import DEFAULT_TARGET_COL, load_dataset, write_data_summary
from agentic_ran.evaluation import evaluate_predictions, predict
from agentic_ran.feature_engineering import FeatureFlags, enrich_features
from agentic_ran.models import create_model
from agentic_ran.preprocessing import build_features, build_features_for_pre_split, split_dataset
from agentic_ran.reporting import save_json, save_plots, save_predictions
from agentic_ran.scenarios import SCENARIOS
from agentic_ran.training import train_model


def _pseudo_actions(df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.asarray([], dtype=np.int64)
    prb_pressure = df.get("prb_pressure", pd.Series(np.zeros(len(df), dtype=float))).to_numpy(dtype=float)
    ratio = df.get("ratio_granted_req", pd.Series(np.ones(len(df), dtype=float))).to_numpy(dtype=float)
    buffer_dl = df.get("buffer_pressure_dl", pd.Series(np.zeros(len(df), dtype=float))).to_numpy(dtype=float)
    traffic = df.get("traffic_class", pd.Series(["unknown"] * len(df))).astype(str).to_numpy()

    actions = np.zeros(len(df), dtype=np.int64)
    actions[(prb_pressure > 1.3) & (traffic == "eMBB")] = 1
    actions[(prb_pressure > 1.3) & (traffic == "MTC")] = 2
    actions[(prb_pressure > 1.2) & (traffic == "URLLC")] = 3
    actions[(ratio < 0.85) & (traffic == "eMBB")] = 1
    actions[(ratio < 0.85) & (traffic == "MTC")] = 2
    actions[(ratio < 0.90) & (traffic == "URLLC")] = 3
    actions[(prb_pressure < 0.75) & (traffic == "eMBB")] = 4
    actions[(prb_pressure < 0.75) & (traffic == "MTC")] = 5
    actions[(prb_pressure < 0.80) & (traffic == "URLLC")] = 6
    actions[(buffer_dl > np.quantile(buffer_dl, 0.90)) & (traffic == "URLLC")] = 8
    return actions


def _align_actions(actions: np.ndarray, sequence_length: int) -> np.ndarray:
    if sequence_length <= 1:
        return actions
    return actions[sequence_length:]


def _write_agentic_outputs(
    results_dir: Path,
    test_raw: pd.DataFrame,
    y_eval: np.ndarray,
    y_pred: np.ndarray,
    action_pred: np.ndarray,
    confidence: np.ndarray,
) -> None:
    rows = []
    for i in range(len(y_eval)):
        base = test_raw.iloc[i] if i < len(test_raw) else {}
        act = ACTION_SPACE.get(int(action_pred[i]), ACTION_SPACE[0])
        rows.append(
            {
                "Timestamp": base.get("Timestamp", ""),
                "experiment_second": base.get("experiment_second", np.nan),
                "slice_id": base.get("slice_id", np.nan),
                "traffic_class": base.get("traffic_class", "unknown"),
                "predicted_throughput": float(y_pred[i]),
                "actual_throughput": float(y_eval[i]),
                "action_id": int(action_pred[i]),
                "action_name": act["action_name"],
                "target_slice": act["target_slice"],
                "recommended_policy": act["recommended_policy"],
                "recommended_prb_delta": act["recommended_prb_delta"],
                "confidence": float(confidence[i]),
                "reason": recommend_action(dict(base)).get("reason", "model-generated action"),
            }
        )
    dec_df = pd.DataFrame(rows)
    dec_df.to_csv(results_dir / "agentic_decisions.csv", index=False)

    summary = {
        "total_decisions": int(len(dec_df)),
        "action_distribution": {str(k): int(v) for k, v in Counter(dec_df["action_name"]).items()},
        "average_confidence": float(dec_df["confidence"].mean()) if not dec_df.empty else 0.0,
        "top_reasons": dec_df["reason"].value_counts().head(5).to_dict() if not dec_df.empty else {},
        "per_slice_action_counts": {str(k): int(v) for k, v in (dec_df.groupby("slice_id")["action_name"].value_counts().to_dict() if not dec_df.empty else {}).items()},
        "per_traffic_class_action_counts": {str(k): int(v) for k, v in (dec_df.groupby("traffic_class")["action_name"].value_counts().to_dict() if not dec_df.empty else {}).items()},
        "labels_note": "Action labels are pseudo-labels generated from an interpretable rule-based policy; they are not operator ground truth.",
    }
    save_json(results_dir / "agentic_summary.json", summary)


def run(
    scenario_name: str,
    target_col: str = DEFAULT_TARGET_COL,
    log_target: bool = False,
    loss: str | None = None,
    peak_weight: float | None = None,
    sequence_length: int | None = None,
    model_type: str | None = None,
    use_time_features: bool = True,
    use_traffic_features: bool = True,
    use_agentic_policy: bool = True,
    use_action_head: bool = True,
) -> None:
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}. Available: {', '.join(SCENARIOS)}")

    base = SCENARIOS[scenario_name]
    config = replace(
        base,
        loss=loss or base.loss,
        peak_weight=base.peak_weight if peak_weight is None else peak_weight,
        sequence_length=sequence_length or base.sequence_length,
        model_type=model_type or base.model_type,
    )

    epochs = int(os.getenv("EPOCHS", "5"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results_dir = Path("results") / scenario_name
    plots_dir = results_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)

    split_dir = Path("shared_data") / "splits"
    split_files = {"train": split_dir / "train.csv", "val": split_dir / "val.csv", "test": split_dir / "test.csv"}
    split_summary = {}
    if (split_dir / "summary.json").exists():
        split_summary = json.loads((split_dir / "summary.json").read_text(encoding="utf-8"))

    scenario_use_time = use_time_features and scenario_name != "without_time_features"
    scenario_use_traffic = use_traffic_features and scenario_name not in {"without_time_features", "with_time_features"}
    scenario_use_agentic = use_agentic_policy and scenario_name in {
        "with_agentic_policy_features",
        "agentic_residual_mlp",
        "agentic_liquid_residual",
        "agentic_sequence_attention",
        "agentic_patch_kan_mixer",
    }

    flags = FeatureFlags(
        use_time_features=scenario_use_time,
        use_traffic_features=scenario_use_traffic,
        use_agentic_policy_features=scenario_use_agentic,
        allow_synthetic_time_index=True,
    )

    if all(path.exists() for path in split_files.values()):
        train_df = enrich_features(pd.read_csv(split_files["train"]), flags=flags)
        val_df = enrich_features(pd.read_csv(split_files["val"]), flags=flags)
        test_df = enrich_features(pd.read_csv(split_files["test"]), flags=flags)
        (x_train, y_train), (x_val, y_val), (x_test, y_test), feat_meta = build_features_for_pre_split(train_df, val_df, test_df, sequence_length=config.sequence_length)
        train_actions = _align_actions(_pseudo_actions(train_df), config.sequence_length)
        val_actions = _align_actions(_pseudo_actions(val_df), config.sequence_length)
        test_actions = _align_actions(_pseudo_actions(test_df), config.sequence_length)
        test_raw_aligned = test_df.iloc[config.sequence_length:].reset_index(drop=True) if config.sequence_length > 1 else test_df.reset_index(drop=True)

        data_summary = {
            "source": "pre_split",
            "source_files_used": split_summary.get("preprocessing", {}).get("files_used", []),
            "source_root": split_summary.get("preprocessing", {}).get("source_root", "dataset"),
            "num_metrics_files_used": split_summary.get("preprocessing", {}).get("metrics_file_count", 0),
            "target_column": split_summary.get("preprocessing", {}).get("target_column", target_col),
            "selected_features": split_summary.get("preprocessing", {}).get("feature_names", []),
            "rows_per_split": split_summary.get("split", {}).get("rows", {}),
            "split_ratio": {"train": 0.60, "val": 0.10, "test": 0.30},
            "log_target": bool(log_target),
        }
    else:
        df, data_summary = load_dataset(Path("shared_data"), target_col=target_col)
        df = enrich_features(df, flags=flags)
        actions = _pseudo_actions(df)
        x, y, feat_meta = build_features(df.drop(columns=["Timestamp"], errors="ignore"), sequence_length=config.sequence_length)
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_dataset(x, y)
        (a_train, a_val, a_test) = split_dataset(_align_actions(actions, config.sequence_length), _align_actions(actions, config.sequence_length))
        train_actions, val_actions, test_actions = a_train[0], a_val[0], a_test[0]
        test_raw_aligned = df.iloc[-len(y_test):].reset_index(drop=True)

    if log_target:
        y_train_model = np.log1p(np.clip(y_train, a_min=0.0, a_max=None))
        y_val_model = np.log1p(np.clip(y_val, a_min=0.0, a_max=None))
        y_test_model = np.log1p(np.clip(y_test, a_min=0.0, a_max=None))
    else:
        y_train_model, y_val_model, y_test_model = y_train, y_val, y_test

    input_dim = x_train.shape[-1]
    model = create_model(config, input_dim=input_dim)

    multi_task = use_action_head and "agentic" in config.model_type
    train_pack = (x_train, y_train_model, train_actions) if multi_task else (x_train, y_train_model)
    val_pack = (x_val, y_val_model, val_actions) if multi_task else (x_val, y_val_model)

    history, q85 = train_model(
        model=model,
        train_set=train_pack,
        val_set=val_pack,
        epochs=epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        device=device,
        log_path=results_dir / "training_log.csv",
        loss_name=config.loss,
        peak_weight=config.peak_weight,
        checkpoint_path=results_dir / "best_model.pt",
    )

    pred_out = predict(model, x_test, device=device)
    if isinstance(pred_out, tuple):
        y_pred_model, action_pred, confidence = pred_out
    else:
        y_pred_model = pred_out
        action_pred = test_actions
        confidence = np.ones_like(y_pred_model, dtype=np.float32) * 0.5

    if log_target:
        y_pred, y_eval = np.expm1(y_pred_model), np.expm1(y_test_model)
    else:
        y_pred, y_eval = y_pred_model, y_test

    is_agentic_model = "agentic" in config.model_type
    metrics = evaluate_predictions(
        y_eval,
        y_pred,
        action_true=test_actions,
        action_pred=action_pred,
        is_agentic_model=is_agentic_model,
    )

    model_metadata = {
        "scenario": scenario_name,
        "model_type": config.model_type,
        "backend": config.backend,
        "logical_profile": config.logical_profile,
        "sequence_length": config.sequence_length,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "loss": config.loss,
        "peak_weight": config.peak_weight,
        "train_q85": q85,
        "epochs": epochs,
        "device": device,
        "target_column": data_summary.get("target_column", target_col),
        "log_target": bool(log_target),
        "use_time_features": scenario_use_time,
        "use_traffic_features": scenario_use_traffic,
        "use_agentic_policy": scenario_use_agentic,
        "use_action_head": use_action_head,
        **feat_meta,
    }

    save_predictions(
        results_dir / "predictions.csv",
        y_eval,
        y_pred,
        sample_id=test_raw_aligned.get("sample_id", pd.Series(range(len(y_eval)))).tolist()[: len(y_eval)],
        global_index=test_raw_aligned.get("global_index", pd.Series(range(len(y_eval)))).tolist()[: len(y_eval)],
        timestamp=test_raw_aligned.get("Timestamp", pd.Series([""] * len(y_eval))).tolist()[: len(y_eval)],
        source_file=test_raw_aligned.get("source_file", pd.Series(["unknown"] * len(y_eval))).tolist()[: len(y_eval)],
        scenario=scenario_name,
        model_type=config.model_type,
    )
    _write_agentic_outputs(results_dir, test_raw_aligned, y_eval, y_pred, action_pred, confidence)
    save_json(results_dir / "metrics.json", metrics)
    save_json(results_dir / "model_metadata.json", model_metadata)
    write_data_summary(results_dir / "data_summary.json", data_summary)
    save_json(results_dir / "status.json", {"status": "success", "error": None})
    save_plots(plots_dir, y_eval, y_pred, history, scenario_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=SCENARIOS.keys())
    parser.add_argument("--target-col", default=DEFAULT_TARGET_COL)
    parser.add_argument("--log-target", action="store_true")
    parser.add_argument("--loss", choices=["mse", "huber", "weighted_huber"], default=None)
    parser.add_argument("--peak-weight", type=float, default=None)
    parser.add_argument("--model", choices=["mlp", "attention", "liquid", "xlstm", "patchtst", "tsmixer", "kan", "agentic_patch_kan_mixer", "residual_mlp", "residual_tcn", "residual_liquid_tcn", "agentic_mlp", "agentic_residual_mlp", "agentic_sequence_model", "agentic_sequence_attention", "agentic_liquid_model", "agentic_liquid_residual"], default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--use-time-features", action="store_true", default=True)
    parser.add_argument("--no-use-time-features", action="store_false", dest="use_time_features")
    parser.add_argument("--use-traffic-features", action="store_true", default=True)
    parser.add_argument("--no-use-traffic-features", action="store_false", dest="use_traffic_features")
    parser.add_argument("--use-agentic-policy", action="store_true", default=True)
    parser.add_argument("--no-use-agentic-policy", action="store_false", dest="use_agentic_policy")
    parser.add_argument("--use-action-head", action="store_true", default=True)
    parser.add_argument("--no-use-action-head", action="store_false", dest="use_action_head")
    args = parser.parse_args()
    run(
        args.scenario,
        target_col=args.target_col,
        log_target=args.log_target,
        loss=args.loss,
        peak_weight=args.peak_weight,
        sequence_length=args.sequence_length,
        model_type=args.model,
        use_time_features=args.use_time_features,
        use_traffic_features=args.use_traffic_features,
        use_agentic_policy=args.use_agentic_policy,
        use_action_head=args.use_action_head,
    )
