from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from agentic_ran.data_loading import DEFAULT_TARGET_COL, load_dataset, write_data_summary
from agentic_ran.evaluation import evaluate_predictions, predict
from agentic_ran.models import create_model
from agentic_ran.preprocessing import build_features, build_features_for_pre_split, split_dataset
from agentic_ran.reporting import save_json, save_plots, save_predictions
from agentic_ran.scenarios import SCENARIOS
from agentic_ran.training import train_model


def run(scenario_name: str, target_col: str = DEFAULT_TARGET_COL, log_target: bool = False) -> None:
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}. Available: {', '.join(SCENARIOS)}")

    config = SCENARIOS[scenario_name]
    epochs = int(os.getenv("EPOCHS", "5"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results_dir = Path("results") / scenario_name
    plots_dir = results_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)

    split_dir = Path("shared_data") / "splits"
    split_files = {
        "train": split_dir / "train.csv",
        "val": split_dir / "val.csv",
        "test": split_dir / "test.csv",
    }

    split_summary_path = split_dir / "summary.json"
    split_summary = {}
    if split_summary_path.exists():
        split_summary = json.loads(split_summary_path.read_text(encoding="utf-8"))

    if all(path.exists() for path in split_files.values()):
        train_df = pd.read_csv(split_files["train"])
        val_df = pd.read_csv(split_files["val"])
        test_df = pd.read_csv(split_files["test"])
        (x_train, y_train), (x_val, y_val), (x_test, y_test), feat_meta = build_features_for_pre_split(
            train_df,
            val_df,
            test_df,
            sequence_length=config.sequence_length,
        )
        data_summary = {
            "source": "pre_split",
            "source_files_used": split_summary.get("preprocessing", {}).get("files_used", []),
            "num_metrics_files_used": split_summary.get("preprocessing", {}).get("metrics_file_count", 0),
            "target_column": split_summary.get("preprocessing", {}).get("target_column", target_col),
            "selected_features": split_summary.get("preprocessing", {}).get("feature_names", []),
            "rows_per_split": split_summary.get("split", {}).get("rows", {}),
            "split_files": {k: str(v) for k, v in split_files.items()},
            "split_ratio": {"train": 0.60, "val": 0.10, "test": 0.30},
            "log_target": bool(log_target),
        }
    else:
        df, data_summary = load_dataset(Path("shared_data"), target_col=target_col)
        x, y, feat_meta = build_features(df, sequence_length=config.sequence_length)
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_dataset(x, y)
        data_summary["log_target"] = bool(log_target)

    if log_target:
        y_train_model = np.log1p(np.clip(y_train, a_min=0.0, a_max=None))
        y_val_model = np.log1p(np.clip(y_val, a_min=0.0, a_max=None))
        y_test_model = np.log1p(np.clip(y_test, a_min=0.0, a_max=None))
    else:
        y_train_model = y_train
        y_val_model = y_val
        y_test_model = y_test

    input_dim = x_train.shape[-1]
    model = create_model(config, input_dim=input_dim)

    history = train_model(
        model=model,
        train_set=(x_train, y_train_model),
        val_set=(x_val, y_val_model),
        epochs=epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        device=device,
        log_path=results_dir / "training_log.csv",
    )

    y_pred_model = predict(model, x_test, device=device)
    if log_target:
        y_pred = np.expm1(y_pred_model)
        y_eval = np.expm1(y_test_model)
    else:
        y_pred = y_pred_model
        y_eval = y_test

    metrics = evaluate_predictions(y_eval, y_pred)

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
        "epochs": epochs,
        "device": device,
        "target_column": data_summary.get("target_column", target_col),
        "log_target": bool(log_target),
        **feat_meta,
    }

    save_predictions(results_dir / "predictions.csv", y_eval, y_pred)
    save_json(results_dir / "metrics.json", metrics)
    save_json(results_dir / "model_metadata.json", model_metadata)
    write_data_summary(results_dir / "data_summary.json", data_summary)
    save_json(results_dir / "status.json", {"status": "success", "error": None})
    save_plots(plots_dir, y_eval, y_pred, history, scenario_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=SCENARIOS.keys())
    parser.add_argument("--target-col", default=DEFAULT_TARGET_COL)
    parser.add_argument("--log-target", action="store_true", help="Train on log1p(target) and inverse-transform predictions before evaluation.")
    args = parser.parse_args()
    run(args.scenario, target_col=args.target_col, log_target=args.log_target)
