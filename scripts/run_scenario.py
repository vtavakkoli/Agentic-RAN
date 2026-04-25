from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import torch

from agentic_ran.data_loading import load_dataset, write_data_summary
from agentic_ran.evaluation import evaluate_predictions, predict
from agentic_ran.models import create_model
from agentic_ran.preprocessing import build_features, build_features_for_pre_split, split_dataset
from agentic_ran.reporting import save_json, save_plots, save_predictions
from agentic_ran.scenarios import SCENARIOS
from agentic_ran.training import train_model


def run(scenario_name: str) -> None:
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
            "files_used": {k: str(v) for k, v in split_files.items()},
            "rows": int(len(train_df) + len(val_df) + len(test_df)),
            "split_ratio": {"train": 0.60, "val": 0.10, "test": 0.30},
        }
    else:
        df, data_summary = load_dataset(Path("shared_data"))
        x, y, feat_meta = build_features(df, sequence_length=config.sequence_length)
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_dataset(x, y)

    input_dim = x_train.shape[-1]
    model = create_model(config, input_dim=input_dim)

    history = train_model(
        model=model,
        train_set=(x_train, y_train),
        val_set=(x_val, y_val),
        epochs=epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        device=device,
        log_path=results_dir / "training_log.csv",
    )

    y_pred = predict(model, x_test, device=device)
    metrics = evaluate_predictions(y_test, y_pred)

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
        **feat_meta,
    }

    save_predictions(results_dir / "predictions.csv", y_test, y_pred)
    save_json(results_dir / "metrics.json", metrics)
    save_json(results_dir / "model_metadata.json", model_metadata)
    write_data_summary(results_dir / "data_summary.json", data_summary)
    save_json(results_dir / "status.json", {"status": "success", "error": None})
    save_plots(plots_dir, y_test, y_pred, history, scenario_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=SCENARIOS.keys())
    args = parser.parse_args()
    run(args.scenario)
