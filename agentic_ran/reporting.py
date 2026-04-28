from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_predictions(
    path: Path,
    y_true,
    y_pred,
    sample_id=None,
    global_index=None,
    timestamp=None,
    source_file=None,
    scenario: str = "",
    model_type: str = "",
) -> None:
    n = len(y_true)
    df = pd.DataFrame(
        {
            "sample_id": sample_id if sample_id is not None else list(range(n)),
            "global_index": global_index if global_index is not None else list(range(n)),
            "Timestamp": timestamp if timestamp is not None else [""] * n,
            "source_file": source_file if source_file is not None else ["unknown"] * n,
            "y_true": y_true,
            "y_pred": y_pred,
            "scenario": [scenario] * n,
            "model_type": [model_type] * n,
        }
    )
    df.to_csv(path, index=False)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_plots(plot_dir: Path, y_true, y_pred, history: list[dict], scenario_name: str):
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    n = min(250, len(y_true))
    plt.plot(y_true[:n], label="ground truth", linewidth=1.5)
    plt.plot(y_pred[:n], label="prediction", linewidth=1.2)
    plt.title(f"Predictions vs Ground Truth ({scenario_name})")
    plt.xlabel("Sample")
    plt.ylabel("Target")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "predictions_vs_truth.png", dpi=150)
    plt.close()

    hist_df = pd.DataFrame(history)
    plt.figure(figsize=(8, 5))
    plt.plot(hist_df["epoch"], hist_df["train_loss"], marker="o", label="train")
    plt.plot(hist_df["epoch"], hist_df["val_loss"], marker="o", label="val")
    plt.title(f"Training Curve ({scenario_name})")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "training_curve.png", dpi=150)
    plt.close()
