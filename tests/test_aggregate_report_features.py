from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import scripts.aggregate_report as aggregate_report


def test_aggregate_report_handles_feature_and_new_metrics(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    scenario = "lightweight-32"
    sdir = Path("results/scenarios") / scenario
    model_dir = sdir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    (sdir / "status.json").write_text(
        json.dumps(
            {
                "scenario_name": scenario,
                "success": True,
                "metrics_path": str(model_dir / "metrics.json"),
                "preds_path": str(sdir / "preds.csv"),
                "dataset_path": "",
                "split_metadata": {"train_start_index": 0, "train_end_index": 5, "val_start_index": 6, "val_end_index": 7, "test_start_index": 8, "test_end_index": 9, "train_rows": 6, "val_rows": 2, "test_rows": 2, "train_pct": 0.6, "val_pct": 0.2, "test_pct": 0.2},
                "end_time": "2024-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "lightweight-32",
                "model_backend": "Ridge",
                "logical_profile": "tabular_baseline",
                "profile_note": "note",
                "features": ["dl_cqi", "ul_sinr"],
                "epochs": 1,
                "split_metadata": {"train_start_index": 0, "train_end_index": 5, "val_start_index": 6, "val_end_index": 7, "test_start_index": 8, "test_end_index": 9, "train_rows": 6, "val_rows": 2, "test_rows": 2, "train_pct": 0.6, "val_pct": 0.2, "test_pct": 0.2},
                "end_time": "2024-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "metrics.json").write_text(
        json.dumps({"test": {"MAE": 1, "RMSE": 1, "MAPE": 1, "sMAPE": 1, "wMAPE": 1, "R2": 0.5}, "val": {}}),
        encoding="utf-8",
    )
    pd.DataFrame({"epoch": [1]}).to_csv(model_dir / "epoch_metrics.csv", index=False)
    pd.DataFrame(
        {"time_ms": [1, 2], "y_true": [1.0, 2.0], "y_pred": [1.1, 1.9], "abs_error": [0.1, 0.1], "pct_error": [10.0, -5.0]}
    ).to_csv(sdir / "preds.csv", index=False)

    Path("results").mkdir(exist_ok=True)
    Path("results/feature_importance.json").write_text(
        json.dumps({"feature_importance": [{"rank": 1, "feature": "dl_cqi", "importance": 0.9}]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(aggregate_report, "supported_scenarios", lambda: [scenario])
    aggregate_report.main()

    out = Path("results/final/report.html").read_text(encoding="utf-8")
    assert "Global Feature Importance" in out
    assert "sMAPE" in out
    assert "wMAPE" in out
    assert "Chronological Time-Series Split" in out

    assert Path("results/final/split_timeline.png").exists()
    status_df = pd.read_csv("results/final/scenario_status.csv")
    assert pd.notna(status_df.loc[0, "benchmark_score"]) if "benchmark_score" in status_df.columns else True
