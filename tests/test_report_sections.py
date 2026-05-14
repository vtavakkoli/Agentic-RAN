import json
from pathlib import Path

from scripts.aggregate_report import aggregate


def test_sections_exist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model_dir = Path("results/train/artifacts/agentic_residual_mlp")
    model_dir.mkdir(parents=True)
    (model_dir / "status.json").write_text(json.dumps({"status": "success", "error": None}), encoding="utf-8")
    (model_dir / "metrics.json").write_text(json.dumps({
        "r2": 0.9, "rmse": 0.1, "mae": 0.05, "smape": 1.0, "wmape": 1.0,
        "composite_score": 80.0, "action_accuracy": 0.8, "action_macro_f1": 0.75,
        "peak_mae": 0.06, "normal_mae": 0.04, "peak_rmse": 0.08, "peak_r2": 0.7
    }), encoding="utf-8")
    (model_dir / "model_metadata.json").write_text(json.dumps({
        "model_type": "agentic_residual_mlp", "backend": "pytorch", "logical_profile": "agentic",
        "sequence_length": 1, "epochs": 1, "use_agentic_policy": True, "is_agentic_model": True,
        "has_action_head": True, "test_global_indices_hash": "same", "test_y_hash": "same"
    }), encoding="utf-8")
    (model_dir / "data_summary.json").write_text(json.dumps({"rows_per_split": {"train": 1, "val": 1, "test": 1}}), encoding="utf-8")
    (model_dir / "agentic_summary.json").write_text(json.dumps({"average_confidence": 0.9, "action_distribution": {"increase_urllc": 3}}), encoding="utf-8")
    path = aggregate(Path("results"))
    html = path.read_text(encoding="utf-8")
    assert "Executive summary" in html
    assert "Agentic policy section" in html
    assert "Agentic action-head comparison" in html
