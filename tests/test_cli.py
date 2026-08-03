from __future__ import annotations

from pathlib import Path

from agentic_ran.cli import main


def test_generate_and_train_commands(tmp_path: Path) -> None:
    data = tmp_path / "data.csv"
    model = tmp_path / "model.joblib"
    metrics = tmp_path / "metrics.json"
    assert main(["generate-data", "--output", str(data), "--rows", "180", "--seed", "6"]) == 0
    assert data.exists()
    assert main(["train", "--data", str(data), "--model", str(model), "--metrics", str(metrics), "--seed", "6"]) == 0
    assert model.exists()
    assert metrics.exists()
