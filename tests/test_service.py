from __future__ import annotations

from pathlib import Path

from agentic_ran.config import Settings
from agentic_ran.service import PolicyService, bootstrap


def test_bootstrap_generates_data_model_and_metrics(tmp_path: Path) -> None:
    settings = Settings(
        dataset_path=tmp_path / "data.csv",
        model_path=tmp_path / "model.joblib",
        policy_config_path=Path("configs/policies.yaml"),
        results_dir=tmp_path / "results",
        top_k=4,
        random_seed=13,
        auto_bootstrap=True,
    )
    model_path, metrics = bootstrap(settings)
    assert model_path.exists()
    assert settings.dataset_path.exists()
    assert (settings.results_dir / "training_metrics.json").exists()
    assert metrics["accuracy"] > 0.7
    loaded = PolicyService.load(settings)
    assert loaded.proposer.metadata.rows > 1_000
