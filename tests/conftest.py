from __future__ import annotations

from pathlib import Path

import pytest

from agentic_ran.config import Settings
from agentic_ran.data import generate_dataset, write_dataset
from agentic_ran.model import train_policy_proposer
from agentic_ran.policies import load_policy_catalog
from agentic_ran.service import PolicyService


@pytest.fixture(scope="session")
def trained_service(tmp_path_factory: pytest.TempPathFactory) -> PolicyService:
    root = tmp_path_factory.mktemp("agentic-ran")
    dataset = root / "dataset.csv"
    model = root / "model.joblib"
    results = root / "results"
    write_dataset(generate_dataset(rows=650, seed=11), dataset)
    proposer, _ = train_policy_proposer(generate_dataset(rows=650, seed=11), seed=11)
    proposer.save(model)
    settings = Settings(
        dataset_path=dataset,
        model_path=model,
        policy_config_path=Path("configs/policies.yaml"),
        results_dir=results,
        top_k=4,
        random_seed=11,
        auto_bootstrap=False,
    )
    return PolicyService(settings, proposer, load_policy_catalog(settings.policy_config_path))
