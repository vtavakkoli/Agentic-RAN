"""Independent training-run statistics and evidence-boundary tests."""
from __future__ import annotations

import json

import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("gymnasium")

from agentic_ran.research_ran_gym_campaign import (
    METHODS, SCENARIOS, _sign_flip, campaign, summarize,
)


def _mock_episodes(n: int = 5) -> pd.DataFrame:
    rows = []
    for run in range(n):
        for scenario in SCENARIOS:
            for seed in (5, 6):
                for policy in METHODS:
                    rows.append({
                        "train_run": run, "scenario": scenario, "seed": seed,
                        "policy": policy, "total_reward": 10.0 if policy == "ppo_wsp" else 9.0,
                        "mean_sla_violation_rate": 0.01,
                        "switch_rate": 0.2,
                        "receipt_fraction": 1.0,
                    })
    return pd.DataFrame(rows)


def test_training_seed_bootstrap_does_not_pseudoreplicate_episodes():
    means, scenario, comparisons = summarize(_mock_episodes(), train_seeds=5, seed=2026)
    assert len(means) == 5 * len(METHODS)
    assert len(scenario) == len(SCENARIOS) * len(METHODS)
    assert comparisons["ppo_frozen"]["training_seed_clusters"] == 5
    assert comparisons["ppo_frozen"]["mean_reward_delta"] == pytest.approx(1.0)
    assert comparisons["ppo_frozen"]["one_sided_train_seed_sign_flip_p"] == pytest.approx(1 / 32)
    # A clear bootstrap effect does NOT imply multiple-testing corrected significance.
    assert not comparisons["ppo_frozen"]["significant_positive_gain"]
    assert all(x["sla_nonregression"] for x in comparisons.values())


def test_campaign_rejects_missing_experiment_balancing():
    broken = _mock_episodes()
    broken = broken.drop(broken.index[0])
    with pytest.raises(ValueError, match="unmatched"):
        summarize(broken, train_seeds=5, seed=2026)


def test_campaign_rejects_false_publication_scale(tmp_path):
    with pytest.raises(ValueError, match="research preset"):
        campaign(tmp_path / "no", training_runs=2, test_seeds=4, steps=5)


def test_campaign_train_validate_heldout_export(tmp_path):
    output = tmp_path / "smoke"
    report = campaign(
        output, base_seed=42, training_runs=2, test_seeds=4,
        ppo_updates=1, episodes_per_update=2, logged_episodes=12,
        validation_episodes=4, steps=5, wsp_rounds=1,
        critic_iterations=1, smoke=True,
    )
    assert report["training_runs"] == 2
    assert report["total_executed_episodes"] == 2 * 4 * 4 * len(METHODS)
    assert report["total_executed_steps"] == 2 * 4 * 4 * len(METHODS) * 5
    assert not report["publication_ready_for_ieee_access"]
    assert report["study_artifacts_complete"]
    assert all(item["training_seed_clusters"] == 2 for item in report["comparisons"].values())
    assert (output / "research_report.html").is_file()
    assert (output / "scenario_means.csv").is_file()
    assert (output / "protocol_frozen.json").is_file()
    assert (output / "all_heldout_steps.csv.gz").is_file()
    hashes = json.loads((output / "artifact_sha256.json").read_text())
    assert "html" in hashes and len(hashes["html"]) == 64
    with pytest.raises(FileExistsError, match="empty"):
        campaign(output, smoke=True, training_runs=2)
