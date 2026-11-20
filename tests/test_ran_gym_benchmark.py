"""Regression tests for the external Gym benchmark and publication gate."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from agentic_ran.research_publication_gate import validate as validate_gate
from agentic_ran.research_ran_gym import paired_statistics, run


def test_paired_statistics_exact_positive_gain():
    rows = [(scenario, seed) for scenario in ("normal", "bursty") for seed in range(6)]
    candidate = pd.DataFrame([{"scenario": sc, "seed": s, "total_reward": 2.0,
                               "mean_sla_violation_rate": 0.0} for sc, s in rows])
    baseline = pd.DataFrame([{"scenario": sc, "seed": s, "total_reward": 1.0,
                              "mean_sla_violation_rate": 0.0} for sc, s in rows])
    result = paired_statistics(candidate, baseline, draws=500)
    assert result["paired_episodes"] == 12
    assert result["mean_total_reward_delta"] == pytest.approx(1.0)
    assert result["bootstrap_ci95"] == [1.0, 1.0]
    assert result["one_sided_sign_flip_p"] < 0.05
    assert result["sla_non_regression"] is True


def test_refuses_unmatched_pairs():
    c = pd.DataFrame([{"scenario": "a", "seed": 1, "total_reward": 3,
                       "mean_sla_violation_rate": 0.0}])
    b = pd.DataFrame([{"scenario": "b", "seed": 1, "total_reward": 1,
                       "mean_sla_violation_rate": 0.0}])
    with pytest.raises(ValueError, match="unmatched"):
        paired_statistics(c, b)


def test_publication_gate_never_accepts_toy():
    offline = {
        "run_status": "EXPERIMENT-COMPLETE", "evidence_status": "READY-FOR-MANUSCRIPT",
        "seen_test_rows": 10, "unseen_test_rows": 10,
    }
    active = {
        "backend": "toy-gym-smoke", "executed_gym_step": True,
        "is_toy_smoke": True, "receipt_enforced": True,
        "all_baselines_significantly_outperformed": True,
    }
    result = validate_gate(offline, active)
    assert result["gate_passed"] is False
    assert result["checks"]["RAN Gym is an external backend"] is False


def test_external_config_requires_artifact_and_mapping(tmp_path):
    config = {
        "schema": "agentic-ran-gym-benchmark-v1",
        "backend": "ns-o-ran-gym",
        "environment_factory": "agentic_ran.research_ran_gym_adapters:make_nsoran_traffic_steering",
        "policies": {
            "candidate": "agentic_ran.research_ran_gym_adapters:make_toy_queue_policy",
            "noop": "agentic_ran.research_ran_gym_adapters:make_noop_policy",
        },
        "scenarios": ["scenario-one"], "test_seeds": [1, 2],
        "max_steps": 4, "simulator_revision": "dummy",
        "observation_action_mapping_documented": False,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        run(config_path, tmp_path / "output")
    config["observation_action_mapping_documented"] = True
    config["frozen_model_sha256"] = "0" * 64
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="frozen_model_path"):
        run(config_path, tmp_path / "output")


def test_toy_gym_smoke_produces_realized_steps_and_never_publication_pass(tmp_path):
    pytest.importorskip("gymnasium")
    config = {
        "schema": "agentic-ran-gym-benchmark-v1",
        "backend": "toy-gym-smoke",
        "environment_factory": "agentic_ran.research_ran_gym_adapters:make_toy_env",
        "policies": {
            "candidate": "agentic_ran.research_ran_gym_adapters:make_toy_queue_policy",
            "noop": "agentic_ran.research_ran_gym_adapters:make_noop_policy",
            "random": "agentic_ran.research_ran_gym_adapters:make_random_policy",
        },
        "scenarios": ["normal", "bursty"],
        "test_seeds": [101, 102], "max_steps": 6,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    result = run(path, tmp_path / "out")
    assert result["episodes"] == 12
    assert result["steps"] == 72
    assert result["receipt_enforced"] is False
    assert result["publication_superiority_ready"] is False
    assert result["service_metrics_complete"] is False
    first = pd.read_csv(tmp_path / "out" / "ran_gym_episodes.csv")
    assert len(first) == 12
    run(path, tmp_path / "out2")
    second = pd.read_csv(tmp_path / "out2" / "ran_gym_episodes.csv")
    assert first["total_reward"].to_list() == second["total_reward"].to_list()


def test_toy_smoke_cannot_force_superiority(tmp_path):
    pytest.importorskip("gymnasium")
    config = {
        "schema": "agentic-ran-gym-benchmark-v1",
        "backend": "toy-gym-smoke",
        "environment_factory": "agentic_ran.research_ran_gym_adapters:make_toy_env",
        "policies": {
            "candidate": "agentic_ran.research_ran_gym_adapters:make_toy_queue_policy",
            "noop": "agentic_ran.research_ran_gym_adapters:make_noop_policy",
        },
        "scenarios": ["normal"], "test_seeds": [101, 102], "max_steps": 3,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(RuntimeError, match="NOT demonstrated"):
        run(path, tmp_path / "out", require_superiority=True)
    assert (tmp_path / "out" / "ran_gym_summary.json").exists()
