"""Functional, reproducible held-out tests for the compatible final toy Gym algorithms.

These tests verify executed action-dependent Gym steps, NOT PRB/NR PHY.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("torch")

from agentic_ran.research_ran_gym_final_algorithms import run


def test_all_final_algorithms_actuate_matched_gym_steps(tmp_path):
    report = run(
        tmp_path / "final", seed=123, ppo_updates=1,
        episodes_per_update=2, logged_episodes=12,
        validation_episodes=4, test_seeds=4, steps=5,
        wsp_rounds=1, critic_iterations=1,
    )
    assert report["actual_gym_step_executed"]
    assert report["same_five_action_space"]
    assert report["matched_initial_states_verified"]
    assert report["training_and_validation_frozen_before_test"]
    assert not report["test_used_for_hyperparameter_selection"]
    assert not report["publication_superiority_ready"]
    assert not report["native_nsoran_gym_controllers_tested"]
    assert report["executed_episodes"] == 4 * 4 * 6
    expected = {
        "ppo_frozen", "pulse_ran", "twin_cql",
        "distributional_cvar", "twin_cql_cvar_hybrid",
    }
    assert expected == set(report["paired_wsp_vs"])
    assert all(v["paired_episodes"] == 16 for v in report["paired_wsp_vs"].values())
    assert all(v["independent_seed_clusters"] == 4 for v in report["paired_wsp_vs"].values())
    assert all(v["sla_non_regression"] in (True, False) for v in report["paired_wsp_vs"].values())
    saved = json.loads((tmp_path / "final" / "gym_final_summary.json").read_text())
    assert saved["publication_restriction"]
    assert (tmp_path / "final" / "gym_final_steps.csv.gz").is_file()
    assert (tmp_path / "final" / "ppo_wsp.pt").is_file()


def test_rejects_inadequate_partition_sizes(tmp_path):
    with pytest.raises(ValueError, match="too few independent"):
        run(tmp_path, ppo_updates=1, episodes_per_update=2, logged_episodes=12,
            validation_episodes=4, test_seeds=2, steps=5)
