"""Statistical and calibration regression tests for the selective controller."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agentic_ran.research_support import ConformalActionSupport
from agentic_ran.selective_evidence import (
    accounting_summary, cluster_accounting, combine_chunks,
    paired_cluster_inference,
)


def _frame(n=12):
    return pd.DataFrame({
        "scenario": ["seen"] * n,
        "training_config": np.repeat(["tr1", "tr2"], n // 2),
        "experiment": ["exp1"] * n,
    })


def test_abstention_penalty_hold_dm_and_observed_are_three_distinct_estimands():
    frame = _frame()
    accepted = np.array([True, False, False, True, False, False] * 2)
    selected = np.where(accepted, 0.8, np.nan)
    reference = np.full(len(frame), 0.7)
    incumbent = np.full(len(frame), 0.6)
    supported = np.array([True, True, False, True, True, False] * 2)
    logged_is_incumbent = np.array([True, True, False, True, True, False] * 2)
    logged_observed = np.full(len(frame), 0.65)
    parts = cluster_accounting(
        frame, accepted, selected, reference, incumbent,
        supported, logged_is_incumbent, logged_observed,
    )
    summary = accounting_summary(combine_chunks(parts))
    assert summary["actuation_coverage"] == pytest.approx(1 / 3)
    assert summary["fixed_penalty_score"] == pytest.approx(0.8 / 3 - 0.12 * 2 / 3)
    assert summary["hold_incumbent_evaluator_support_rate"] == pytest.approx(0.5)
    assert summary["observed_logged_incumbent_rate"] == pytest.approx(0.5)
    for estimator in (
        "fixed_penalty", "hold_incumbent_dm", "observed_logged_hold_proxy"
    ):
        test = paired_cluster_inference(
            combine_chunks(parts), estimator,
            seed=12, bootstrap_samples=80, permutation_samples=80,
        )
        assert test["cluster_count"] == 2
        assert test["bootstrap_ci95"] is not None


def test_combine_chunks_uses_actual_cluster_not_chunk_count():
    frame = _frame()
    accepted = np.ones(12, dtype=bool)
    def part(ix):
        sub = frame.iloc[ix]
        return cluster_accounting(
            sub, accepted[ix], np.full(len(ix), 0.9),
            np.full(len(ix), 0.7), np.full(len(ix), np.nan),
            np.zeros(len(ix), bool), np.zeros(len(ix), bool),
            np.full(len(ix), np.nan),
        )
    original = combine_chunks(part(np.arange(12)))
    partitioned = combine_chunks(part(np.arange(0, 4)) + part(np.arange(4, 12)))
    pd.testing.assert_frame_equal(original, partitioned)


def test_missing_incumbent_never_becomes_supported_or_zero_utility():
    frame = _frame()
    acc = np.zeros(len(frame), bool)
    rows = cluster_accounting(
        frame, acc, np.full(len(frame), np.nan),
        np.full(len(frame), 0.5),
        np.full(len(frame), np.nan),
        np.zeros(len(frame), bool),
        np.zeros(len(frame), bool),
        np.full(len(frame), np.nan),
    )
    result = paired_cluster_inference(combine_chunks(rows), "hold_incumbent_dm")
    assert result["matched_rows"] == 0
    assert result["paired_row_coverage"] == 0
    assert result["mean_cluster_delta"] is None


def test_episode_blocked_calibration_and_short_resolution_refusal():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(120, 3))
    slices = np.array(["eMBB"] * 120)
    actions = np.array(["A"] * 120)
    episodes_policy = np.repeat([f"P{i}" for i in range(20)], 6)
    episodes_evaluator = np.repeat([f"E{i}" for i in range(20)], 6)
    model = ConformalActionSupport(
        methods=("gaussian",), max_reference=200, min_group=12,
        calibration_fraction=0.35, seed=15,
    ).fit({
        "policy": (x, slices, actions, episodes_policy),
        "evaluator": (x, slices, actions, episodes_evaluator),
    })
    assert model.calibration_protocol == "held_out_episode_maximum"
    for key, n in model.calibration_units.items():
        assert n <= 20
    # Insufficient independent episodes for 1% conformal resolution.
    assert not model.action_support(
        x[:4], slices[:4], ["A"], "gaussian", 0.01,
    ).any()


def test_cross_role_episode_leakage_is_rejected():
    x = np.zeros((24, 2))
    tags = ["same"] * 24
    roles = {
        name: (x, ["eMBB"] * 24, ["A"] * 24, tags)
        for name in ("policy", "evaluator")
    }
    with pytest.raises(ValueError, match="overlap"):
        ConformalActionSupport().fit(roles)
