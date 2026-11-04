"""Incumbent-aware RAN control: test hard-gate integrity and estimand boundaries."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agentic_ran.research_safe_benchmark import run_synthetic_comparison
from agentic_ran.research_action_audit import audit_recorded_actions
from agentic_ran.research_safe_improvement import (
    MODES, select_safe_improvement, validated_jev_hints,
)


def _case(n=40):
    scores = np.tile([0.74, 0.90, 0.73], (n, 1)).astype(float)
    sla = np.tile([0.12, 0.14, 0.60], (n, 1)).astype(float)
    uncertainty = np.tile([0.04, 0.06, 0.10], (n, 1)).astype(float)
    allowed = np.ones((n, 3), bool)
    support = np.ones((n, 3), bool)
    incumbent = np.zeros(n, dtype=int)
    prior = np.tile([0.75, 0.20, 0.05], (n, 1)).astype(float)
    return scores, sla, uncertainty, allowed, support, incumbent, prior


@pytest.mark.parametrize("mode", [x for x in MODES if x != "jev_guided"])
def test_safety_mask_never_bypassed(mode):
    score, sla, unc, allowed, support, incumbent, prior = _case()
    support[:, 1] = False
    selected = select_safe_improvement(
        score, sla, unc, allowed, support, incumbent, mode=mode,
        max_sla=0.5, max_uncertainty=0.5, behavior_prior=prior,
    )
    assert not selected.abstain.any()
    assert set(selected.actions).issubset({0})
    # Disable all actions, including incumbent: never issue unsupported hold.
    support[:] = False
    none = select_safe_improvement(
        score, sla, unc, allowed, support, incumbent, mode=mode,
        max_sla=0.5, max_uncertainty=0.5, behavior_prior=prior,
    )
    assert none.abstain.all()
    assert (none.actions == -1).all()


def test_relative_advantage_does_not_switch_when_not_better():
    score, sla, unc, allowed, support, incumbent, prior = _case()
    selected = select_safe_improvement(
        score, sla, unc, allowed, support, incumbent,
        mode="advantage_lcb", advantage_penalty=5,
        max_sla=0.5, max_uncertainty=0.5,
    )
    assert (selected.actions == incumbent).all()
    assert selected.incumbent_eligible.all()
    assert not selected.switched.any()


def test_relative_advantage_switches_when_large_and_supported():
    score, sla, unc, allowed, support, incumbent, prior = _case()
    selected = select_safe_improvement(
        score, sla, unc, allowed, support, incumbent,
        mode="advantage_lcb", advantage_penalty=0.1,
        max_sla=0.5, max_uncertainty=0.5,
    )
    assert (selected.actions == 1).all()
    assert selected.switched.all()


def test_hold_requires_verified_incumbent_even_when_alternative_exists():
    score, sla, unc, allowed, support, incumbent, prior = _case(3)
    support[:, 0] = False
    for mode in ("advantage_lcb", "behavior_anchored_lcb", "verified_hold"):
        selected = select_safe_improvement(
            score, sla, unc, allowed, support, incumbent, mode=mode,
            max_sla=0.5, max_uncertainty=0.5, behavior_prior=prior,
        )
        assert selected.abstain.all()
        assert (selected.reasons == "unverified_incumbent").all()
    selected = select_safe_improvement(
        score, sla, unc, allowed, support, incumbent,
        mode="constraint_conditioned", max_sla=0.5, max_uncertainty=0.5,
    )
    assert (selected.actions == 1).all()


def test_constraint_conditioning_penalizes_risk_inside_safety_envelope():
    score, sla, unc, allowed, support, incumbent, prior = _case(4)
    sla[:, 1] = 0.48
    high_risk = select_safe_improvement(
        score, sla, unc, allowed, support, incumbent,
        mode="constraint_conditioned", risk_weight=1.0,
        max_sla=0.5, max_uncertainty=0.5,
    )
    assert (high_risk.actions == 0).all()


def test_jev_adapter_rejects_unknown_actions_and_unsafe_hints():
    allowed = ("a", "b")
    assert np.allclose(validated_jev_hints({"a": 0.1}, allowed), [0.1, 0])
    with pytest.raises(ValueError, match="unknown"):
        validated_jev_hints({"dangerous": 0.2}, allowed)
    with pytest.raises(ValueError, match="bounded"):
        validated_jev_hints({"a": float("inf")}, allowed)
    score, sla, unc, allow_mask, support, incumbent, prior = _case(8)
    # Jev can prefer action 1 but cannot revive an unsupported action.
    support[:, 1] = False
    advice = np.tile([0.0, 0.25, 0.0], (8, 1))
    result = select_safe_improvement(
        score, sla, unc, allow_mask, support, incumbent,
        mode="jev_guided", advice_scores=advice, advice_weight=1.0,
        max_sla=0.5, max_uncertainty=0.5,
    )
    assert (result.actions == 0).all()
    with pytest.raises(ValueError, match="externally recorded"):
        select_safe_improvement(
            score, sla, unc, allow_mask, support, incumbent,
            mode="jev_guided", max_sla=0.5, max_uncertainty=0.5,
        )


def test_action_audit_separates_static_episodes_from_config_diversity():
    frame = pd.DataFrame({
        "episode_id": ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
        "slice_type": ["eMBB"] * 12,
        "action": ["A"] * 4 + ["B"] * 4 + ["A", "A", "B", "B"],
        "timestamp_s": [0, 1, 2, 3] * 3,
        "scenario": ["example"] * 12,
    })
    detailed, summary = audit_recorded_actions(frame)
    assert detailed["distinct_actions_across_episodes"].iloc[0] == 2
    assert summary["episodes_by_slice"] == 3
    assert summary["episodes_with_multiple_recorded_actions"] == 1
    assert summary["adjacent_observed_action_changes"] == 1
    assert summary["within_episode_action_effects_identified"] is False


def test_audit_ties_never_count_as_sequential_changes():
    frame = pd.DataFrame({
        "episode_id": ["a", "a", "a"],
        "slice_type": ["URLLC"] * 3,
        "timestamp_s": [1, 1, 2],
        "action": ["A", "B", "B"],
    })
    _, summary = audit_recorded_actions(frame)
    assert summary["timestamp_tie_rows"] == 1
    assert summary["adjacent_observed_action_changes"] == 0


def test_synthetic_comparison_is_reproducible_and_labels_toy_estimands():
    first = run_synthetic_comparison(seed=23, episodes=8, steps=5)
    second = run_synthetic_comparison(seed=23, episodes=8, steps=5)
    assert first == second
    assert first["experiment"] == "synthetic_potential_outcomes_only_not_COMMAG"
    assert len(first["ranking_by_toy_known_hold"]) == 5
    for mode, stats in first["results"].items():
        assert 0 <= stats["coverage"] <= 1
        assert 0 <= stats["toy_sla_violation_rate"] <= 1
        assert np.isfinite(stats["toy_known_hold_score"])
        if mode != "conformal_rerank":
            assert len(stats["paired_known_hold_ci95"]) == 2
