"""Behavior-constrained, Boltzmann and behavior-tree controller regressions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agentic_ran.publication_support import CONTEXT_COLUMNS
from agentic_ran.research_behavior import (
    MODES, BehaviorPrior, select_behavior, stable_uniforms,
)


def _inputs(n=20):
    scores = np.tile([0.9, 0.85, 0.5], (n, 1))
    risk = np.tile([0.15, 0.12, 0.8], (n, 1))
    unc = np.full((n, 3), 0.1)
    allowed = np.ones((n, 3), bool)
    support = np.ones((n, 3), bool)
    prior = np.tile([0.2, 0.7, 0.1], (n, 1))
    incumbents = np.ones(n, dtype=int)
    return scores, risk, unc, allowed, support, prior, incumbents


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_respects_support_and_predicted_risk(mode):
    args = _inputs()
    selected = select_behavior(
        *args, mode=mode, temperature=0.07,
        uniforms=np.linspace(0.01, 0.99, 20) if mode == "boltzmann" else None,
    )
    assert not selected.abstain.any()
    assert set(selected.actions).issubset({0, 1})
    assert (selected.eligible_counts >= 1).all()
    assert (selected.probability_entropy >= 0).all()


@pytest.mark.parametrize("mode", MODES)
def test_no_fallback_bypasses_common_gate(mode):
    score, risk, unc, allowed, support, prior, inc = _inputs(7)
    support[:, :] = False
    selected = select_behavior(
        score, risk, unc, allowed, support, prior, inc,
        mode=mode, uniforms=np.full(7, 0.3) if mode == "boltzmann" else None,
    )
    assert selected.abstain.all()
    assert (selected.actions == -1).all()


def test_behavior_tree_prioritizes_verified_incumbent_not_unsafe_hold():
    score, risk, unc, allowed, support, prior, inc = _inputs(2)
    score[0] = [0.85, 0.848, 0.7]
    score[1] = [0.95, 0.84, 0.7]
    selected = select_behavior(
        score, risk, unc, allowed, support, prior, inc,
        mode="behavior_tree", hold_margin=0.03,
    )
    assert selected.actions.tolist() == [1, 0]
    # Never hold a policy action excluded from support.
    support[0, 1] = False
    selected2 = select_behavior(
        score, risk, unc, allowed, support, prior, inc,
        mode="behavior_tree", hold_margin=0.03,
    )
    assert selected2.actions[0] == 0


def test_behavior_clone_and_kl_can_differ_from_unregularized_value_policy():
    score, risk, unc, allowed, support, prior, inc = _inputs(3)
    clone = select_behavior(
        score, risk, unc, allowed, support, prior, inc, mode="behavior_clone",
    )
    regularized = select_behavior(
        score, risk, unc, allowed, support, prior, inc, mode="behavior_kl",
        temperature=0.2, behavior_weight=1.0,
    )
    assert (clone.actions == 1).all()
    assert (regularized.actions == 1).all()


def test_boltzmann_stochastic_with_reproducible_row_stable_uniforms():
    frame = pd.DataFrame({
        "episode_id": [f"e{i // 3}" for i in range(150)],
        "timestamp_s": [i % 3 for i in range(150)],
        "base_station": ["bs1"] * 150,
    })
    all_u = stable_uniforms(frame, seed=42)
    assert (all_u == np.r_[
        stable_uniforms(frame.iloc[:37], seed=42),
        stable_uniforms(frame.iloc[37:], seed=42),
    ]).all()
    scores, risk, unc, allowed, support, prior, inc = _inputs(150)
    selected = select_behavior(
        scores, risk, unc, allowed, support, prior, inc,
        mode="boltzmann", temperature=0.2, uniforms=all_u,
    )
    assert len(set(selected.actions)) == 2
    assert float(np.mean(selected.probability_entropy)) > 0


def test_behavior_model_only_uses_predecision_features_and_missing_action_zero():
    rng = np.random.default_rng(7)
    data = pd.DataFrame({
        field: rng.uniform(0.2, 1.5, 200) for field in CONTEXT_COLUMNS
    })
    data["slice_type"] = np.where(np.arange(200) % 2, "eMBB", "URLLC")
    data["action"] = np.where(np.arange(200) % 3, "a", "b")
    data["future_sla"] = np.where(data["action"] == "a", 100, -100)
    fit = BehaviorPrior.fit(data, ("a", "b", "never"), seed=7)
    probs = fit.probabilities(data[:10])
    assert probs.shape == (10, 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert (probs[:, 2] == 0).all()
    modified = data.copy()
    modified["future_sla"] *= -1
    assert np.allclose(
        probs, fit.probabilities(modified[:10]),
    )


def test_invalid_zero_temperature_rejected():
    with pytest.raises(ValueError, match="temperature"):
        select_behavior(*_inputs(2), mode="boltzmann", temperature=0)
