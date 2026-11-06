"""Regression tests for the adaptive hybrid and matched counterfactual diagnostics."""
from __future__ import annotations

import numpy as np
import pytest

from agentic_ran.research_hybrid import (
    HYBRID_PROFILES, HybridConfig, select_hybrid, select_validation_profile,
)
from agentic_ran.research_hybrid_evidence import compare_cluster_policies
from agentic_ran.research_support import rerank


def _case(n: int = 48):
    rng = np.random.default_rng(101)
    score = rng.uniform(0.6, 0.92, size=(n, 3))
    sla = rng.uniform(0.08, 0.65, size=(n, 3))
    unc = rng.uniform(0.01, 0.7, size=(n, 3))
    allow = rng.random((n, 3)) > 0.1
    support = rng.random((n, 3)) > 0.18
    incumbent = rng.choice([-1, 0, 1, 2], size=n)
    slices = np.array(["URLLC", "mMTC", "eMBB"])[np.arange(n) % 3]
    ood = rng.uniform(0, 0.8, size=n)
    behavior = np.tile([0.2, 0.5, 0.3], (n, 1))
    return score, sla, unc, allow, support, incumbent, slices, ood, behavior


def _run(profile: HybridConfig, data=None):
    score, sla, unc, allow, support, inc, slices, ood, behavior = (
        _case() if data is None else data
    )
    return select_hybrid(
        score, sla, unc, allow, support, inc, slices, ood,
        profile=profile, max_sla=0.5, max_uncertainty=0.55,
        max_ood=0.6, behavior_prior=behavior,
    )


def test_exact_conformal_anchor_including_invalid_incumbents_and_ood():
    score, sla, unc, allow, support, inc, slices, ood, behavior = _case()
    original = rerank(
        score, sla, unc, allow & (ood[:, None] <= 0.6), support,
        inc, method="test", alpha=0.2,
        max_sla=0.5, max_uncertainty=0.55, switch_cost=0.01,
    )
    hybrid = _run(HYBRID_PROFILES[0])
    assert np.array_equal(original.actions, hybrid.actions)
    assert np.array_equal(original.abstain, hybrid.abstain)
    assert np.array_equal(original.eligible_counts, hybrid.eligible_counts)


@pytest.mark.parametrize("profile", HYBRID_PROFILES, ids=lambda p: p.name)
def test_all_profiles_fail_closed_and_never_increase_action_coverage(profile):
    score, sla, unc, allow, support, inc, slices, ood, behavior = _case()
    selected = _run(profile)
    anchor = _run(HYBRID_PROFILES[0])
    assert np.array_equal(selected.abstain, anchor.abstain)
    assert np.all(selected.actions[selected.abstain] == -1)
    idx = np.flatnonzero(~selected.abstain)
    if len(idx):
        a = selected.actions[idx]
        assert (sla[idx, a] <= 0.5).all()
        assert (unc[idx, a] <= 0.55).all()
        assert allow[idx, a].all()
        assert support[idx, a].all()
        assert (ood[idx] <= 0.6).all()


def test_ineligible_incumbent_does_not_block_supported_alternative():
    data = list(_case(n=4))
    score, sla, unc, allow, support, inc, slices, ood, behavior = data
    support[:, :] = False
    support[:, 1] = True
    sla[:, :] = 0.15
    unc[:, :] = 0.10
    allow[:, :] = True
    inc[:] = 0
    ood[:] = 0.1
    selected = _run(HybridConfig("hold", hold_margin=0.2), tuple(data))
    assert (selected.actions == 1).all()
    assert not selected.abstain.any()
    assert not selected.eligible_incumbent.any()


def test_incumbent_hysteresis_only_holds_verified_incumbent():
    data = list(_case(n=4))
    score, sla, unc, allow, support, inc, slices, ood, behavior = data
    score[:, :] = [0.74, 0.75, 0.2]
    sla[:, :] = 0.15
    unc[:, :] = 0.10
    allow[:, :] = True
    support[:, :] = True
    inc[:] = 0
    ood[:] = 0.10
    selected = _run(HybridConfig("hold", hold_margin=0.02), tuple(data))
    assert (selected.actions == 0).all()
    assert selected.verified_hold.all()
    support[:, 0] = False
    fallback = _run(HybridConfig("hold", hold_margin=0.02), tuple(data))
    assert (fallback.actions == 1).all()
    assert not fallback.verified_hold.any()


def test_behavior_preferences_do_not_hard_filter_candidate():
    data = list(_case(n=5))
    score, sla, unc, allow, support, inc, slices, ood, prior = data
    score[:, :] = [0.4, 0.9, 0.3]
    prior[:, :] = [0.6, 0.0, 0.4]
    support[:, :] = False
    support[:, 1] = True
    sla[:, :] = 0.10
    unc[:, :] = 0.1
    allow[:, :] = True
    ood[:] = 0.1
    selected = _run(HybridConfig("behavior", behavior_weight=0.5), tuple(data))
    assert (selected.actions == 1).all()


def test_slice_adaptation_can_choose_different_risk_utility_tradeoffs():
    data = list(_case(n=2))
    score, sla, unc, allow, support, inc, slices, ood, prior = data
    score[:, :] = [0.8, 0.87, 0.1]
    sla[:, :] = [0.15, 0.45, 0.2]
    unc[:, :] = 0.1
    support[:, :] = True
    allow[:, :] = True
    inc[:] = -1
    slices[:] = ["URLLC", "eMBB"]
    ood[:] = 0.1
    selected = _run(HybridConfig("slice_risk", risk_weight=0.15), tuple(data))
    assert selected.actions.tolist() == [0, 1]


def test_validation_selector_requires_positive_gain_and_constraints():
    ref = {
        "coverage": 0.9, "unconditional_decision_score": 0.67,
        "conditional_dm_sla_violation": 0.01,
    }
    anchor = {"profile": "conformal_anchor", **ref, "selected_switch_rate": 0.4}
    worse = {"profile": "bad", "coverage": 0.9,
             "unconditional_decision_score": 0.668,
             "conditional_dm_sla_violation": 0.001, "selected_switch_rate": 0.1}
    winner, reason = select_validation_profile([anchor, worse], ref)
    assert winner["profile"] == "conformal_anchor"
    assert reason.startswith("no_validation")
    improvement = {"profile": "good", "coverage": 0.899,
                   "unconditional_decision_score": 0.671,
                   "conditional_dm_sla_violation": 0.011,
                   "selected_switch_rate": 0.3}
    selected, reason = select_validation_profile([anchor, worse, improvement], ref)
    assert selected["profile"] == "good"
    assert "improvement" in reason
    improvement["conditional_dm_sla_violation"] = 0.02
    selected, _ = select_validation_profile([anchor, improvement], ref)
    assert selected["profile"] == "conformal_anchor"


def _cluster(episode: str, score: float, accepted: int = 5):
    return {
        "scenario": "demo", "training_config": episode,
        "experiment": "exp1", "rows": 10,
        "accepted": accepted, "abstained": 10 - accepted,
        "hold_supported_abstentions": 0, "logged_incumbent_abstentions": 0,
        "reference_sum": 7.0, "penalized_sum": score,
        "hold_pair_count": 0, "hold_pair_difference_sum": 0.0,
        "observed_pair_count": 0, "observed_pair_difference_sum": 0.0,
        "accepted_utility_sum": 6.0,
    }


def test_paired_comparison_detects_gain_with_shared_abstention():
    hybrid = [_cluster("a", 7.1), _cluster("b", 7.2)]
    baseline = [_cluster("a", 7.0), _cluster("b", 7.0)]
    summary = compare_cluster_policies(
        hybrid, baseline, seed=3,
        bootstrap_samples=200, permutation_samples=100,
    )
    assert summary["abstention_penalty_cancels"] is True
    assert summary["all_row_score_delta"] == pytest.approx(0.015)
    assert summary["paired_cluster_inference"]["mean_cluster_delta"] > 0
    baseline[0]["accepted"] = 4
    assert not compare_cluster_policies(
        hybrid, baseline, seed=3,
        bootstrap_samples=30, permutation_samples=20,
    )["abstention_penalty_cancels"]
