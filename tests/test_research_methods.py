"""Synthetic scientific regressions for exploratory safe policy methods."""
from __future__ import annotations

import numpy as np

from agentic_ran.research_benchmark import evaluate_support
from agentic_ran.research_planning import ACTIONS, fit_dynamics, next_state, plan
from agentic_ran.research_support import ConformalActionSupport, rerank


def _fit(seed=30):
    rng = np.random.default_rng(seed)
    roles = {}
    for role in ("policy", "evaluator"):
        x = rng.normal(size=(180, 3))
        roles[role] = (
            x, np.repeat(["eMBB"], 180), np.repeat(["A", "B"], 90),
        )
    return ConformalActionSupport(seed=seed).fit(roles)


def test_unseen_actions_fail_closed():
    estimator = _fit()
    x = np.random.default_rng(9).normal(size=(4, 3))
    assert (estimator.pvalues(x, ["eMBB"]*4, "missing", "gaussian") == 0).all()
    assert not estimator.action_support(x, ["eMBB"]*4, ["missing"], "knn").any()


def test_nonfinite_context_is_unsupported():
    estimator = _fit()
    x = np.zeros((2, 3))
    x[0, 1] = np.nan
    assert estimator.pvalues(x, ["eMBB"]*2, "A", "knn")[0] == 0


def test_rerank_chooses_next_eligible_action_and_abstains_if_none():
    choice = rerank(
        np.array([[0.8, 0.9, 0.95], [0.8, 0.9, 0.95]]),
        np.array([[0.3, 0.4, 0.1], [0.6, 0.7, 0.8]]),
        np.zeros((2, 3)),
        np.ones((2, 3), dtype=bool),
        np.array([[True, True, False], [True, True, True]]),
        [0, 0], method="gaussian",
    )
    assert choice.actions.tolist() == [1, -1]
    assert choice.reranked.tolist() == [True, False]
    assert choice.abstain.tolist() == [False, True]


def test_no_unsafe_fallback_ever_issued():
    decision = rerank(
        np.ones((1, 2)), np.ones((1, 2)), np.zeros((1, 2)),
        np.ones((1, 2), bool), np.ones((1, 2), bool),
        [0], method="knn",
    )
    assert decision.actions[0] == -1


def test_action_causal_simulator_and_learned_model():
    state = np.array([2.0, 1.1, 0.9])
    assert next_state(state, ACTIONS[0], 0)[0] > next_state(state, ACTIONS[2], 0)[0]
    model = fit_dynamics(count=900)
    assert model.rmse < 0.13
    for horizon in (1, 2, 4):
        assert plan(state, float(ACTIONS[1]), 0, horizon, model) in (-1, 0, 1, 2)


def test_method_selection_is_validation_only():
    report = evaluate_support(
        seed=17, train_per_role=450, validation=80, test=100,
    )
    assert report["selection"]["winner"] in report["all_results"]["validation"]
    assert 0 <= report["heldout_seen"]["actuation_coverage"] <= 1
    assert 0 <= report["heldout_shifted"]["actuation_coverage"] <= 1
