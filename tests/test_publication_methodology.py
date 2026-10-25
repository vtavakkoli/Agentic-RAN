"""Regression tests for publication-methodology safeguards."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from agentic_ran.offline_policy import STATE_COLUMNS, StudyConfig
from agentic_ran.publication_science import _replay_switches, _temporal_pairs
from agentic_ran.publication_support import (
    CONTEXT_COLUMNS,
    StateConditionalSupportAudit,
)
from agentic_ran.publication_v2 import PubConfig, fit_cql, fit_fqi


def test_temporal_switches_do_not_confuse_logged_action_disagreement_with_churn():
    frame = pd.DataFrame(
        {
            "episode_id": ["a", "b", "a", "b", "a"],
            "timestamp_s": [2, 1, 1, 2, 3],
        }
    )
    # Episode a in time order: X, X, Y (one switch).
    # Episode b in time order: X, Y (one switch).
    choices = np.asarray(["X", "X", "X", "Y", "Y"], dtype=object)
    before, after = _temporal_pairs(frame)
    flags, rate, transitions = _replay_switches(choices, (before, after))
    assert transitions == 3
    assert int(flags.sum()) == 2
    assert np.isclose(rate, 2 / 3)
    assert not flags[2]  # First timestamp of episode a.


def _frame(offset: float, *, rows: int = 64, action: str = "A") -> pd.DataFrame:
    rng = np.random.default_rng(19)
    values = {
        key: rng.normal(loc=offset, scale=0.05, size=rows)
        for key in CONTEXT_COLUMNS
    }
    return pd.DataFrame(
        {
            **values,
            "slice_type": ["eMBB"] * rows,
            "action": [action] * rows,
        }
    )


def test_state_conditional_support_distinguishes_near_and_far_contexts():
    train_policy = _frame(0.0)
    train_evaluator = _frame(0.0)
    audit = StateConditionalSupportAudit(
        train_policy, train_evaluator, max_reference=64, max_queries=64
    )
    near = _frame(0.0)
    far = _frame(20.0)
    near_rows = audit.audit(near, {"proposed": ["A"] * 64}, split="seen", fold="A")
    far_rows = audit.audit(far, {"proposed": ["A"] * 64}, split="unseen", fold="A")
    assert len(near_rows) == len(far_rows) == 1
    assert near_rows[0]["joint_local_overlap"] > 0.5
    assert far_rows[0]["joint_local_overlap"] == 0.0
    assert "not causal" in far_rows[0]["interpretation"]


def test_state_conditional_support_reports_missing_role_reference():
    policy = _frame(0.0, action="A")
    evaluator = _frame(0.0, action="B")
    audit = StateConditionalSupportAudit(policy, evaluator)
    result = audit.audit(
        _frame(0.0, action="A"), {"p": ["A"] * 64}, split="seen", fold="A"
    )[0]
    assert result["evaluator_reference_rows"] == 0
    assert result["joint_local_overlap"] == 0


def test_value_baselines_respect_slice_mask_in_bellman_backup():
    n = 40
    data = {
        name: np.linspace(0.1, 1.0, n) for name in STATE_COLUMNS
    }
    data.update({
        f"next_{name}": np.linspace(0.2, 1.1, n)
        for name in STATE_COLUMNS
    })
    data.update({
        "slice_type": ["eMBB"] * n,
        "action": ["A"] * n,
        "_utility_reward": np.linspace(0.3, 0.7, n),
        "done": [False] * (n - 1) + [True],
    })
    train = pd.DataFrame(data)
    bundle = SimpleNamespace(
        config=StudyConfig(estimators=8, min_samples_leaf=2),
        actions=["A", "B"],
        actions_by_slice={"eMBB": {"A"}},
    )
    config = PubConfig(fqi_iterations=2, cql_epochs=3)
    fqi = fit_fqi(train, bundle, config, seed=3)
    assert np.isfinite(fqi.predict(
        __import__("agentic_ran.offline_policy", fromlist=["_candidate_matrix"])
        ._candidate_matrix(train, bundle.actions, "A")
    )).all()
    cql = fit_cql(train, bundle, config, seed=3)
    assert np.isfinite(cql.q(train)).all()
