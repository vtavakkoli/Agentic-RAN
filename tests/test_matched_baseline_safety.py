"""Validation of matched safety and hysteresis wrappers for RL baselines."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from agentic_ran.offline_policy import StudyConfig
from agentic_ran.publication_science import _choose_from_precomputed


def test_external_policy_proposals_use_the_same_safety_gate_as_controller():
    bundle = SimpleNamespace(
        actions=["A", "B"],
        actions_by_slice={"eMBB": {"A", "B"}},
        fallback_by_slice={"eMBB": "A"},
    )
    frame = pd.DataFrame({"slice_type": ["eMBB"] * 4})
    pre = {
        "targets": frame,
        "current": np.array(["A"] * 4),
        "allowed": np.ones((4, 2), dtype=bool),
        "ood": np.array([0.0, 0.0, 0.99, 0.0]),
        "q_norm": np.zeros((4, 2)),
        "predictions": {
            "sla": np.array([[0.49, 0.01],
                             [0.10, 0.95],
                             [0.49, 0.01],
                             [0.10, 0.11]]),
            "energy": np.zeros((4, 2)),
            "stability": np.zeros((4, 2)),
            "uncertainty": np.zeros((4, 2)),
        },
        "temporal_pairs": (
            np.array([], dtype=int), np.array([], dtype=int),
        ),
    }
    config = StudyConfig(
        sla_threshold=0.5, uncertainty_threshold=0.5, ood_threshold=0.5
    )
    decision = _choose_from_precomputed(
        bundle, pre, config, switch_margin=0.02,
        proposed_actions=["B"] * 4,
    )
    assert decision.selected_action.to_list() == ["B", "A", "A", "A"]
    assert decision.fallback_reason.to_list() == [
        "", "safety", "ood", "hysteresis",
    ]
    assert decision.rollback.to_list() == [False, True, True, False]


def test_invalid_external_policy_proposal_is_not_executed():
    bundle = SimpleNamespace(
        actions=["A", "B"],
        actions_by_slice={"eMBB": {"A", "B"}},
        fallback_by_slice={"eMBB": "A"},
    )
    frame = pd.DataFrame({"slice_type": ["eMBB"]})
    pre = {
        "targets": frame, "current": np.array(["A"]),
        "allowed": np.ones((1, 2), dtype=bool),
        "ood": np.array([0.0]), "q_norm": np.zeros((1, 2)),
        "predictions": {
            key: np.zeros((1, 2))
            for key in ("sla", "energy", "stability", "uncertainty")
        },
        "temporal_pairs": (
            np.array([], dtype=int), np.array([], dtype=int),
        ),
    }
    decisions = _choose_from_precomputed(
        bundle, pre, StudyConfig(), 0.0, proposed_actions=["UNSEEN"]
    )
    assert decisions.selected_action.to_list() == ["A"]
    assert decisions.rollback.to_list() == [True]
