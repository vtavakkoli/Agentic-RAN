"""Regression tests for Agentic-RAN v3 joint-slice advantage-gated research."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from agentic_ran.offline_policy import STATE_COLUMNS
from agentic_ran.research_joint_slice_advantage import (
    GROUP_KEYS, PROFILES, JointInputs, aligned_joint_frame,
    evaluate_joint_profile, exact_multiple_choice_knapsack,
    paired_joint_clusters, parse_prb_costs, physical_prb_budgets,
)


def _frame():
    rows = []
    for split, tr, group_count in (
        ("train", "tr0", 4), ("validation", "tr12", 2),
        ("test_seen", "tr15", 2), ("test_unseen", "tr16", 2),
    ):
        for second in range(group_count):
            for sl in ("eMBB", "mMTC", "URLLC"):
                for ue in range(2):
                    row = {
                        "publication_split": split,
                        "scenario": "rome_static_far" if split == "test_unseen"
                        else "rome_static_close",
                        "training_config": tr, "experiment": "exp1",
                        "base_station": "bs1", "timestamp_s": second,
                        "slice_type": sl, "action": "round_robin:prb=2",
                        "done": second == group_count - 1,
                    }
                    for col in STATE_COLUMNS:
                        row[col] = 2.0 if col == "slice_prb" else (
                            3.0 + second + ue + len(sl) * .2
                        )
                        row[f"next_{col}"] = row[col] + 0.10
                    rows.append(row)
    return pd.DataFrame(rows)


def test_alignment_averages_real_ue_data_and_never_invents_missing_slice():
    frame = _frame()
    rep, report = aligned_joint_frame(frame[frame.publication_split == "train"])
    assert len(rep) == 12
    assert report["complete_bs_seconds"] == 4
    assert rep["joint_group"].value_counts().eq(3).all()
    assert rep["slice_type"].iloc[:3].tolist() == ["eMBB", "mMTC", "URLLC"]
    original = rep.iloc[0]["dl_buffer_bytes"]
    assert original == pytest.approx(
        frame.query("publication_split=='train' and timestamp_s==0 and slice_type=='eMBB'")[
            "dl_buffer_bytes"
        ].mean()
    )
    incomplete = frame.loc[
        ~(frame.publication_split.eq("train") & frame.timestamp_s.eq(3)
          & frame.slice_type.eq("URLLC"))
    ].copy()
    rep2, report2 = aligned_joint_frame(
        incomplete[incomplete.publication_split == "train"]
    )
    assert len(rep2) == 9
    assert report2["missing_slice_bs_seconds"] == 1
    with pytest.raises(ValueError, match="no matched"):
        aligned_joint_frame(frame[frame.slice_type == "eMBB"])


def test_physical_budget_requires_real_phy_manifest_not_logged_actions(tmp_path):
    full = _frame()
    train = full[full.publication_split.eq("train")]
    with pytest.raises(RuntimeError, match="AGENTIC_RAN_JAGC_BWP_MANIFEST"):
        physical_prb_budgets(train, manifest_path=None)
    path = tmp_path / "bwp.json"
    physical = {
        "schema": "jagc-phy-bwp-v1",
        "cells": {
            "bs1": {
                "backend": "ns3-5glena",
                "carrier_id": "NR-carrier-0",
                "bwp_id": "DL-BWP-0",
                "total_prbs": 52,
                "reserved_prbs": 4,
                "numerology": 1,
                "source": "nr_phy_observed",
            },
        },
    }
    path.write_text(json.dumps(physical))
    budget = physical_prb_budgets(train, manifest_path=path)
    assert budget["budget_by_base_station"]["bs1"] == 48
    assert budget["global_budget"] == 48
    assert budget["budget_units"] == "integer_PRBs_in_reported_single_carrier_BWP"
    assert budget["training_groups"] == 4
    # Changing logged PRB values must NOT change the external PHY capacity.
    modified = train.copy()
    modified["slice_prb"] = 99
    assert physical_prb_budgets(modified, manifest_path=path)["global_budget"] == 48
    physical["cells"]["bs1"]["source"] = "configured_ns3_demo"
    path.write_text(json.dumps(physical))
    with pytest.raises(ValueError, match="physical capacity"):
        physical_prb_budgets(train, manifest_path=path)


def test_knapsack_is_exact_under_shared_budget_and_fails_closed():
    values = np.array([
        [0.2, 0.9, 0.2],
        [0.3, 0.2, 0.9],
        [0.3, 0.8, 0.9],
    ])
    valid = np.ones((3, 3), dtype=bool)
    costs = np.array([1, 2, 4])
    choice, value = exact_multiple_choice_knapsack(values, valid, costs, 4)
    assert choice is not None
    assert costs[choice].sum() <= 4
    assert choice.tolist() == [1, 0, 0] or choice.tolist() == [0, 0, 1]
    assert np.isfinite(value)
    none, score = exact_multiple_choice_knapsack(values, valid, costs, 2)
    assert none is None
    assert score == -np.inf
    valid[1] = False
    none, _ = exact_multiple_choice_knapsack(values, valid, costs, 10)
    assert none is None
    with pytest.raises(ValueError, match="uninterpretable"):
        parse_prb_costs(["my_action"])


def _prepared():
    rep = _frame()
    rep, alignment = aligned_joint_frame(rep[rep.publication_split == "test_seen"])
    n = len(rep) // 3
    scores = np.tile(np.array([
        [0.3, 0.9], [0.3, 0.75], [0.3, 0.75],
    ], dtype=float), (n, 1, 1))
    eval_scores = scores.copy()
    return JointInputs(
        rep, alignment, scores, np.zeros_like(scores),
        np.ones_like(scores, dtype=bool),
        np.zeros((n, 3), dtype=int),
        np.zeros((n, 3), dtype=int),
        eval_scores,
        np.full_like(scores, .1),
        np.ones_like(scores, dtype=bool),
        ["round_robin:prb=1", "round_robin:prb=2"],
        np.array([1, 2]), np.full(n, 4),
    )


def test_joint_gate_improves_without_budget_violation_and_anchor_exact():
    prepared = _prepared()
    anchor, _ = evaluate_joint_profile(
        prepared, PROFILES[0], reference_sla_threshold=.5,
    )
    assert anchor["independent_dm_delta"] == 0.0
    result, decisions = evaluate_joint_profile(
        prepared, PROFILES[1], reference_sla_threshold=.5,
        include_decisions=True,
    )
    assert result["joint_selection_rate"] == 1.0
    assert result["training_proxy_budget_compliance"] == 1.0
    assert result["independent_dm_delta"] > 0.15
    assert decisions is not None
    assert (decisions["selected_budget_proxy_ok"]).all()
    strict, _ = evaluate_joint_profile(
        prepared, type(PROFILES[1])("very_strict", 1.0, 0.0, 0.0),
        reference_sla_threshold=.5,
    )
    assert strict["joint_selection_rate"] == 0.0
    assert strict["independent_dm_delta"] == 0.0
    restricted = _prepared()
    restricted.eligible[:, 2, :] = False
    fail, _ = evaluate_joint_profile(
        restricted, PROFILES[1], reference_sla_threshold=.5,
    )
    assert fail["no_feasible_joint_action_rate"] == 1.0
    assert fail["independent_dm_delta"] == 0.0


def test_cluster_pairing_reports_no_causal_claim_and_is_zero_for_anchor():
    prepared = _prepared()
    _, decisions = evaluate_joint_profile(
        prepared, PROFILES[0], reference_sla_threshold=.5,
        include_decisions=True,
    )
    assert decisions is not None
    stats = paired_joint_clusters(decisions, seed=88, bootstrap_samples=100)
    assert stats["clusters"] == 1
    assert stats["ci95"] is None
    assert stats["unweighted_mean_cluster_delta"] == 0.0
    assert "not_observed_causal" in stats["source"]
