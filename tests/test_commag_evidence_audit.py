"""Tests for estimate labeling and overlap denominator integrity."""
import json

import pandas as pd
import pytest

from agentic_ran.research_commag_evidence_audit import audit


def test_offline_audit_never_certifies_causal_effect(tmp_path):
    src = tmp_path / "source"
    src.mkdir()
    pd.DataFrame([
        {"split": "unseen", "method": "proposed", "slice_type": "URLLC",
         "selected_rows": 90, "sampled_rows": 10, "joint_local_overlap": .2},
        {"split": "unseen", "method": "proposed", "slice_type": "URLLC",
         "selected_rows": 10, "sampled_rows": 10, "joint_local_overlap": .8},
    ]).to_csv(src / "conditional_support.csv", index=False)
    pd.DataFrame([{"split": "unseen", "model": "proposed",
                   "estimate_type": "independent_direct_method", "rows": 100,
                   "selected_ope_mean_utility": .7, "evaluator_support_rate": 1.}
                  ]).to_csv(src / "publication_baselines.csv", index=False)
    pd.DataFrame([{"valid_adjacent_change_rows": 0}]).to_csv(
        src / "research_action_transition_audit.csv", index=False)
    for name in ("ood_detection", "latency_profile"):
        (src / f"{name}.json").write_text(json.dumps({}), encoding="utf-8")
    result = audit(src, tmp_path / "out")
    row = result["conditional_overlap"][0]
    assert row["sample_weighted_overlap"] == pytest.approx(.5)
    assert row["selected_row_weighted_overlap"] == pytest.approx(.26)
    assert result["unsupported_selected_action_outcomes"] == "unidentified"
    assert not result["can_identify_dynamic_policy_effects"]
