"""COMMAG shadow recommendations use actual slice-buffer measurements, not toy rollouts."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from agentic_ran.research_pulse_commag import build_shadow, run


def _fixture():
    cases = (
        ("train", "rome_static_close", "tr0", "exp1", 1,
         {"eMBB": 1000., "mMTC": 400., "URLLC": 100.}),
        ("train", "rome_static_close", "tr0", "exp1", 2,
         {"eMBB": 2000., "mMTC": 450., "URLLC": 200.}),
        ("validation", "rome_static_close", "tr12", "exp1", 3,
         {"eMBB": 100., "mMTC": 450., "URLLC": 2.}),
        ("test_seen", "rome_static_close", "tr15", "exp2", 4,
         {"eMBB": 500., "mMTC": 450., "URLLC": 160.}),
        ("test_unseen", "rome_static_far", "tr15", "exp2", 5,
         {"eMBB": 3000., "mMTC": 450., "URLLC": 0.}),
    )
    rows = []
    for split, scenario, config, exp, second, values in cases:
        for slice_name, buffer in values.items():
            for ue in range(2):
                rows.append({
                    "publication_split": split, "scenario": scenario,
                    "training_config": config, "experiment": exp,
                    "base_station": "bs1", "timestamp_s": second,
                    "slice_type": slice_name,
                    "dl_buffer_bytes": buffer + ue * 0.001,
                    "slice_prb": {"eMBB": 4., "mMTC": 1., "URLLC": 2.}[slice_name],
                    "reward": 0.4 + ue * .05,
                })
    # Missing URLLC in this second: it MUST NOT become an invented joint state.
    rows.append({
        "publication_split": "test_seen", "scenario": "rome_static_close",
        "training_config": "tr15", "experiment": "exp2",
        "base_station": "bs1", "timestamp_s": 99,
        "slice_type": "eMBB", "dl_buffer_bytes": 50,
        "slice_prb": 4, "reward": 0.5,
    })
    return pd.DataFrame(rows)


def test_actual_commag_buffer_priority_and_train_only_scaling():
    data, summary = build_shadow(_fixture())
    assert len(data) == 5
    assert set(data["pulse_mode"]) == {
        "balanced", "urllc_priority", "embb_priority",
    }
    assert data.loc[data.publication_split.eq("validation"), "pulse_mode"].iloc[0] == "balanced"
    assert data.loc[data.publication_split.eq("test_seen"), "pulse_mode"].iloc[0] == "urllc_priority"
    assert data.loc[data.publication_split.eq("test_unseen"), "pulse_mode"].iloc[0] == "embb_priority"
    assert summary["aligned_observation_coverage"]["test_seen"]["alignment_rate"] == .5
    assert summary["dataset_has_actuated_pulse_ran"] is False
    assert summary["ns3_was_executed"] is False
    assert summary["new_colosseum_experiment_was_executed"] is False
    assert summary["reported_counterfactual_utility"] is False
    assert np.allclose(
        data[["target_eMBB_share", "target_mMTC_share", "target_URLLC_share"]].sum(axis=1),
        1.0,
    )
    assert set(data["reward_source"]) == {
        "recorded_commag_actions_only_not_pulse"
    }


def test_holdout_reward_cannot_change_shadow_decisions():
    frame = _fixture()
    original, first = build_shadow(frame)
    shifted = frame.copy()
    shifted.loc[shifted.publication_split != "train", "reward"] = -123.
    changed, second = build_shadow(shifted)
    assert original.pulse_mode.tolist() == changed.pulse_mode.tolist()
    assert first["train_only_calibration"] == second["train_only_calibration"]
    # Even held-out buffer shifts cannot affect the TRAIN-only byte scales.
    shifted.loc[shifted.publication_split == "test_unseen", "dl_buffer_bytes"] += 100.
    _, third = build_shadow(shifted)
    assert first["train_only_calibration"] == third["train_only_calibration"]


def test_shadow_rejects_invalid_dataset_and_missing_joint_slices():
    frame = _fixture()
    with pytest.raises(ValueError, match="quantile"):
        build_shadow(frame, train_quantile=.2)
    with pytest.raises(ValueError, match="missing columns"):
        build_shadow(frame.drop(columns="dl_buffer_bytes"))
    bad = frame.copy()
    bad.loc[0, "dl_buffer_bytes"] = np.inf
    with pytest.raises(ValueError, match="nonfinite"):
        build_shadow(bad)
    only_embb = frame[frame.slice_type == "eMBB"]
    with pytest.raises(ValueError, match="No aligned"):
        build_shadow(only_embb)


def test_real_prepared_dataset_path_and_outputs(tmp_path):
    path = tmp_path / "commag_publication_transitions.csv.gz"
    _fixture().to_csv(path, compression="gzip", index=False)
    output = tmp_path / "pulse"
    summary = run(path, output)
    assert summary["study"] == "PULSE-RAN-COMMAG-shadow"
    assert (output / "pulse_commag_shadow_recommendations.csv.gz").is_file()
    text = (output / "pulse_commag_shadow_summary.json").read_text()
    assert json.loads(text)["recommendation_counts"]["test_unseen"]["embb_priority"] == 1
    with pytest.raises(FileNotFoundError, match="prepare-full-commag"):
        run(tmp_path / "missing.csv.gz", output)
