from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from agentic_ran.offline_policy import STATE_COLUMNS
from agentic_ran.publication_crossfit import run_crossfit_benchmark
from agentic_ran.publication_science import FinalScienceConfig
from agentic_ran.publication_v2 import PubConfig

ACTIONS = [
    ("round_robin:prb=1", 0, 1),
    ("waterfilling:prb=2", 1, 2),
    ("proportional_fair:prb=4", 2, 4),
]
SLICES = ("eMBB", "mMTC", "URLLC")


def _state(slice_type: str, scheduler: int, prb: int, offset: float) -> dict[str, float]:
    bitrate = {"eMBB": 0.78, "mMTC": 0.024, "URLLC": 0.08}[slice_type] + offset
    buffer = {"eMBB": 900.0, "mMTC": 450.0, "URLLC": 180.0}[slice_type] * (1.0 + offset)
    requested = 10.0 + (scheduler % 2)
    granted = min(requested, 7.5 + prb * 0.35 + offset)
    return {
        "num_ues": 10.0,
        "slice_prb": float(prb),
        "power_multiplier": 1.0 + 0.03 * scheduler,
        "scheduler_code": float(scheduler),
        "dl_mcs": 10.0 + scheduler + offset,
        "dl_buffer_bytes": buffer,
        "dl_bitrate_mbps": bitrate,
        "dl_errors_pct": 2.0 + scheduler * 0.3,
        "dl_cqi": 9.0 + scheduler,
        "ul_mcs": 9.0 + scheduler,
        "ul_buffer_bytes": buffer * 0.65,
        "ul_bitrate_mbps": bitrate * 0.7,
        "ul_errors_pct": 1.5 + scheduler * 0.2,
        "ul_sinr": 11.0 + scheduler,
        "requested_prbs": requested,
        "granted_prbs": granted,
        "grant_ratio": granted / requested,
    }


def _rows_for_cell(
    *,
    publication_split: str,
    scenario: str,
    training_config: str,
    experiment: str,
    base_station: str = "bs1",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for slice_index, slice_type in enumerate(SLICES):
        episode = f"{scenario}/{training_config}/{experiment}/{base_station}/{slice_type}"
        for step, (action, next_scheduler, next_prb) in enumerate(ACTIONS):
            _, current_scheduler, current_prb = ACTIONS[(step + 1) % len(ACTIONS)]
            offset = 0.005 * (
                slice_index + step + int(training_config.removeprefix("tr")) % 3
            )
            current = _state(slice_type, current_scheduler, current_prb, offset)
            following = _state(slice_type, next_scheduler, next_prb, offset + 0.01)
            row: dict[str, object] = {
                "episode_id": episode,
                "timestamp_s": step,
                "scenario": scenario,
                "training_config": training_config,
                "experiment": experiment,
                "base_station": base_station,
                "imsi_file": f"synthetic-{slice_type}",
                "source_path": "synthetic",
                "slice_type": slice_type,
                "action": action,
                "reward": 0.5,
                "done": step == len(ACTIONS) - 1,
                "split": "train" if publication_split == "train" else "test",
                "publication_split": publication_split,
                "shift_type": (
                    "rf_distance_shift"
                    if publication_split == "test_unseen"
                    else "seen_condition"
                ),
            }
            row.update(current)
            row.update({f"next_{column}": following[column] for column in STATE_COLUMNS})
            rows.append(row)
    return rows


def _synthetic_publication_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for config in range(6):
        for experiment in ("exp1", "exp2"):
            rows.extend(
                _rows_for_cell(
                    publication_split="train",
                    scenario="rome_static_close",
                    training_config=f"tr{config}",
                    experiment=experiment,
                )
            )
    for config in (6, 7):
        for experiment in ("exp1", "exp2"):
            rows.extend(
                _rows_for_cell(
                    publication_split="validation",
                    scenario="rome_static_close",
                    training_config=f"tr{config}",
                    experiment=experiment,
                )
            )
    for config in (8, 9):
        for experiment in ("exp1", "exp2"):
            rows.extend(
                _rows_for_cell(
                    publication_split="test_seen",
                    scenario="rome_static_close",
                    training_config=f"tr{config}",
                    experiment=experiment,
                )
            )
            rows.extend(
                _rows_for_cell(
                    publication_split="test_unseen",
                    scenario="rome_static_far",
                    training_config=f"tr{config}",
                    experiment=experiment,
                )
            )
    frame = pd.DataFrame(rows)
    numeric = [*STATE_COLUMNS, *[f"next_{name}" for name in STATE_COLUMNS], "reward"]
    assert np.isfinite(frame[numeric].to_numpy(float)).all()
    return frame


def test_crossfit_publication_pipeline_runs_end_to_end_on_synthetic_data(
    tmp_path: Path, monkeypatch,
):
    monkeypatch.setenv("AGENTIC_RAN_EXPERIMENTAL_SUPPORT_GUARD", "1")
    monkeypatch.setenv("AGENTIC_RAN_EVAL_MATCHED_WRAPPERS", "1")
    monkeypatch.setenv("AGENTIC_RAN_EVAL_CONFORMAL_RERANK", "1")
    monkeypatch.setenv("AGENTIC_RAN_EVAL_BEHAVIOR_VARIANTS", "1")
    monkeypatch.setenv("AGENTIC_RAN_EVAL_SAFE_IMPROVEMENT", "1")
    monkeypatch.setenv("AGENTIC_RAN_EVAL_ADAPTIVE_HYBRID", "1")
    monkeypatch.setenv("AGENTIC_RAN_EVAL_JOINT_SLICE_ADVANTAGE", "1")
    # An explicit, test-only PHY manifest replaces the historical
    # data-derived 14-PRB proxy; these are synthetic unit-test values.
    import json
    manifest = tmp_path / "test-only-physical-bwp.json"
    manifest.write_text(json.dumps({
        "schema": "jagc-phy-bwp-v1",
        "cells": {
            "bs1": {
                "backend": "ns3-5glena",
                "carrier_id": "synthetic-test-carrier",
                "bwp_id": "test-BWP",
                "total_prbs": 52,
                "reserved_prbs": 4,
                "numerology": 1,
                "source": "nr_phy_observed",
            }
        }
    }))
    monkeypatch.setenv("AGENTIC_RAN_JAGC_BWP_MANIFEST", str(manifest))
    data = tmp_path / "synthetic.csv.gz"
    output = tmp_path / "publication"
    _synthetic_publication_frame().to_csv(data, index=False, compression="gzip")

    pub_cfg = PubConfig(
        train_configs=(0, 1, 2, 3, 4, 5),
        validation_configs=(6, 7),
        test_configs=(8, 9),
        train_scenarios=("rome_static_close",),
        unseen_scenarios=("rome_static_far",),
        base_stations=(1,),
        experiments=(1, 2),
        bootstrap_samples=200,
        permutation_samples=200,
        random_seed=11,
        fqi_iterations=2,
        gamma=0.95,
        cql_alpha=0.1,
        cql_epochs=4,
        cql_learning_rate=0.01,
    )
    science = FinalScienceConfig(
        ope_config_count=2,
        evaluator_folds=2,
        evaluator_max_iter=32,
        target_validation_ood_fpr=0.10,
        sla_threshold_grid=(0.45,),
        uncertainty_threshold_grid=(0.60,),
        planning_weight_grid=(0.0,),
        switch_margin_grid=(0.01,),
        cql_alpha_grid=(0.10,),
        cql_tuning_epochs=3,
        bootstrap_samples=200,
        permutation_samples=200,
        evaluator_support_gate=0.95,
        latency_samples=1,
        random_seed=11,
    )

    summary = run_crossfit_benchmark(
        data,
        output,
        pub_cfg,
        science,
        seed=11,
        literature_reference={"type": "literature_reference_only", "reported_results": {}},
    )

    assert summary["run_status"] == "EXPERIMENT-COMPLETE"
    assert summary["partition"]["strategy"] == "experiment_role_swap_crossfit"
    assert len(summary["partition"]["folds"]) == 2
    for fold in summary["partition"]["folds"]:
        assert fold["policy_fit_experiment"] not in fold["ope_fit_experiments"]
        assert fold["common_policy_action_coverage"] == 1.0
        assert fold["minimum_common_actions_per_slice"] >= 2
    assert summary["readiness_gates"][
        "proposed evaluator support >= 95% on seen test"
    ]
    assert summary["readiness_gates"][
        "proposed evaluator support >= 95% on unseen test"
    ]
    assert (output / "report.html").is_file()
    assert (output / "tradeoff_summary.json").is_file()
    assert (output / "partition.json").is_file()
    assert (output / "publication_ablations.csv").is_file()
    assert (output / "conditional_support.csv").is_file()
    assert len(pd.read_csv(output / "publication_ablations.csv")) == 20
    assert not pd.read_csv(output / "conditional_support.csv").empty
    assert "state_conditional_overlap" in summary
    assert summary["experimental_support_guard"]["enabled"]
    assert summary["experimental_matched_wrappers"]["enabled"]
    models = set(pd.read_csv(output / "publication_baselines.csv")["model"])
    assert {"support_guarded", "fqi_matched", "cql_linear_matched",
            "behavior_hgb_matched"} <= models
    research = summary["research_conformal_rerank"]
    assert research["enabled"]
    assert research["publication_ready"] is False
    assert research["conformal_calibration_unit"] == "held_out_episode_maximum"
    for split in ("seen", "unseen"):
        assert research["paired_accounting"][split]["rows"] > 0
        assert "fixed_penalty" in research["paired_cluster_statistics"][split]
    assert (output / "research_conformal_cluster_evidence.csv").is_file()
    assert (output / "research_conformal_paired_statistics.json").is_file()
    assert (output / "research_conformal_slice_scenario.csv").is_file()
    assert (output / "research_conformal_validation.csv").is_file()
    assert (output / "research_conformal_final_tests.csv").is_file()
    joint = summary["research_joint_slice_advantage"]
    assert joint["enabled"]
    assert not joint["selection_uses_test"]
    assert len(joint["selection"]) == 2
    assert len(joint["comparison"]) == 4
    assert not joint["publication_ready_as_real_ran_superiority"]
    for selected in joint["selection"]:
        assert not selected["selection_uses_final_test"]
        assert selected["train_only_budget"]["training_groups"] > 0
        assert selected["validation_model_headroom"]["validation_only"]
        assert selected["validation_model_headroom"]["optimistic_model_headroom_absolute"] >= -1e-9
        assert set(selected["alignment"]) == {"validation", "seen", "unseen"}
        assert selected["selected_profile"] in {
            "original_agentic_ran", "joint_utility", "joint_margin_005",
            "joint_margin_010", "joint_uncertainty_025",
            "joint_uncertainty_050", "joint_switch_010",
        }
    for file in (
        "research_joint_slice_validation.csv",
        "research_joint_slice_final_tests.csv",
        "research_joint_slice_decisions.csv.gz",
        "research_joint_slice_selection.json",
    ):
        assert (output / file).is_file()
    joint_rows = pd.read_csv(output / "research_joint_slice_final_tests.csv")
    assert set(joint_rows["split"]) == {"seen", "unseen"}
    assert (joint_rows["represented_slice_rows"] > 0).all()
    joint_decisions = pd.read_csv(output / "research_joint_slice_decisions.csv.gz")
    assert set(joint_decisions["split"]) == {"seen", "unseen"}

    hybrid = summary["research_adaptive_hybrid"]
    assert hybrid["enabled"]
    assert len(hybrid["held_out_selections"]) == 2
    for split in ("seen", "unseen"):
        comparison = hybrid["paired_evidence"][split]["conformal"]
        assert comparison["abstention_penalty_cancels"]
        assert comparison["same_cluster_coverage"]
        assert comparison["candidate"]["rows"] > 0
        assert comparison["paired_cluster_inference"]["cluster_count"] >= 2
    for filename in (
        "research_hybrid_validation.csv",
        "research_hybrid_final_tests.csv",
        "research_hybrid_cluster_evidence.csv",
        "research_hybrid_slice_scenario.csv",
        "research_hybrid_paired_statistics.json",
    ):
        assert (output / filename).is_file()
    hybrid_final = pd.read_csv(output / "research_hybrid_final_tests.csv")
    assert len(hybrid_final) == 4
    assert set(hybrid_final["split"]) == {"seen", "unseen"}
    assert set(hybrid_final["profile"]).issubset(set(hybrid["validation_grid_profiles"]))
    assert len(pd.read_csv(output / "research_hybrid_validation.csv")) == 18
    safe = summary["research_safe_improvement"]
    assert safe["enabled"]
    assert not safe["jev_inference_available"]
    assert len(safe["modes"]) == 4
    assert len(safe["meta_selection"]) == 2
    for mode in safe["modes"]:
        for split in ("seen", "unseen"):
            assert safe["paired_evidence"][mode][split]["accounting"]["rows"] > 0
    for name in (
        "research_safe_improvement_validation.csv",
        "research_safe_improvement_final_tests.csv",
        "research_safe_improvement_cluster_evidence.csv",
        "research_safe_improvement_slice_scenario.csv",
        "research_safe_improvement_paired_statistics.json",
        "research_action_transition_audit.csv",
    ):
        assert (output / name).is_file()
    safe_tests = pd.read_csv(output / "research_safe_improvement_final_tests.csv")
    assert set(safe_tests["mode"]) == set(safe["modes"])
    assert len(safe_tests) == 2 * 2 * len(safe["modes"])
    assert all("adjacent_observed_action_changes" in x
               for x in safe["action_transition_audit"].values())

    behavior = summary["research_behavior_variants"]
    assert behavior["enabled"]
    assert set(behavior["modes"]) == {
        "boltzmann", "behavior_clone", "behavior_kl", "behavior_tree",
    }
    assert len(behavior["meta_selection"]) == 2
    assert all(
        value["selected_mode"] in behavior["modes"] + [None]
        for value in behavior["meta_selection"]
    )
    for mode in behavior["modes"]:
        for split in ("seen", "unseen"):
            assert behavior["paired_evidence"][mode][split]["accounting"]["rows"] > 0
    assert (output / "research_behavior_validation.csv").is_file()
    assert (output / "research_behavior_final_tests.csv").is_file()
    assert (output / "research_behavior_cluster_evidence.csv").is_file()
    assert (output / "research_behavior_paired_statistics.json").is_file()
    tests = pd.read_csv(output / "research_behavior_final_tests.csv")
    assert set(tests["mode"]) == set(behavior["modes"])
    assert len(tests) == 2 * 2 * len(behavior["modes"])

    assert "seen:support_guarded_vs_proposed" in summary["clustered_statistics"]
    assert "unseen:proposed_vs_fqi_matched" in summary["clustered_statistics"]
    report = (output / "report.html").read_text(encoding="utf-8")
    assert "Experiment cross-fit and positivity support" in report
    assert "Allowed manuscript claims" in report
    assert "experiment role-swap" in report
