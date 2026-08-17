from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agentic_ran.publication_report import render_report
from agentic_ran.publication_science import (
    FinalScienceConfig,
    IndependentOPE,
    _augment_shortcut,
    _paired_group_statistics,
    _score_from_predictions,
    choose_ope_configs,
    load_science_config,
    ood_detection_metrics,
    transition_audit,
)


def _science() -> FinalScienceConfig:
    return FinalScienceConfig(
        ope_config_count=2,
        bootstrap_samples=200,
        permutation_samples=200,
        latency_samples=2,
    )


def test_science_config_loads_and_rejects_invalid_values(tmp_path):
    config = tmp_path / "science.yaml"
    config.write_text(
        """
final_publication:
  ope_config_count: 2
  evaluator_folds: 3
  evaluator_max_iter: 32
  target_validation_ood_fpr: 0.02
  sla_threshold_grid: [0.4, 0.5]
  uncertainty_threshold_grid: [0.4, 0.6]
  planning_weight_grid: [0.0, 0.05]
  switch_margin_grid: [0.0, 0.01]
  cql_alpha_grid: [0.1, 0.2]
  bootstrap_samples: 200
  permutation_samples: 200
  evaluator_support_gate: 0.9
""",
        encoding="utf-8",
    )
    loaded = load_science_config(config)
    assert loaded.ope_config_count == 2
    assert loaded.evaluator_folds == 3
    assert loaded.switch_margin_grid == (0.0, 0.01)

    with pytest.raises(ValueError, match="ope_config_count"):
        FinalScienceConfig(ope_config_count=1).validate()
    with pytest.raises(ValueError, match="evaluator_folds"):
        FinalScienceConfig(evaluator_folds=1).validate()
    with pytest.raises(ValueError, match="target_validation_ood_fpr"):
        FinalScienceConfig(target_validation_ood_fpr=0.0).validate()
    with pytest.raises(ValueError, match="statistical resampling"):
        FinalScienceConfig(bootstrap_samples=20).validate()
    with pytest.raises(ValueError, match="evaluator_support_gate"):
        FinalScienceConfig(evaluator_support_gate=0.0).validate()
    with pytest.raises(ValueError, match="calibration grids"):
        FinalScienceConfig(sla_threshold_grid=()).validate()


def test_ope_partition_is_disjoint_and_coverage_aware():
    rows = []
    actions = ["round_robin:prb=1", "waterfilling:prb=2", "proportional_fair:prb=4"]
    for config in range(6):
        for index, action in enumerate(actions):
            rows.append(
                {
                    "training_config": f"tr{config}",
                    "slice_type": ("eMBB", "mMTC", "URLLC")[index],
                    "action": action,
                }
            )
    partition = choose_ope_configs(pd.DataFrame(rows), 2)
    assert set(partition["policy_fit_configs"]).isdisjoint(partition["ope_fit_configs"])
    assert len(partition["ope_fit_configs"]) == 2
    assert partition["policy_action_coverage"] > 0
    assert partition["ope_action_coverage"] > 0

    with pytest.raises(ValueError, match="not enough training configurations"):
        choose_ope_configs(pd.DataFrame(rows).query("training_config in ['tr0','tr1','tr2']"), 2)


def test_independent_score_uses_selected_action_predictions_and_support():
    frame = pd.DataFrame(
        {
            "episode_id": ["e1", "e2"],
            "scenario": ["rome_static_close", "rome_static_close"],
            "training_config": ["tr15", "tr15"],
            "experiment": ["exp1", "exp1"],
            "base_station": ["bs1", "bs1"],
            "slice_type": ["eMBB", "URLLC"],
            "scheduler_code": [0, 1],
            "slice_prb": [1, 2],
        }
    )
    actions = ["round_robin:prb=1", "waterfilling:prb=2"]
    evaluator = IndependentOPE(
        actions=actions,
        energy_scale=(0.0, 1.0),
        models={},
        support_by_slice={"eMBB": {actions[0]}, "URLLC": {actions[1]}},
        fit_summary={},
    )
    predictions = {
        "sla": np.asarray([[0.1, 0.8], [0.7, 0.2]], dtype=float),
        "energy": np.asarray([[0.3, 0.9], [0.8, 0.4]], dtype=float),
        "stability": np.asarray([[0.2, 0.9], [0.8, 0.1]], dtype=float),
    }
    summary, detail = _score_from_predictions(
        evaluator,
        frame,
        actions,
        predictions,
        # use defaults so utility weights and SLA threshold match production logic
        __import__("agentic_ran.offline_policy", fromlist=["StudyConfig"]).StudyConfig(),
        "test_policy",
        rollback=[False, True],
        ood_score=[0.1, 0.7],
    )
    assert summary["estimate_type"] == "independent_direct_method"
    assert summary["evaluator_support_rate"] == 1.0
    assert summary["rollback_rate"] == 0.5
    assert detail["evaluator_supported"].all()
    assert detail["cluster_id"].nunique() == 1
    assert detail["ood_score"].tolist() == [0.1, 0.7]
    assert (detail["utility"] > 0).all()


def test_shortcut_interpretation_distinguishes_missing_change_evidence():
    base = pd.DataFrame([{"variant": "full_state", "agreement_all": 0.8}])
    missing = _augment_shortcut(base, {"within_episode_change_evidence_available": False})
    assert missing.iloc[0]["change_evidence_available"] is np.False_ or not bool(
        missing.iloc[0]["change_evidence_available"]
    )
    assert "persistence/configuration" in missing.iloc[0]["interpretation"]
    present = _augment_shortcut(base, {"within_episode_change_evidence_available": True})
    assert bool(present.iloc[0]["change_evidence_available"])
    assert "policy-change performance" in present.iloc[0]["interpretation"]


def test_clustered_statistics_use_cluster_means_not_ue_rows():
    clusters = [f"c{i}" for i in range(8)]
    proposed_rows = []
    baseline_rows = []
    for index, cluster in enumerate(clusters):
        for _ in range(5 + index):
            proposed_rows.append({"cluster_id": cluster, "utility": 0.72 + index * 0.001})
            baseline_rows.append({"cluster_id": cluster, "utility": 0.70 + index * 0.001})
    result = _paired_group_statistics(
        pd.DataFrame(proposed_rows),
        pd.DataFrame(baseline_rows),
        _science(),
        seed=3,
        group_column="cluster_id",
    )
    assert result["groups"] == 8
    assert np.isclose(result["mean_cluster_delta_utility"], 0.02)
    assert result["bootstrap_ci95_low"] > 0
    assert result["permutation_method"] == "exact_random_sign"


def test_ood_metrics_separate_seen_and_unseen_scores():
    seen = pd.DataFrame({"ood_score": np.linspace(0.05, 0.35, 80)})
    unseen = pd.DataFrame(
        {
            "ood_score": np.linspace(0.55, 0.95, 80),
            "scenario": np.resize(np.asarray(["rome_static_far", "rome_slow_close"]), 80),
        }
    )
    metrics, sensitivity = ood_detection_metrics(seen, unseen, threshold=0.5)
    assert metrics["overall"]["auroc"] > 0.95
    assert metrics["overall"]["auprc"] > 0.95
    assert metrics["overall"]["seen_detection_rate"] == 0.0
    assert metrics["overall"]["unseen_detection_rate"] == 1.0
    assert set(metrics["per_unseen_scenario"]) == {"rome_static_far", "rome_slow_close"}
    assert len(sensitivity) == 11


def test_transition_audit_detects_scheduler_prb_and_joint_changes():
    frame = pd.DataFrame(
        {
            "scenario": ["rome_static_close"] * 3,
            "training_config": ["tr15"] * 3,
            "experiment": ["exp1"] * 3,
            "base_station": ["bs1"] * 3,
            "episode_id": ["e1"] * 3,
            "scheduler_code": [0, 0, 1],
            "slice_prb": [1, 1, 2],
            "next_scheduler_code": [0, 1, 1],
            "next_slice_prb": [1, 2, 2],
            "action": [
                "round_robin:prb=1",
                "waterfilling:prb=2",
                "waterfilling:prb=2",
            ],
        }
    )
    table, summary = transition_audit(frame)
    assert summary["scheduler_change_rows"] == 1
    assert summary["prb_change_rows"] == 1
    assert summary["joint_action_change_rows"] == 1
    assert summary["within_episode_change_evidence_available"] is True
    assert table.iloc[0]["joint_action_change_rows"] == 1


def test_html_report_is_self_contained_and_contains_svg(tmp_path):
    baselines = pd.DataFrame(
        [
            {
                "split": "seen",
                "model": "proposed",
                "selected_ope_mean_utility": 0.75,
                "selected_ope_sla_violation_rate": 0.10,
                "selected_ope_mean_energy_proxy": 0.68,
                "policy_churn_rate": 0.30,
            },
            {
                "split": "seen",
                "model": "fqi",
                "selected_ope_mean_utility": 0.73,
                "selected_ope_sla_violation_rate": 0.12,
                "selected_ope_mean_energy_proxy": 0.70,
                "policy_churn_rate": 0.50,
            },
        ]
    )
    stats = {
        "seen:proposed_vs_fqi": {
            "mean_cluster_delta_utility": 0.02,
            "bootstrap_ci95_low": 0.01,
            "bootstrap_ci95_high": 0.03,
            "cohens_dz": 0.5,
        }
    }
    ood = {
        "seen": {"by_scenario": {"rome_static_close": {"mean_ood": 0.2, "p95_ood": 0.4}}},
        "unseen": {"by_scenario": {"rome_static_far": {"mean_ood": 0.5, "p95_ood": 0.7}}},
    }
    path = render_report(
        tmp_path,
        summary={
            "run_status": "EXPERIMENT-COMPLETE",
            "evidence_status": "READY-FOR-MANUSCRIPT",
            "readiness_gates": {"independent OPE": True},
            "warnings": [],
        },
        baselines=baselines,
        clustered_statistics=stats,
        ood=ood,
        ood_detection={"overall": {"auroc": 0.9, "auprc": 0.9}},
        transition_audit=pd.DataFrame([{"joint_action_change_rows": 2}]),
        shortcut=pd.DataFrame([{"variant": "full_state", "agreement_all": 0.8}]),
        calibration=pd.DataFrame([{"calibration_score": 0.7}]),
        latency={"end_to_end_p95_ms": 100.0},
        literature_reference={},
    )
    text = path.read_text(encoding="utf-8")
    lower = text.lower()
    assert "<svg" in text
    assert "How the evaluation works" in text
    assert "Cluster-level statistical inference" in text
    assert "<script" not in lower
    assert "src=\"http" not in lower
    assert "href=\"http" not in lower
