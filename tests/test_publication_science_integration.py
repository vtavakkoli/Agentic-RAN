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


def test_crossfit_publication_pipeline_runs_end_to_end_on_synthetic_data(tmp_path: Path):
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
    report = (output / "report.html").read_text(encoding="utf-8")
    assert "Experiment cross-fit and positivity support" in report
    assert "Allowed manuscript claims" in report
    assert "experiment role-swap" in report
