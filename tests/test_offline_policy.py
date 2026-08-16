from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from agentic_ran.offline_policy import (
    STATE_COLUMNS,
    StudyConfig,
    evaluate_bundle,
    prepare_report,
    run_seed_study,
    train_bundle,
    train_study,
    validate_dataset,
)


def _fixture(episodes_per_split: int = 4, rows_per_episode: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    actions = ["round_robin:prb=4", "waterfilling:prb=8", "proportional_fair:prb=12"]
    schedulers = [0, 1, 2]
    prbs = [4, 8, 12]
    slices = ["eMBB", "mMTC", "URLLC"]
    rows = []
    for split, experiment in (("train", "exp1"), ("test", "exp2")):
        for episode_number in range(episodes_per_split):
            slice_type = slices[episode_number % len(slices)]
            for t in range(rows_per_episode):
                current_idx = (t + episode_number) % 3
                next_idx = (t + episode_number + (1 if t % 7 == 0 else 0)) % 3
                requested = 20 + (t % 9)
                granted = min(requested, 14 + prbs[next_idx] + (t % 4))
                row = {
                    "episode_id": f"{split}/ep{episode_number}",
                    "timestamp_s": t,
                    "scenario": "fixture",
                    "training_config": f"tr{episode_number}",
                    "experiment": experiment,
                    "base_station": "bs1",
                    "imsi_file": f"ue{episode_number}",
                    "source_path": "fixture.csv",
                    "slice_type": slice_type,
                    "action": actions[next_idx],
                    "reward": 0.6 + 0.1 * np.sin(t / 4),
                    "done": t == rows_per_episode - 1,
                    "split": split,
                }
                current = {
                    "num_ues": 8 + episode_number,
                    "slice_prb": prbs[current_idx],
                    "power_multiplier": 1.0 + 0.05 * current_idx,
                    "scheduler_code": schedulers[current_idx],
                    "dl_mcs": 8 + current_idx + np.sin(t / 5),
                    "dl_buffer_bytes": 80 + (t * 7) % 120,
                    "dl_bitrate_mbps": 0.2 + 0.25 * current_idx + 0.02 * (t % 5),
                    "dl_errors_pct": (t + current_idx) % 5,
                    "dl_cqi": 7 + current_idx,
                    "ul_mcs": 5 + current_idx,
                    "ul_buffer_bytes": 20 + (t % 7),
                    "ul_bitrate_mbps": 0.05 + 0.01 * (t % 4),
                    "ul_errors_pct": (t + 1) % 4,
                    "ul_sinr": 9 + current_idx + np.cos(t / 6),
                    "requested_prbs": requested,
                    "granted_prbs": max(1, granted - 2),
                    "grant_ratio": max(1, granted - 2) / requested,
                }
                next_state = dict(current)
                next_state.update(
                    {
                        "slice_prb": prbs[next_idx],
                        "power_multiplier": 1.0 + 0.05 * next_idx,
                        "scheduler_code": schedulers[next_idx],
                        "dl_mcs": 8 + next_idx + np.sin((t + 1) / 5),
                        "dl_buffer_bytes": max(0, current["dl_buffer_bytes"] + rng.normal(0, 6)),
                        "dl_bitrate_mbps": max(0.01, 0.25 + 0.30 * next_idx + 0.02 * (t % 5)),
                        "requested_prbs": requested,
                        "granted_prbs": granted,
                        "grant_ratio": granted / requested,
                    }
                )
                row.update(current)
                row.update({f"next_{key}": value for key, value in next_state.items()})
                rows.append(row)
    frame = pd.DataFrame(rows)
    assert set(STATE_COLUMNS).issubset(frame.columns)
    return frame


def _small_config() -> StudyConfig:
    return StudyConfig(estimators=12, oof_folds=2, q_iterations=2, robustness_seeds=2, latency_samples=4)


def test_train_select_and_ablations() -> None:
    frame = _fixture()
    validate_dataset(frame)
    train = frame[frame["split"].eq("train")].reset_index(drop=True)
    test = frame[frame["split"].eq("test")].reset_index(drop=True)
    bundle = train_bundle(train, _small_config(), seed=7)
    full, decisions = evaluate_bundle(bundle, test, measure_latency=False)
    no_ood, _ = evaluate_bundle(bundle, test, variant="without_ood_gate", measure_latency=False)
    no_safety, _ = evaluate_bundle(bundle, test, variant="without_safety_gate", measure_latency=False)
    no_plan, _ = evaluate_bundle(bundle, test, variant="without_planning", measure_latency=False)
    no_adapt, _ = evaluate_bundle(bundle, test, variant="without_real_data_adaptation", measure_latency=False)
    assert len(decisions) == len(test)
    assert decisions["selected_action"].notna().all()
    assert 0 <= full["rollback_rate"] <= 1
    assert 0 <= full["policy_churn_rate"] <= 1
    assert all(item["rows"] == len(test) for item in (no_ood, no_safety, no_plan, no_adapt))


def test_seed_study_has_requested_families() -> None:
    frame = _fixture()
    train = frame[frame["split"].eq("train")].reset_index(drop=True)
    test = frame[frame["split"].eq("test")].reset_index(drop=True)
    seeds, ablations, baselines = run_seed_study(train, test, _small_config(), base_seed=10, seeds=2)
    assert len(seeds) == 2
    assert set(ablations["variant"]) >= {
        "full",
        "without_safety_gate",
        "without_ood_gate",
        "without_planning",
        "without_real_data_adaptation",
    }
    assert set(baselines["model"]) >= {
        "logistic_regression",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "multi_objective_offline_policy",
    }


def test_reports_are_written(tmp_path: Path) -> None:
    frame = _fixture()
    data = tmp_path / "transitions.csv.gz"
    frame.to_csv(data, index=False, compression="gzip")
    prepare = tmp_path / "prepare"
    prepare_report(data, None, prepare)
    assert (prepare / "report.html").exists()

    artifacts = tmp_path / "artifacts"
    train_output = tmp_path / "train"
    train_study(data, artifacts, train_output, config=_small_config(), seed=4)
    assert (artifacts / "offline_policy_bundle.joblib").exists()
    assert (train_output / "report.html").exists()
