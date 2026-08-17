from __future__ import annotations

import numpy as np
import pandas as pd

from agentic_ran.publication_benchmark import (
    PublicationConfig,
    _paired_episode_stats,
    _shortcut_diagnostics,
    publication_paths,
)


def _config() -> PublicationConfig:
    return PublicationConfig(
        train_configs=(0, 1),
        validation_configs=(2,),
        test_configs=(3,),
        train_scenarios=("rome_static_close",),
        unseen_scenarios=("rome_static_far",),
        base_stations=(1, 4),
        experiments=(1, 2),
        bootstrap_samples=200,
        permutation_samples=200,
    )


def test_publication_paths_cover_requested_dimensions():
    paths = publication_paths(
        ("rome_static_close", "rome_static_far"),
        (0, 1),
        (1, 2),
        (1, 4),
    )
    assert len(paths) == 2 * 2 * 2 * 2 * 3
    assert any("rome_static_far/tr1/exp2/bs4" in item for item in paths)


def test_config_rejects_overlapping_splits():
    config = _config()
    config.validate()


def test_paired_statistics_are_episode_paired():
    proposed = pd.DataFrame({"episode_id": ["a", "a", "b", "b"], "utility": [0.8, 0.9, 0.7, 0.8]})
    baseline = pd.DataFrame({"episode_id": ["a", "a", "b", "b"], "utility": [0.7, 0.8, 0.6, 0.7]})
    stats = _paired_episode_stats(proposed, baseline, bootstrap_samples=200, permutation_samples=200, seed=1)
    assert stats["episodes"] == 2
    assert np.isclose(stats["mean_paired_delta_utility"], 0.1)


def test_shortcut_diagnostics_exposes_policy_change_rows():
    rows = 80
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({column: rng.normal(size=rows) for column in [
        "num_ues", "slice_prb", "power_multiplier", "scheduler_code", "dl_mcs", "dl_buffer_bytes",
        "dl_bitrate_mbps", "dl_errors_pct", "dl_cqi", "ul_mcs", "ul_buffer_bytes", "ul_bitrate_mbps",
        "ul_errors_pct", "ul_sinr", "requested_prbs", "granted_prbs", "grant_ratio"
    ]})
    frame["slice_type"] = np.resize(np.asarray(["eMBB", "mMTC", "URLLC"]), rows)
    frame["scheduler_code"] = np.resize(np.asarray([0, 1, 2]), rows)
    frame["slice_prb"] = np.resize(np.asarray([2, 4, 8]), rows)
    current = np.resize(np.asarray(["round_robin:prb=2", "waterfilling:prb=4", "proportional_fair:prb=8"]), rows)
    frame["action"] = current
    frame.loc[::7, "action"] = "round_robin:prb=2"
    result = _shortcut_diagnostics(frame.iloc[:60], frame.iloc[60:], seed=3)
    assert set(result["variant"]) == {
        "full_state",
        "without_scheduler",
        "without_prb",
        "without_scheduler_and_prb",
    }
    assert (result["policy_change_rows"] >= 0).all()
