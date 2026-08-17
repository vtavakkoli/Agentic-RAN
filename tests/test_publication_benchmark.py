from __future__ import annotations

import numpy as np
import pandas as pd

from agentic_ran.commag import COMMAG_REVISION
from agentic_ran.publication import PAPER_REFERENCE
from agentic_ran.publication_data import discover_tree
from agentic_ran.publication_v2 import PubConfig, filter_paths, paired, shortcut


def _config() -> PubConfig:
    return PubConfig(
        train_configs=(0, 1),
        validation_configs=(2,),
        test_configs=(3,),
        train_scenarios=("rome_static_close",),
        unseen_scenarios=("rome_static_far",),
        base_stations=(1, 4),
        experiments=(1, 2),
        bootstrap_samples=200,
        permutation_samples=200,
        cql_epochs=5,
    )


def test_git_tree_discovery_reuses_cached_listing(tmp_path):
    cache = tmp_path / f".tree-git-{COMMAG_REVISION}.txt"
    expected = [
        "slice_traffic/rome_static_close/tr0/exp1/bs1/slices_bs1/1010123456002_metrics.csv",
        "slice_traffic/rome_static_far/tr3/exp2/bs4/slices_bs4/1010123456035_metrics.csv",
    ]
    cache.write_text("\n".join(expected) + "\n", encoding="utf-8")
    assert discover_tree(tmp_path) == expected


def test_filter_paths_keeps_all_existing_ue_files_in_requested_cells():
    cfg = _config()
    tree = [
        "slice_traffic/rome_static_close/tr0/exp1/bs1/slices_bs1/1010123456002_metrics.csv",
        "slice_traffic/rome_static_close/tr0/exp1/bs1/slices_bs1/1010123456003_metrics.csv",
        "slice_traffic/rome_static_close/tr0/exp1/bs4/slices_bs4/1010123456035_metrics.csv",
        "slice_traffic/rome_static_close/tr0/exp2/bs1/slices_bs1/1010123456002_metrics.csv",
        "slice_traffic/rome_static_close/tr0/exp2/bs4/slices_bs4/1010123456035_metrics.csv",
        "slice_traffic/rome_static_close/tr1/exp1/bs1/slices_bs1/1010123456002_metrics.csv",
        "slice_traffic/rome_static_close/tr1/exp1/bs4/slices_bs4/1010123456035_metrics.csv",
        "slice_traffic/rome_static_close/tr1/exp2/bs1/slices_bs1/1010123456002_metrics.csv",
        "slice_traffic/rome_static_close/tr1/exp2/bs4/slices_bs4/1010123456035_metrics.csv",
        "slice_traffic/rome_static_close/tr2/exp1/bs1/slices_bs1/1010123456002_metrics.csv",
        "slice_traffic/rome_static_close/tr2/exp1/bs4/slices_bs4/1010123456035_metrics.csv",
        "slice_traffic/rome_static_close/tr2/exp2/bs1/slices_bs1/1010123456002_metrics.csv",
        "slice_traffic/rome_static_close/tr2/exp2/bs4/slices_bs4/1010123456035_metrics.csv",
        "slice_traffic/rome_static_close/tr3/exp1/bs1/slices_bs1/1010123456002_metrics.csv",
        "slice_traffic/rome_static_close/tr3/exp1/bs4/slices_bs4/1010123456035_metrics.csv",
        "slice_traffic/rome_static_close/tr3/exp2/bs1/slices_bs1/1010123456002_metrics.csv",
        "slice_traffic/rome_static_close/tr3/exp2/bs4/slices_bs4/1010123456035_metrics.csv",
        "slice_traffic/rome_static_far/tr0/exp1/bs1/slices_bs1/1010123456002_metrics.csv",
        "slice_traffic/rome_static_far/tr0/exp1/bs4/slices_bs4/1010123456035_metrics.csv",
        "slice_traffic/rome_static_far/tr0/exp2/bs1/slices_bs1/1010123456002_metrics.csv",
        "slice_traffic/rome_static_far/tr0/exp2/bs4/slices_bs4/1010123456035_metrics.csv",
        "slice_traffic/rome_static_far/tr1/exp1/bs1/slices_bs1/1010123456002_metrics.csv",
        "slice_traffic/rome_static_far/tr1/exp1/bs4/slices_bs4/1010123456035_metrics.csv",
        "slice_traffic/rome_static_far/tr1/exp2/bs1/slices_bs1/1010123456002_metrics.csv",
        "slice_traffic/rome_static_far/tr1/exp2/bs4/slices_bs4/1010123456035_metrics.csv",
        "slice_traffic/rome_static_far/tr2/exp1/bs1/slices_bs1/1010123456002_metrics.csv",
        "slice_traffic/rome_static_far/tr2/exp1/bs4/slices_bs4/1010123456035_metrics.csv",
        "slice_traffic/rome_static_far/tr2/exp2/bs1/slices_bs1/1010123456002_metrics.csv",
        "slice_traffic/rome_static_far/tr2/exp2/bs4/slices_bs4/1010123456035_metrics.csv",
        "slice_traffic/rome_static_far/tr3/exp1/bs1/slices_bs1/1010123456002_metrics.csv",
        "slice_traffic/rome_static_far/tr3/exp1/bs4/slices_bs4/1010123456035_metrics.csv",
        "slice_traffic/rome_static_far/tr3/exp2/bs1/slices_bs1/1010123456002_metrics.csv",
        "slice_traffic/rome_static_far/tr3/exp2/bs4/slices_bs4/1010123456035_metrics.csv",
    ]
    selected = filter_paths(tree, cfg)
    assert len(selected) == len(tree)
    assert selected[0].startswith("slice_traffic/")


def test_config_accepts_disjoint_seen_and_unseen_splits():
    _config().validate()


def test_paired_statistics_are_episode_paired():
    proposed = pd.DataFrame(
        {"episode_id": ["a", "a", "b", "b"], "utility": [0.8, 0.9, 0.7, 0.8]}
    )
    baseline = pd.DataFrame(
        {"episode_id": ["a", "a", "b", "b"], "utility": [0.7, 0.8, 0.6, 0.7]}
    )
    stats = paired(proposed, baseline, _config(), seed=1)
    assert stats["episodes"] == 2
    assert np.isclose(stats["mean_paired_delta_utility"], 0.1)
    assert stats["bootstrap_ci95_low"] > 0


def _shortcut_frame(rows: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    columns = [
        "num_ues",
        "slice_prb",
        "power_multiplier",
        "scheduler_code",
        "dl_mcs",
        "dl_buffer_bytes",
        "dl_bitrate_mbps",
        "dl_errors_pct",
        "dl_cqi",
        "ul_mcs",
        "ul_buffer_bytes",
        "ul_bitrate_mbps",
        "ul_errors_pct",
        "ul_sinr",
        "requested_prbs",
        "granted_prbs",
        "grant_ratio",
    ]
    frame = pd.DataFrame({column: rng.normal(size=rows) for column in columns})
    frame["slice_type"] = np.resize(np.asarray(["eMBB", "mMTC", "URLLC"]), rows)
    frame["scheduler_code"] = np.resize(np.asarray([0, 1, 2]), rows)
    frame["slice_prb"] = np.resize(np.asarray([2, 4, 8]), rows)
    current = np.resize(
        np.asarray(["round_robin:prb=2", "waterfilling:prb=4", "proportional_fair:prb=8"]),
        rows,
    )
    frame["action"] = current
    frame.loc[::7, "action"] = "round_robin:prb=2"
    return frame


def test_shortcut_reports_policy_change_only_performance():
    frame = _shortcut_frame()
    result = shortcut(frame.iloc[:60], frame.iloc[60:], seed=3)
    assert set(result["variant"]) == {
        "full_state",
        "without_scheduler",
        "without_prb",
        "without_scheduler_and_prb",
    }
    assert (result["policy_change_rows"] > 0).all()
    assert result["agreement_policy_change"].notna().all()


def test_original_ppo_is_literature_reference_only():
    assert PAPER_REFERENCE["type"] == "literature_reference_only"
    assert PAPER_REFERENCE["reported_results"]["embb_spectral_efficiency_gain"]["value_percent"] == 20
    assert PAPER_REFERENCE["reported_results"]["urllc_average_buffer_reduction_percent"] == {
        "vs_round_robin": 37,
        "vs_waterfilling": 5,
        "vs_proportional_fair": 17,
    }
    assert "must not be inserted" in PAPER_REFERENCE["comparison_rule"]
