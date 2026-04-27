from __future__ import annotations

import numpy as np
import pandas as pd

from agentic_ran.agentic_policy import ACTION_SPACE, recommend_action
from agentic_ran.feature_engineering import FeatureFlags, enrich_features, get_expected_rbg_allocation
from scripts.prepare_splits import _iter_metrics_files, split_and_save


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Timestamp": pd.date_range("2024-01-01", periods=5, freq="30s"),
            "IMSI": ["imsi-1", "imsi-2", "imsi-3", "imsi-4", "imsi-5"],
            "slice_id": [0, 1, 2, 0, 1],
            "slice_prb": [5, 5, 5, 5, 5],
            "scheduling_policy": [0, 1, 2, 0, 1],
            "sum_requested_prbs": [4, 5, 8, 2, 3],
            "sum_granted_prbs": [3, 5, 6, 2, 3],
            "dl_n_samples": [10, 10, 10, 10, 10],
            "ul_n_samples": [10, 10, 10, 10, 10],
            "dl_buffer [bytes]": [1000, 1200, 1300, 900, 800],
            "ul_buffer [bytes]": [500, 400, 600, 450, 300],
            "dl_cqi": [10, 9, 8, 7, 6],
            "dl_mcs": [12, 11, 10, 9, 8],
            "ul_sinr": [5, 4, 3, 2, 1],
            "ul_rssi": [-90, -91, -92, -93, -94],
            "tx_brate downlink [Mbps]": [1.1, 1.0, 0.8, 0.9, 1.2],
        }
    )


def test_load_only_metrics_glob(tmp_path):
    d = tmp_path / "dataset"
    d.mkdir()
    (d / "a_metrics.csv").write_text("x", encoding="utf-8")
    (d / "b.csv").write_text("x", encoding="utf-8")
    files = _iter_metrics_files([d])
    assert len(files) == 1
    assert files[0].name.endswith("_metrics.csv")


def test_temporal_and_traffic_features_no_nan_inf():
    out = enrich_features(_sample_df(), flags=FeatureFlags())
    for col in ["hour", "day_of_week", "sin_hour", "cos_hour", "experiment_second", "traffic_class_id", "policy_rr", "expected_rbg_slice_0", "ratio_granted_req"]:
        assert col in out.columns
    num = out.select_dtypes(include=[np.number]).to_numpy()
    assert np.isfinite(num).all()


def test_schedule_mapping():
    sched = get_expected_rbg_allocation(35)
    assert sched["expected_rbg_slice_0"] == 1
    assert sched["expected_rbg_slice_2"] == 4


def test_agentic_policy_schema_and_id_validity():
    row = enrich_features(_sample_df().iloc[[2]], flags=FeatureFlags()).iloc[0]
    decision = recommend_action(row)
    assert decision["action_id"] in ACTION_SPACE
    for key in ["action_name", "target_slice", "recommended_policy", "recommended_prb_delta", "confidence", "reason"]:
        assert key in decision


def test_time_ordered_split(tmp_path):
    df = _sample_df()
    frames = [("f", df)]
    summary = split_and_save(frames, tmp_path)
    train = pd.read_csv(summary["files"]["train"])
    test = pd.read_csv(summary["files"]["test"])
    assert pd.to_datetime(train["Timestamp"]).max() <= pd.to_datetime(test["Timestamp"]).min()
