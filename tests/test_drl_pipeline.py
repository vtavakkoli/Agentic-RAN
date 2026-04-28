from __future__ import annotations

import pandas as pd

from agentic_ran.drl_data import SLICE_OBSERVATION_COLS, split_data
from agentic_ran.drl_env import RANControlEnv


def _dataset() -> pd.DataFrame:
    n = 24
    rows = []
    for i in range(n):
        sid = i % 3
        rows.append(
            {
                "Timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=i),
                "source_file": "f_metrics.csv",
                "sample_id": i,
                "global_index": i,
                "slice_id": sid,
                "slice_prb": 5 + sid,
                "scheduling_policy": sid,
                "ratio_granted_req": 0.8,
                "tx_brate downlink [Mbps]": 1.0 + sid,
                "rx_brate uplink [Mbps]": 0.6 + sid,
                "tx_errors downlink (%)": 2.0,
                "rx_errors uplink (%)": 1.0,
                "dl_cqi": 10,
                "ul_sinr": 5,
                "ul_rssi": -90,
                "sum_requested_prbs": 4,
                "sum_granted_prbs": 3,
                "dl_buffer [bytes]": 1.0,
                "ul_buffer [bytes]": 0.2,
                "traffic_class_id": sid,
            }
        )
    return pd.DataFrame(rows)


def test_split_data_slice_windows_shape():
    df = _dataset()
    windows = split_data(df, window_size=8)
    assert set(windows.keys()) == {0, 1, 2}
    assert windows[0].shape == (8, len(SLICE_OBSERVATION_COLS))


def test_split_data_missing_optional_cols_is_robust():
    df = _dataset().drop(columns=["traffic_class_id"])
    windows = split_data(df, window_size=4)
    assert windows[1].shape == (4, len(SLICE_OBSERVATION_COLS))


def test_env_reset_step_specs():
    env = RANControlEnv(_dataset(), window_size=6)
    obs = env.reset()
    assert obs.shape == env.observation_spec.shape
    next_obs, reward, done, info = env.step(0)
    assert next_obs.shape == env.observation_spec.shape
    assert isinstance(reward, float)
    assert "control_message" in info
    assert done is False
