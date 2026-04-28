from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

METRICS_GLOB = "**/*_metrics.csv"

METRIC_COLUMNS = [
    "Timestamp",
    "num_ues",
    "IMSI",
    "RNTI",
    "slicing_enabled",
    "slice_id",
    "slice_prb",
    "power_multiplier",
    "scheduling_policy",
    "dl_mcs",
    "dl_n_samples",
    "dl_buffer [bytes]",
    "tx_brate downlink [Mbps]",
    "tx_pkts downlink",
    "tx_errors downlink (%)",
    "dl_cqi",
    "ul_mcs",
    "ul_n_samples",
    "ul_buffer [bytes]",
    "rx_brate uplink [Mbps]",
    "rx_pkts uplink",
    "rx_errors uplink (%)",
    "ul_rssi",
    "ul_sinr",
    "phr",
    "sum_requested_prbs",
    "sum_granted_prbs",
    "dl_pmi",
    "dl_ri",
    "ul_n",
    "ul_turbo_iters",
]

SLICE_OBSERVATION_COLS = [
    "dl_buffer [bytes]",
    "tx_brate downlink [Mbps]",
    "rx_brate uplink [Mbps]",
    "tx_errors downlink (%)",
    "rx_errors uplink (%)",
    "dl_cqi",
    "ul_sinr",
    "ul_rssi",
    "sum_requested_prbs",
    "sum_granted_prbs",
    "ratio_granted_req",
    "slice_prb",
    "scheduling_policy",
    "traffic_class_id",
]


def _read_single(path: Path, keep_zero_requested_prbs: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = set(METRIC_COLUMNS)
    missing = sorted(expected.difference(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    work = df.loc[:, METRIC_COLUMNS].copy()
    work["Timestamp"] = pd.to_datetime(work["Timestamp"], errors="coerce")
    work = work.dropna(subset=["Timestamp"]).copy()

    for col in [c for c in METRIC_COLUMNS if c != "Timestamp" and c != "IMSI" and c != "RNTI"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    if not keep_zero_requested_prbs:
        work = work.loc[work["sum_requested_prbs"] > 0].copy()

    work["ratio_granted_req"] = np.clip(
        np.nan_to_num(work["sum_granted_prbs"] / work["sum_requested_prbs"], nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        1.0,
    )
    work["dl_buffer [bytes]"] = work["dl_buffer [bytes]"] / 100000.0
    return work.reset_index(drop=True)


def entire_dataset_from_single_file(path: Path, keep_zero_requested_prbs: bool = False) -> pd.DataFrame:
    df = _read_single(path, keep_zero_requested_prbs=keep_zero_requested_prbs)
    df["source_file"] = str(path)
    df["sample_id"] = np.arange(len(df), dtype=np.int64)
    df["global_index"] = df["sample_id"]
    return df


def entire_dataset_from_folder(folder: Path, keep_zero_requested_prbs: bool = False) -> pd.DataFrame:
    files = sorted(folder.glob(METRICS_GLOB))
    if not files:
        raise FileNotFoundError(f"No *_metrics.csv found under {folder}")

    frames: list[pd.DataFrame] = []
    offset = 0
    for path in files:
        frame = _read_single(path, keep_zero_requested_prbs=keep_zero_requested_prbs)
        if frame.empty:
            continue
        frame["source_file"] = str(path)
        frame["sample_id"] = np.arange(len(frame), dtype=np.int64)
        frame["global_index"] = np.arange(offset, offset + len(frame), dtype=np.int64)
        offset += len(frame)
        frames.append(frame)

    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.sort_values(["Timestamp", "source_file"], kind="stable").reset_index(drop=True)
    dataset["global_index"] = np.arange(len(dataset), dtype=np.int64)
    return dataset


def split_data(df: pd.DataFrame, window_size: int, pad_mode: str = "repeat_last") -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for sid in [0, 1, 2]:
        part = df.loc[df["slice_id"] == sid, SLICE_OBSERVATION_COLS].copy()
        for col in SLICE_OBSERVATION_COLS:
            part[col] = pd.to_numeric(part[col], errors="coerce").fillna(0.0)
        arr = part.to_numpy(dtype=np.float32)
        if len(arr) >= window_size:
            out[sid] = arr[-window_size:]
            continue

        pad_len = max(0, window_size - len(arr))
        if pad_len == 0:
            out[sid] = arr
            continue

        if len(arr) == 0:
            pad = np.zeros((window_size, len(SLICE_OBSERVATION_COLS)), dtype=np.float32)
            out[sid] = pad
            continue

        if pad_mode == "repeat_last":
            last = np.repeat(arr[-1:, :], repeats=pad_len, axis=0)
            out[sid] = np.concatenate([arr, last], axis=0)
        else:
            pad = np.zeros((pad_len, arr.shape[1]), dtype=np.float32)
            out[sid] = np.concatenate([pad, arr], axis=0)

    return out
