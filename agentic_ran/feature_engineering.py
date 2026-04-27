from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

TRAFFIC_CLASS_MAP = {
    "eMBB": {2, 5, 8, 12, 15, 18, 22, 25, 28, 32, 35, 38},
    "MTC": {3, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39},
    "URLLC": {1, 4, 7, 10, 11, 14, 17, 20, 21, 24, 27, 30, 31, 34, 37, 40},
}
TRAFFIC_CLASS_ID = {"eMBB": 0, "MTC": 1, "URLLC": 2, "unknown": -1}
SLICE_EXPECTED_CLASS = {0: "eMBB", 1: "MTC", 2: "URLLC"}
POLICY_NAME_MAP = {0: "RR", 1: "WF", 2: "PF"}
EPS = 1e-6


@dataclass(frozen=True)
class FeatureFlags:
    use_time_features: bool = True
    use_traffic_features: bool = True
    use_agentic_policy_features: bool = True
    allow_synthetic_time_index: bool = False


def extract_ue_id(imsi_value: object) -> int:
    if pd.isna(imsi_value):
        return -1
    text = str(imsi_value)
    matches = re.findall(r"(\d+)", text)
    if not matches:
        return -1
    ue = int(matches[-1])
    if 1 <= ue <= 40:
        return ue
    for m in matches:
        ue = int(m)
        if 1 <= ue <= 40:
            return ue
    return -1


def map_traffic_class(ue_id: int) -> str:
    for name, members in TRAFFIC_CLASS_MAP.items():
        if ue_id in members:
            return name
    return "unknown"


def map_policy_name(policy_value: object) -> str:
    try:
        return POLICY_NAME_MAP.get(int(policy_value), "unknown")
    except Exception:
        return "unknown"


def get_expected_rbg_allocation(experiment_second: float) -> dict[str, int | str | float]:
    phase_bounds = [
        (0, 30, (2, 2, 2)),
        (30, 60, (1, 2, 4)),
        (60, 90, (1, 4, 2)),
        (90, 120, (2, 1, 4)),
        (120, 150, (2, 4, 1)),
        (150, 180, (4, 2, 1)),
        (180, 210, (4, 1, 2)),
        (210, 240, (2, 2, 3)),
        (240, 270, (2, 3, 2)),
        (270, 300, (3, 2, 2)),
        (300, 330, (3, 3, 1)),
        (330, 360, (3, 1, 3)),
        (360, 390, (1, 3, 3)),
        (390, 420, (1, 2, 4)),
        (420, 450, (1, 4, 2)),
        (450, 480, (4, 2, 1)),
    ]
    sec = float(experiment_second) % 480.0
    for start, end, alloc in phase_bounds:
        if start <= sec < end:
            return {
                "expected_rbg_slice_0": alloc[0],
                "expected_rbg_slice_1": alloc[1],
                "expected_rbg_slice_2": alloc[2],
                "rbg_schedule_phase": f"{start}-{end}",
                "seconds_to_next_resize": end - sec,
                "seconds_since_last_resize": sec - start,
            }
    fallback = phase_bounds[-1]
    return {
        "expected_rbg_slice_0": fallback[2][0],
        "expected_rbg_slice_1": fallback[2][1],
        "expected_rbg_slice_2": fallback[2][2],
        "rbg_schedule_phase": f"{fallback[0]}-{fallback[1]}",
        "seconds_to_next_resize": fallback[1] - sec,
        "seconds_since_last_resize": sec - fallback[0],
    }


def add_temporal_features(df: pd.DataFrame, allow_synthetic_time_index: bool = False) -> pd.DataFrame:
    out = df.copy()
    if "Timestamp" not in out.columns:
        if not allow_synthetic_time_index:
            raise ValueError("Timestamp column is required for temporal features. Set allow_synthetic_time_index=True to bypass.")
        out["time_index"] = np.arange(len(out), dtype=np.int64)
        out["elapsed_seconds"] = out["time_index"].astype(float)
        out["experiment_second"] = out["elapsed_seconds"] % 480.0
        return out

    out["Timestamp"] = pd.to_datetime(out["Timestamp"], errors="coerce")
    bad = out["Timestamp"].isna().sum()
    if bad > 0:
        LOGGER.warning("Dropping %s rows with invalid Timestamp values.", int(bad))
        out = out.dropna(subset=["Timestamp"]).copy()

    out = out.sort_values("Timestamp", kind="stable").reset_index(drop=True)
    out["time_index"] = np.arange(len(out), dtype=np.int64)
    ts = out["Timestamp"]
    out["hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["day_of_month"] = ts.dt.day
    out["month"] = ts.dt.month
    out["minute_of_day"] = ts.dt.hour * 60 + ts.dt.minute
    elapsed = (ts - ts.iloc[0]).dt.total_seconds()
    out["elapsed_seconds"] = elapsed
    out["experiment_second"] = np.mod(elapsed, 480.0)
    out["sin_hour"] = np.sin(2 * math.pi * out["hour"] / 24.0)
    out["cos_hour"] = np.cos(2 * math.pi * out["hour"] / 24.0)
    out["sin_day_of_week"] = np.sin(2 * math.pi * out["day_of_week"] / 7.0)
    out["cos_day_of_week"] = np.cos(2 * math.pi * out["day_of_week"] / 7.0)
    return out


def enrich_features(df: pd.DataFrame, flags: FeatureFlags | None = None) -> pd.DataFrame:
    flags = flags or FeatureFlags()
    out = df.copy()
    def _numeric_col(name: str) -> pd.Series:
        if name in out.columns:
            return pd.to_numeric(out[name], errors="coerce").fillna(0.0)
        return pd.Series(np.zeros(len(out), dtype=float), index=out.index)

    if flags.use_time_features:
        out = add_temporal_features(out, allow_synthetic_time_index=flags.allow_synthetic_time_index)

    out["sum_requested_prbs"] = _numeric_col("sum_requested_prbs")
    out["sum_granted_prbs"] = _numeric_col("sum_granted_prbs")
    out["slice_prb"] = _numeric_col("slice_prb")
    out["dl_n_samples"] = _numeric_col("dl_n_samples")
    out["ul_n_samples"] = _numeric_col("ul_n_samples")
    out["dl_buffer [bytes]"] = _numeric_col("dl_buffer [bytes]")
    out["ul_buffer [bytes]"] = _numeric_col("ul_buffer [bytes]")
    out["dl_cqi"] = _numeric_col("dl_cqi")
    out["dl_mcs"] = _numeric_col("dl_mcs")
    out["ul_sinr"] = _numeric_col("ul_sinr")
    out["ul_rssi"] = _numeric_col("ul_rssi")

    out["ratio_granted_req"] = out["sum_granted_prbs"] / np.maximum(out["sum_requested_prbs"], EPS)
    out["prb_pressure"] = out["sum_requested_prbs"] / np.maximum(out["sum_granted_prbs"], EPS)
    out["normalized_slice_prb"] = out["slice_prb"] / 15.0
    out["buffer_pressure_dl"] = out["dl_buffer [bytes]"] / np.maximum(out["dl_n_samples"], EPS)
    out["buffer_pressure_ul"] = out["ul_buffer [bytes]"] / np.maximum(out["ul_n_samples"], EPS)
    out["spectral_quality_proxy"] = out["dl_cqi"] * out["dl_mcs"]
    out["uplink_quality_proxy"] = out["ul_sinr"] - np.abs(out["ul_rssi"])

    if flags.use_traffic_features:
        imsi_series = out["IMSI"] if "IMSI" in out.columns else pd.Series([-1] * len(out), index=out.index)
        out["ue_id"] = imsi_series.apply(extract_ue_id)
        out["traffic_class"] = out["ue_id"].apply(map_traffic_class)
        out["traffic_class_id"] = out["traffic_class"].map(TRAFFIC_CLASS_ID).fillna(-1).astype(int)
        out["is_embb"] = (out["traffic_class"] == "eMBB").astype(int)
        out["is_mtc"] = (out["traffic_class"] == "MTC").astype(int)
        out["is_urllc"] = (out["traffic_class"] == "URLLC").astype(int)

        out["slice_id"] = pd.to_numeric(out.get("slice_id"), errors="coerce").fillna(-1).astype(int)
        out["slice_expected_traffic_class"] = out["slice_id"].map(SLICE_EXPECTED_CLASS).fillna("unknown")
        out["slice_expected_traffic_id"] = out["slice_expected_traffic_class"].map(TRAFFIC_CLASS_ID).fillna(-1).astype(int)
        out["traffic_matches_slice"] = (out["traffic_class"] == out["slice_expected_traffic_class"]).astype(int)
        out["slice_is_embb"] = (out["slice_expected_traffic_class"] == "eMBB").astype(int)
        out["slice_is_mtc"] = (out["slice_expected_traffic_class"] == "MTC").astype(int)
        out["slice_is_urllc"] = (out["slice_expected_traffic_class"] == "URLLC").astype(int)

    out["scheduling_policy"] = _numeric_col("scheduling_policy").astype(int)
    out["scheduling_policy_name"] = out["scheduling_policy"].apply(map_policy_name)
    out["policy_rr"] = (out["scheduling_policy"] == 0).astype(int)
    out["policy_wf"] = (out["scheduling_policy"] == 1).astype(int)
    out["policy_pf"] = (out["scheduling_policy"] == 2).astype(int)

    if flags.use_agentic_policy_features and "experiment_second" in out.columns:
        schedule = out["experiment_second"].apply(get_expected_rbg_allocation).apply(pd.Series)
        out = pd.concat([out, schedule], axis=1)
        out["expected_rbg_for_slice"] = out.apply(
            lambda r: r.get(f"expected_rbg_slice_{int(r['slice_id'])}", np.nan) if r.get("slice_id", -1) in (0, 1, 2) else np.nan,
            axis=1,
        )

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out
