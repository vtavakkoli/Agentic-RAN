from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from oran_sim.config import FEATURE_ORDER


# ---------------------------
# Synthetic generator (unchanged)
# ---------------------------

def generate_timeseries(n_steps: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)

    hour = t % 24
    day_of_week = (t // 24) % 7
    is_weekend = (day_of_week >= 5).astype(float)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    daily_cycle = 0.6 + 0.3 * (np.sin(2 * np.pi * (hour - 6) / 24) + 1) / 2
    weekly_cycle = 0.85 + 0.2 * (1 - is_weekend) + 0.05 * is_weekend

    mobility_index = np.clip(0.3 + 0.5 * daily_cycle + rng.normal(0, 0.08, n_steps), 0, 1)
    video_demand = np.clip(0.45 + 0.4 * daily_cycle + 0.08 * is_weekend + rng.normal(0, 0.07, n_steps), 0, 1.4)
    gaming_demand = np.clip(0.3 + 0.35 * daily_cycle + 0.15 * (hour >= 18) + rng.normal(0, 0.08, n_steps), 0, 1.4)
    iot_demand = np.clip(0.35 + 0.05 * hour_cos + rng.normal(0, 0.03, n_steps), 0, 1)

    demand_signal = 0.35 * video_demand + 0.25 * gaming_demand + 0.15 * iot_demand + 0.25 * mobility_index

    ue_count = np.clip(40 + 180 * demand_signal * weekly_cycle + rng.normal(0, 8, n_steps), 10, None)
    prb_utilization = np.clip(0.2 + 0.7 * demand_signal + rng.normal(0, 0.05, n_steps), 0.05, 1)
    sinr = np.clip(22 - 9 * prb_utilization - 2 * mobility_index + rng.normal(0, 1.2, n_steps), -5, 30)
    rsrp = np.clip(-96 + 8 * (1 - mobility_index) - 3 * prb_utilization + rng.normal(0, 2.0, n_steps), -125, -70)
    rsrq = np.clip(-13 + 2.5 * (1 - prb_utilization) + rng.normal(0, 1.0, n_steps), -20, -3)
    handover_rate = np.clip(0.05 + 0.7 * mobility_index + rng.normal(0, 0.04, n_steps), 0.01, 1.3)
    packet_loss = np.clip(0.002 + 0.06 * prb_utilization + 0.02 * np.maximum(0, 0.5 - sinr / 20) + rng.normal(0, 0.002, n_steps), 0, 0.2)
    latency_ms = np.clip(8 + 45 * prb_utilization + 15 * packet_loss + rng.normal(0, 2.5, n_steps), 3, 120)

    traffic_load = np.zeros(n_steps)
    for i in range(n_steps):
        base = (
            0.30 * prb_utilization[i]
            + 0.15 * (ue_count[i] / 220)
            + 0.15 * video_demand[i]
            + 0.12 * gaming_demand[i]
            + 0.08 * iot_demand[i]
            + 0.08 * (1 - np.clip((sinr[i] + 5) / 35, 0, 1))
            + 0.12 * daily_cycle[i]
        )
        traffic_load[i] = (
            base + rng.normal(0, 0.03)
            if i == 0
            else 0.68 * traffic_load[i - 1] + 0.32 * base + rng.normal(0, 0.02)
        )

    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n_steps, freq="h"),
            "rsrp": rsrp,
            "rsrq": rsrq,
            "sinr": sinr,
            "prb_utilization": prb_utilization,
            "ue_count": ue_count,
            "handover_rate": handover_rate,
            "packet_loss": packet_loss,
            "latency_ms": latency_ms,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "mobility_index": mobility_index,
            "video_demand": video_demand,
            "gaming_demand": gaming_demand,
            "iot_demand": iot_demand,
            "traffic_load": np.clip(traffic_load, 0, 1.2),
        }
    )


# ---------------------------
# Real dataset loader (O-RAN KPM folder)
# ---------------------------

_ENB_COLS = ["time", "nof_ue", "dl_brate", "ul_brate"]

# Map UE CSV columns -> normalized feature names you want in FEATURE_ORDER
_UE_TO_FEATURE = {
    "rsrp": "rsrp",
    "dl_snr": "dl_snr",
    "pl": "pl",
    "cfo": "cfo",
    "ul_ta": "ul_ta",
    "dl_mcs": "dl_mcs",
    "ul_mcs": "ul_mcs",
    "dl_bler": "dl_bler",
    "ul_bler": "ul_bler",
    "dl_brate": "dl_brate_ue",
    "ul_brate": "ul_brate_ue",
    "ul_buff": "ul_buff",
    "rf_o": "rf_o",
    "rf_u": "rf_u",
    "rf_l": "rf_l",
}


def _read_enb_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)  # comma-separated
    # keep only needed columns if present
    keep = [c for c in _ENB_COLS if c in df.columns]
    df = df[keep].copy()
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    return df


def _read_ue_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")  # UE metrics are semicolon-separated
    if "time" not in df.columns:
        raise ValueError(f"Missing 'time' column in {path}")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    return df


def _aggregate_ues(ue_metric_files: List[Path]) -> pd.DataFrame:
    frames = []
    for f in ue_metric_files:
        try:
            u = _read_ue_metrics(f)
        except Exception:
            continue

        # select the columns we can map
        available = [c for c in _UE_TO_FEATURE.keys() if c in u.columns]
        if not available:
            continue

        sub = u[["time"] + available].copy()
        # numeric convert
        for c in available:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        frames.append(sub)

    if not frames:
        # return empty frame with time
        return pd.DataFrame(columns=["time"])

    all_ue = pd.concat(frames, axis=0, ignore_index=True)

    # mean over UEs at each time
    agg = all_ue.groupby("time", as_index=False).mean(numeric_only=True)

    # rename to normalized feature columns
    rename = {src: dst for src, dst in _UE_TO_FEATURE.items() if src in agg.columns}
    agg = agg.rename(columns=rename)
    return agg


def load_timeseries_from_kpm(root_dir: str | Path, n_steps: Optional[int] = None) -> pd.DataFrame:
    """
    Build one consolidated timeseries table from the O-RAN KPM dataset folder.

    Output columns:
    - traffic_load (from enb dl_brate)
    - ue_count (from enb nof_ue)
    - plus UE aggregated features that exist (see _UE_TO_FEATURE)
    """
    root = Path(root_dir)
    exp_dirs = sorted(root.glob("cluster_*/slicing_*/scheduling_*/RESERVATION-*"))

    if not exp_dirs:
        raise FileNotFoundError(
            f"No RESERVATION-* folders found under: {root}. "
            f"Expected: cluster_*/slicing_*/scheduling_*/RESERVATION-*"
        )

    out_frames: List[pd.DataFrame] = []

    for exp in exp_dirs:
        enb_path = exp / "bs" / "enb_metrics.csv"
        if not enb_path.exists():
            continue

        enb = _read_enb_metrics(enb_path)
        if enb.empty:
            continue

        # UE files
        ue_files = sorted(exp.glob("ue_*/ue_metrics.csv"))
        ue_agg = _aggregate_ues(ue_files)

        # Join on time (nearest exact match; if mismatch you can resample outside)
        df = enb.merge(ue_agg, on="time", how="left")

        # Required target + context
        if "dl_brate" not in df.columns:
            continue

        df["traffic_load"] = pd.to_numeric(df["dl_brate"], errors="coerce")
        df["ue_count"] = pd.to_numeric(df.get("nof_ue", np.nan), errors="coerce")

        # drop raw enb columns except what you want
        # (keep time if you want debugging; training drops it anyway)
        # Here we keep time so you can inspect; you can drop in generate_data.py if desired.
        # df = df.drop(columns=[c for c in ["dl_brate", "ul_brate", "nof_ue"] if c in df.columns])

        # Add basic sanity fills
        df = df.replace([np.inf, -np.inf], np.nan)

        out_frames.append(df)

    if not out_frames:
        raise RuntimeError(f"Found experiments under {root}, but none produced usable tables (missing enb_metrics?)")

    full = pd.concat(out_frames, axis=0, ignore_index=True)

    # Sort by time within the concatenated stream
    if "time" in full.columns:
        full = full.sort_values("time").reset_index(drop=True)

    # Truncate to n_steps if requested
    if n_steps is not None and n_steps > 0 and len(full) > n_steps:
        full = full.iloc[:n_steps].copy()

    # Ensure all required columns exist for training
    for col in FEATURE_ORDER + ["traffic_load"]:
        if col not in full.columns:
            full[col] = 0.0

    # Prefer numeric types
    for col in FEATURE_ORDER + ["traffic_load"]:
        full[col] = pd.to_numeric(full[col], errors="coerce")

    full = full.fillna(0.0)

    # Optional: keep only features + target (+ time if you want)
    keep = ["time"] + FEATURE_ORDER + ["traffic_load"] if "time" in full.columns else FEATURE_ORDER + ["traffic_load"]
    full = full[keep]

    return full


# ---------------------------
# Training prep (robust)
# ---------------------------

def build_sequences(features: np.ndarray, target: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i in range(seq_len, len(features)):
        x.append(features[i - seq_len : i])
        y.append(target[i])
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


def prepare_dataset(
    csv_path: str,
    feature_count: int,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, StandardScaler, list[str]]:
    df = pd.read_csv(csv_path)

    # Make sure required columns exist even if some experiments lack KPIs
    for col in FEATURE_ORDER + ["traffic_load"]:
        if col not in df.columns:
            df[col] = 0.0

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    selected_features = FEATURE_ORDER[:feature_count]
    x_raw = df[selected_features].values
    y_raw = df["traffic_load"].values.astype(np.float32)

    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x_scaled = x_scaler.fit_transform(x_raw)
    y_scaled = y_scaler.fit_transform(y_raw.reshape(-1, 1)).reshape(-1)

    x_seq, y_seq = build_sequences(x_scaled, y_scaled, seq_len=seq_len)
    return x_seq, y_seq, x_scaler, y_scaler, selected_features


def save_dataframe(df: pd.DataFrame, output: str) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out
