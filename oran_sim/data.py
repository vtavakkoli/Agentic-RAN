from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
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
            "traffic_load": np.clip(traffic_load, 0, 1.2),
        }
    )


# ---------------------------
# Real dataset loader (O-RAN KPM folder)
# ---------------------------

_ENB_COLS = ["time", "nof_ue", "dl_brate", "ul_brate"]

# Map UE CSV columns -> normalized feature names used in FEATURE_ORDER
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


def _safe_empty_df(cols: List[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=cols)


def _read_enb_metrics(path: Path, verbose: bool = False) -> pd.DataFrame:
    # Skip empty files quickly
    try:
        if path.stat().st_size == 0:
            if verbose:
                print(f"[WARN] Empty enb_metrics.csv: {path}")
            return _safe_empty_df(_ENB_COLS)
    except OSError:
        if verbose:
            print(f"[WARN] Cannot stat enb_metrics.csv: {path}")
        return _safe_empty_df(_ENB_COLS)

    try:
        df = pd.read_csv(path)  # comma-separated
    except EmptyDataError:
        if verbose:
            print(f"[WARN] EmptyDataError enb_metrics.csv: {path}")
        return _safe_empty_df(_ENB_COLS)
    except Exception as e:
        if verbose:
            print(f"[WARN] Failed reading enb_metrics.csv: {path} ({e})")
        return _safe_empty_df(_ENB_COLS)

    if df is None or df.empty:
        return _safe_empty_df(_ENB_COLS)

    keep = [c for c in _ENB_COLS if c in df.columns]
    if "time" not in keep or "dl_brate" not in keep:
        if verbose:
            print(f"[WARN] Missing required cols in {path}. Have={list(df.columns)}")
        return _safe_empty_df(_ENB_COLS)

    df = df[keep].copy()
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["dl_brate"] = pd.to_numeric(df["dl_brate"], errors="coerce")
    if "nof_ue" in df.columns:
        df["nof_ue"] = pd.to_numeric(df["nof_ue"], errors="coerce")
    if "ul_brate" in df.columns:
        df["ul_brate"] = pd.to_numeric(df["ul_brate"], errors="coerce")

    df = df.dropna(subset=["time"]).sort_values("time")
    return df


def _read_ue_metrics(path: Path, verbose: bool = False) -> pd.DataFrame:
    try:
        if path.stat().st_size == 0:
            if verbose:
                print(f"[WARN] Empty ue_metrics.csv: {path}")
            return _safe_empty_df(["time"])
    except OSError:
        if verbose:
            print(f"[WARN] Cannot stat ue_metrics.csv: {path}")
        return _safe_empty_df(["time"])

    try:
        df = pd.read_csv(path, sep=";")  # UE metrics are semicolon-separated
    except EmptyDataError:
        if verbose:
            print(f"[WARN] EmptyDataError ue_metrics.csv: {path}")
        return _safe_empty_df(["time"])
    except Exception as e:
        if verbose:
            print(f"[WARN] Failed reading ue_metrics.csv: {path} ({e})")
        return _safe_empty_df(["time"])

    if df is None or df.empty:
        return _safe_empty_df(["time"])

    if "time" not in df.columns:
        if verbose:
            print(f"[WARN] Missing 'time' in ue_metrics.csv: {path}. Have={list(df.columns)}")
        return _safe_empty_df(["time"])

    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    return df


def _aggregate_ues(ue_metric_files: List[Path], verbose: bool = False) -> pd.DataFrame:
    """
    Reads ue_metrics.csv from multiple UE folders and aggregates to one row per 'time'
    by averaging numeric KPIs across UEs.
    Returns columns named per FEATURE_ORDER (where possible) plus 'time'.
    """
    frames: List[pd.DataFrame] = []

    for f in ue_metric_files:
        u = _read_ue_metrics(f, verbose=verbose)
        if u.empty or "time" not in u.columns:
            continue

        available_src_cols = [c for c in _UE_TO_FEATURE.keys() if c in u.columns]
        if not available_src_cols:
            continue

        sub = u[["time"] + available_src_cols].copy()
        for c in available_src_cols:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        frames.append(sub)

    if not frames:
        return _safe_empty_df(["time"])

    all_ue = pd.concat(frames, axis=0, ignore_index=True)

    # Average over UEs for each time
    agg = all_ue.groupby("time", as_index=False).mean(numeric_only=True)

    # Rename to normalized feature columns
    rename = {src: dst for src, dst in _UE_TO_FEATURE.items() if src in agg.columns}
    agg = agg.rename(columns=rename)

    return agg


def load_timeseries_from_kpm(
    root_dir: str | Path,
    n_steps: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build consolidated timeseries from the O-RAN KPM dataset folder.

    Progress output includes:
      - processed experiments / total
      - rows accumulated
      - rows/sec and ETA (rough)
    """
    import time as _time

    root = Path(root_dir)
    exp_dirs = sorted(root.glob("cluster_*/slicing_*/scheduling_*/RESERVATION-*"))

    if not exp_dirs:
        raise FileNotFoundError(
            f"No RESERVATION-* folders found under: {root}. "
            f"Expected: cluster_*/slicing_*/scheduling_*/RESERVATION-*"
        )

    out_frames: List[pd.DataFrame] = []
    skipped = 0
    used = 0
    rows_so_far = 0

    t0 = _time.time()
    last_print = t0

    total = len(exp_dirs)

    for i, exp in enumerate(exp_dirs, start=1):
        enb_path = exp / "bs" / "enb_metrics.csv"
        if not enb_path.exists():
            skipped += 1
            continue

        enb = _read_enb_metrics(enb_path, verbose=verbose)
        if enb.empty:
            skipped += 1
            continue

        ue_files = sorted(exp.glob("ue_*/ue_metrics.csv"))
        ue_agg = _aggregate_ues(ue_files, verbose=verbose)

        # Join on time
        if not ue_agg.empty and "time" in ue_agg.columns:
            df = enb.merge(ue_agg, on="time", how="left")
        else:
            df = enb.copy()

        # Target + context
        df["traffic_load"] = pd.to_numeric(df["dl_brate"], errors="coerce")
        if "nof_ue" in df.columns:
            df["ue_count"] = pd.to_numeric(df["nof_ue"], errors="coerce")
        else:
            df["ue_count"] = float(len([p for p in exp.glob("ue_*") if p.is_dir()]))

        # Ensure columns exist
        for col in FEATURE_ORDER + ["traffic_load"]:
            if col not in df.columns:
                df[col] = 0.0

        # Force numeric
        for col in ["time"] + FEATURE_ORDER + ["traffic_load"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df = df[["time"] + FEATURE_ORDER + ["traffic_load"]].copy()

        out_frames.append(df)
        used += 1
        rows_so_far += len(df)

        # If user asked for n_steps, we can stop early once we have enough
        if n_steps is not None and n_steps > 0 and rows_so_far >= n_steps:
            if verbose:
                print(f"[INFO] Reached requested --step={n_steps} rows. Stopping early.")
            break

        # Progress print every ~1s (and at end)
        now = _time.time()
        if verbose and (now - last_print >= 1.0 or i == total):
            elapsed = max(now - t0, 1e-9)
            rps = rows_so_far / elapsed

            # Estimate total rows based on average rows/experiment so far
            avg_rows_per_exp = rows_so_far / max(i, 1)
            est_total_rows = avg_rows_per_exp * total
            remaining_rows = max(est_total_rows - rows_so_far, 0.0)
            eta_sec = remaining_rows / max(rps, 1e-9)

            def _fmt_eta(sec: float) -> str:
                sec = int(max(sec, 0))
                h = sec // 3600
                m = (sec % 3600) // 60
                s = sec % 60
                return f"{h:02d}:{m:02d}:{s:02d}"

            pct = (i / total) * 100.0
            print(
                f"[PROGRESS] {i}/{total} experiments ({pct:5.1f}%) | "
                f"rows={rows_so_far:,} | {rps:,.1f} rows/s | ETA~{_fmt_eta(eta_sec)}"
            )
            last_print = now

    if not out_frames:
        raise RuntimeError(
            f"Found experiments under {root}, but none produced usable tables. "
            f"Skipped={skipped}, Total={len(exp_dirs)}"
        )

    full = pd.concat(out_frames, axis=0, ignore_index=True)
    full = full.sort_values("time").reset_index(drop=True)

    # Truncate to exact n_steps if requested
    if n_steps is not None and n_steps > 0 and len(full) > n_steps:
        full = full.iloc[:n_steps].copy()

    if verbose:
        elapsed = max(_time.time() - t0, 1e-9)
        print(f"[INFO] Loaded experiments: {used}/{len(exp_dirs)} (skipped={skipped})")
        print(f"[INFO] Final rows: {len(full):,} in {elapsed:.1f}s ({len(full)/elapsed:,.1f} rows/s)")

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

    # Ensure required columns exist
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
