# oran_sim/data.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from sklearn.preprocessing import StandardScaler

from oran_sim.config import FEATURE_ORDER


# ---------------------------
# Synthetic generator (optional fallback)
# ---------------------------

def generate_timeseries(n_steps: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    traffic_load = np.clip(rng.normal(0.6, 0.1, n_steps), 0, 1.2)
    return pd.DataFrame(
        {"timestamp": pd.date_range("2024-01-01", periods=n_steps, freq="h"), "traffic_load": traffic_load}
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


def _read_csv_sniff(path: Path, *, prefer_sep: Optional[str] = None) -> pd.DataFrame:
    """
    Robust CSV reader:
    - skips empty files
    - tries preferred separator first (if provided)
    - then uses sep=None (python engine) to sniff
    """
    if path.stat().st_size == 0:
        raise EmptyDataError("empty file")

    if prefer_sep is not None:
        try:
            return pd.read_csv(path, sep=prefer_sep)
        except Exception:
            pass

    # Sniff separator (handles BOM, weird formatting)
    return pd.read_csv(path, sep=None, engine="python")


def _read_enb_metrics(path: Path, verbose: bool = False) -> pd.DataFrame:
    try:
        df = _read_csv_sniff(path, prefer_sep=",")
    except EmptyDataError:
        if verbose:
            print(f"[WARN] Empty enb_metrics.csv: {path}")
        return _safe_empty_df(_ENB_COLS)
    except Exception as e:
        if verbose:
            print(f"[WARN] Failed reading enb_metrics.csv: {path} ({e})")
        return _safe_empty_df(_ENB_COLS)

    if df.empty:
        return _safe_empty_df(_ENB_COLS)

    df.columns = [c.strip() for c in df.columns]
    if "time" not in df.columns or "dl_brate" not in df.columns:
        if verbose:
            print(f"[WARN] enb_metrics.csv missing required cols in {path}. cols={list(df.columns)}")
        return _safe_empty_df(_ENB_COLS)

    keep = [c for c in _ENB_COLS if c in df.columns]
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
    """
    UE metrics CSV has RELATIVE time from experiment start (ms):
      time;cc;pci;...
    """
    try:
        df = _read_csv_sniff(path, prefer_sep=";")
    except EmptyDataError:
        if verbose:
            print(f"[WARN] Empty ue_metrics.csv: {path}")
        return _safe_empty_df(["time"])
    except Exception as e:
        if verbose:
            print(f"[WARN] Failed reading ue_metrics.csv: {path} ({e})")
        return _safe_empty_df(["time"])

    if df.empty:
        return _safe_empty_df(["time"])

    df.columns = [c.strip() for c in df.columns]
    if "time" not in df.columns:
        if verbose:
            print(f"[WARN] ue_metrics.csv missing 'time' in {path}. cols={list(df.columns)}")
        return _safe_empty_df(["time"])

    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    return df


def _aggregate_ues(
    ue_metric_files: List[Path],
    start_time_ms: float,
    verbose: bool = False
) -> Tuple[pd.DataFrame, int, int]:
    """
    UE time is RELATIVE ms → convert to absolute:
      time_abs = start_time_ms + time

    Returns:
      - dataframe with 'time_abs' + aggregated features (mean across UEs)
      - total UE rows read
      - unique time_abs count
    """
    frames: List[pd.DataFrame] = []
    total_rows = 0

    for f in ue_metric_files:
        u = _read_ue_metrics(f, verbose=verbose)
        if u.empty:
            continue

        available_src = [c for c in _UE_TO_FEATURE.keys() if c in u.columns]
        if not available_src:
            continue

        sub = u[["time"] + available_src].copy()
        for c in available_src:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")

        sub["time_abs"] = start_time_ms + sub["time"]
        sub = sub.drop(columns=["time"])

        total_rows += len(sub)
        frames.append(sub)

    if not frames:
        return _safe_empty_df(["time_abs"]), 0, 0

    all_ue = pd.concat(frames, axis=0, ignore_index=True)

    # Mean across UEs per absolute time
    agg = all_ue.groupby("time_abs", as_index=False).mean(numeric_only=True)

    rename = {src: dst for src, dst in _UE_TO_FEATURE.items() if src in agg.columns}
    agg = agg.rename(columns=rename)

    return agg, total_rows, len(agg)


def load_timeseries_from_kpm(
    root_dir: str | Path,
    n_steps: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build traffic_data.csv rows from dataset-kpm folder.

    - ENB time: absolute epoch ms
    - UE time: relative ms -> convert to absolute using ENB start time
    - Merge by nearest timestamp (merge_asof) with tolerance.
    - Stops early when n_steps is reached.
    """
    root = Path(root_dir)
    exp_dirs = sorted(root.glob("**/cluster_*/slicing_*/scheduling_*/RESERVATION-*"))

    if verbose:
        print(f"[INFO] Input root: {root.resolve()}")
        print(f"[INFO] Found {len(exp_dirs)} experiment dirs (RESERVATION-*)")

    if not exp_dirs:
        raise FileNotFoundError(
            f"No RESERVATION-* dirs found under {root}. "
            f"Expected pattern: **/cluster_*/slicing_*/scheduling_*/RESERVATION-*"
        )

    out_frames: List[pd.DataFrame] = []
    used = 0
    skipped = 0
    total_rows = 0

    for i, exp in enumerate(exp_dirs, start=1):
        if n_steps is not None and n_steps > 0 and total_rows >= n_steps:
            if verbose:
                print(f"[INFO] Reached requested n_steps={n_steps}. Stopping early.")
            break

        enb_path = exp / "bs" / "enb_metrics.csv"
        if not enb_path.exists():
            skipped += 1
            if verbose:
                print(f"[SKIP] {i}/{len(exp_dirs)} no enb_metrics.csv: {exp}")
            continue

        enb = _read_enb_metrics(enb_path, verbose=verbose)
        if enb.empty:
            skipped += 1
            if verbose:
                print(f"[SKIP] {i}/{len(exp_dirs)} enb empty/unreadable: {enb_path}")
            continue

        enb = enb.sort_values("time")
        start_time_ms = float(enb["time"].iloc[0])

        ue_files = sorted(exp.glob("ue_*/ue_metrics.csv"))
        ue_agg, ue_total_rows, ue_unique_times = _aggregate_ues(
            ue_files, start_time_ms=start_time_ms, verbose=verbose
        )

        # Merge ENB (absolute time) with UE (absolute time_abs) by nearest within tolerance
        if not ue_agg.empty and "time_abs" in ue_agg.columns:
            ue_agg = ue_agg.sort_values("time_abs")

            df = pd.merge_asof(
                enb.sort_values("time"),
                ue_agg.sort_values("time_abs"),
                left_on="time",
                right_on="time_abs",
                direction="nearest",
                tolerance=200.0,  # ms (ENB sampling ~250ms)
            )
            df = df.drop(columns=["time_abs"], errors="ignore")
        else:
            df = enb.copy()

        # Target and context
        df["traffic_load"] = pd.to_numeric(df["dl_brate"], errors="coerce")
        df["ue_count"] = pd.to_numeric(df.get("nof_ue", 0), errors="coerce")

        # Ensure required columns exist
        for col in FEATURE_ORDER + ["traffic_load"]:
            if col not in df.columns:
                df[col] = 0.0

        # Numeric cleanup
        for col in ["time"] + FEATURE_ORDER + ["traffic_load"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df = df[["time"] + FEATURE_ORDER + ["traffic_load"]].copy()

        # Trim to remaining n_steps
        if n_steps is not None and n_steps > 0:
            remaining = n_steps - total_rows
            if remaining <= 0:
                break
            if len(df) > remaining:
                df = df.iloc[:remaining].copy()

        out_frames.append(df)
        used += 1
        total_rows += len(df)

        if verbose:
            print(
                f"[OK] {i}/{len(exp_dirs)} {exp} | "
                f"enb_rows={len(enb)} | ue_files={len(ue_files)} | "
                f"ue_rows_read={ue_total_rows} | ue_agg_times={ue_unique_times} | "
                f"merged_rows={len(df)} | total_rows={total_rows} | "
                f"enb_start={int(start_time_ms)}"
            )

        if n_steps is not None and n_steps > 0 and total_rows >= n_steps:
            if verbose:
                print(f"[INFO] Reached requested n_steps={n_steps}. Stopping early.")
            break

    if not out_frames:
        raise RuntimeError(
            f"Nothing loaded. Experiments found={len(exp_dirs)} but all skipped. "
            f"Check folder structure and whether enb_metrics.csv files are non-empty."
        )

    full = pd.concat(out_frames, axis=0, ignore_index=True)
    full = full.sort_values("time").reset_index(drop=True)

    if n_steps is not None and n_steps > 0 and len(full) > n_steps:
        full = full.iloc[:n_steps].copy()

    if verbose:
        print(f"[INFO] Loaded experiments used={used}, skipped={skipped}, total_rows={len(full)}")
        print(f"[INFO] Final columns: {list(full.columns)}")

    return full


# ---------------------------
# Training prep
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
