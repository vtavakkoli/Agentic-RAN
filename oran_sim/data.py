from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from sklearn.preprocessing import StandardScaler

from oran_sim.config import FEATURE_ORDER


# ---------------------------
# Helpers
# ---------------------------

_ENB_COLS = ["time", "nof_ue", "dl_brate", "ul_brate"]

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


def _detect_sep(path: Path) -> str:
    """
    Very robust delimiter detection between ',', ';', and '\t'.
    Uses the first non-empty line and picks the delimiter with the most splits.
    """
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if s:
                    # choose the one that yields more columns
                    candidates = [",", ";", "\t", "|"]
                    best = max(candidates, key=lambda d: s.count(d))
                    # if it finds none, default to comma
                    return best if s.count(best) > 0 else ","
    except Exception:
        pass
    return ","


def _read_csv_auto(path: Path, verbose: bool = False) -> pd.DataFrame:
    # Skip empty quickly
    try:
        if path.stat().st_size == 0:
            if verbose:
                print(f"[WARN] Empty file: {path}")
            return pd.DataFrame()
    except OSError:
        if verbose:
            print(f"[WARN] Cannot stat file: {path}")
        return pd.DataFrame()

    sep = _detect_sep(path)

    try:
        df = pd.read_csv(path, sep=sep, engine="python")
    except EmptyDataError:
        if verbose:
            print(f"[WARN] EmptyDataError: {path}")
        return pd.DataFrame()
    except Exception as e:
        if verbose:
            print(f"[WARN] read_csv failed ({sep=}) for {path}: {e}")
        return pd.DataFrame()

    # strip whitespace in col names
    df.columns = [str(c).strip() for c in df.columns]

    # If delimiter detection failed, sometimes you get 1 column with delimiters inside.
    # Fallback: try the other common delimiter if only 1 column exists.
    if df.shape[1] == 1:
        fallback_sep = ";" if sep == "," else ","
        try:
            df2 = pd.read_csv(path, sep=fallback_sep, engine="python")
            df2.columns = [str(c).strip() for c in df2.columns]
            if df2.shape[1] > df.shape[1]:
                if verbose:
                    print(f"[INFO] Fallback delimiter worked for {path}: using '{fallback_sep}'")
                df = df2
        except Exception:
            pass

    if verbose:
        print(f"[DEBUG] Read {path.name} sep='{sep}' cols={list(df.columns)[:10]}... ({df.shape[1]} cols)")

    return df


def _to_numeric_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------
# Synthetic generator (kept, but not relevant here)
# ---------------------------

def generate_timeseries(n_steps: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    traffic = rng.normal(0.5, 0.1, n_steps).clip(0, 1.2)
    return pd.DataFrame(
        {"timestamp": pd.date_range("2024-01-01", periods=n_steps, freq="h"), "traffic_load": traffic}
    )


# ---------------------------
# Real dataset loader
# ---------------------------

def _read_enb_metrics(path: Path, verbose: bool = False) -> pd.DataFrame:
    df = _read_csv_auto(path, verbose=verbose)
    if df.empty:
        return _safe_empty_df(_ENB_COLS)

    # Ensure required columns exist
    missing = [c for c in ["time", "dl_brate"] if c not in df.columns]
    if missing:
        if verbose:
            print(f"[WARN] enb_metrics missing {missing} in {path}. cols={list(df.columns)}")
        return _safe_empty_df(_ENB_COLS)

    keep = [c for c in _ENB_COLS if c in df.columns]
    df = df[keep].copy()

    df = _to_numeric_cols(df, keep)
    df = df.dropna(subset=["time"]).sort_values("time")
    return df


def _read_ue_metrics(path: Path, verbose: bool = False) -> pd.DataFrame:
    df = _read_csv_auto(path, verbose=verbose)
    if df.empty:
        return _safe_empty_df(["time"])

    if "time" not in df.columns:
        if verbose:
            print(f"[WARN] ue_metrics missing 'time' in {path}. cols={list(df.columns)}")
        return _safe_empty_df(["time"])

    df = df.copy()
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    return df


def _aggregate_ues(ue_metric_files: List[Path], verbose: bool = False) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for f in ue_metric_files:
        u = _read_ue_metrics(f, verbose=verbose)
        if u.empty or "time" not in u.columns:
            continue

        available_src = [c for c in _UE_TO_FEATURE.keys() if c in u.columns]
        if not available_src:
            if verbose:
                print(f"[WARN] No mapped UE columns in {f}. Have={list(u.columns)[:20]}")
            continue

        sub = u[["time"] + available_src].copy()
        sub = _to_numeric_cols(sub, available_src)
        frames.append(sub)

    if not frames:
        return _safe_empty_df(["time"])

    all_ue = pd.concat(frames, axis=0, ignore_index=True)
    agg = all_ue.groupby("time", as_index=False).mean(numeric_only=True)

    rename = {src: dst for src, dst in _UE_TO_FEATURE.items() if src in agg.columns}
    agg = agg.rename(columns=rename)
    return agg


def load_timeseries_from_kpm(
    root_dir: str | Path,
    n_steps: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    import time as _time

    root = Path(root_dir)
    exp_dirs = sorted(root.glob("cluster_*/slicing_*/scheduling_*/RESERVATION-*"))
    if not exp_dirs:
        raise FileNotFoundError(
            f"No RESERVATION-* folders found under: {root}. "
            f"Expected: cluster_*/slicing_*/scheduling_*/RESERVATION-*"
        )

    total = len(exp_dirs)
    out_frames: List[pd.DataFrame] = []
    rows_so_far = 0
    used = 0
    skipped = 0

    t0 = _time.time()
    last_print = t0

    def _print_progress(i: int) -> None:
        nonlocal last_print
        if not verbose:
            return
        now = _time.time()
        if now - last_print < 0.5 and i < total:
            return
        last_print = now
        elapsed = now - t0
        rate = elapsed / i if i > 0 else 0.0
        eta = rate * (total - i) if i > 0 else 0.0
        pct = (i / total) * 100.0
        print(
            f"[PROGRESS] {i}/{total} ({pct:5.1f}%) | rows={rows_so_far} | elapsed={elapsed:6.1f}s | ETA≈{eta:6.1f}s"
        )

    for i, exp in enumerate(exp_dirs, start=1):
        enb_path = exp / "bs" / "enb_metrics.csv"
        if not enb_path.exists():
            skipped += 1
            _print_progress(i)
            continue

        enb = _read_enb_metrics(enb_path, verbose=verbose)
        if enb.empty:
            skipped += 1
            _print_progress(i)
            continue

        ue_files = sorted(exp.glob("ue_*/ue_metrics.csv"))
        ue_agg = _aggregate_ues(ue_files, verbose=verbose)

        df = enb.merge(ue_agg, on="time", how="left") if (not ue_agg.empty and "time" in ue_agg.columns) else enb.copy()

        # Target from ENB dl_brate
        df["traffic_load"] = pd.to_numeric(df["dl_brate"], errors="coerce")

        # ue_count from ENB nof_ue
        df["ue_count"] = pd.to_numeric(df.get("nof_ue", np.nan), errors="coerce")

        # Ensure required columns exist
        for col in FEATURE_ORDER + ["traffic_load"]:
            if col not in df.columns:
                df[col] = 0.0

        # Make numeric + fill
        for col in ["time"] + FEATURE_ORDER + ["traffic_load"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        df = df[["time"] + FEATURE_ORDER + ["traffic_load"]].copy()

        out_frames.append(df)
        used += 1
        rows_so_far += len(df)

        if n_steps is not None and n_steps > 0 and rows_so_far >= n_steps:
            _print_progress(i)
            if verbose:
                print(f"[INFO] Reached requested n_steps={n_steps} (rows_so_far={rows_so_far}). Stopping early.")
            break

        _print_progress(i)

    if not out_frames:
        raise RuntimeError(f"No usable experiments loaded. Total={total}, skipped={skipped}")

    full = pd.concat(out_frames, axis=0, ignore_index=True)
    full = full.sort_values("time").reset_index(drop=True)

    if n_steps is not None and n_steps > 0 and len(full) > n_steps:
        full = full.iloc[:n_steps].copy()

    if verbose:
        print(f"[DONE] experiments used={used}/{total}, skipped={skipped}, rows={len(full)}")
        # quick sanity check: show non-zero counts
        nz = (full[FEATURE_ORDER + ["traffic_load"]] != 0).sum().to_dict()
        print(f"[SANITY] non-zero counts: {nz}")

    return full


# ---------------------------
# Training prep + save (unchanged)
# ---------------------------

def build_sequences(features: np.ndarray, target: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i in range(seq_len, len(features)):
        x.append(features[i - seq_len : i])
        y.append(target[i])
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


def prepare_dataset(
    csv_path: str, feature_count: int, seq_len: int
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, StandardScaler, list[str]]:
    df = pd.read_csv(csv_path)

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
