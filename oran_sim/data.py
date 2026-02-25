from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from oran_sim.config import FEATURE_ORDER

REQUIRED_OUTPUT_COLUMNS = [
    "time_ms",
    "traffic_load",
    "num_ues",
    "dl_mcs",
    "ul_mcs",
    "dl_cqi",
    "ul_sinr",
    "ul_rssi",
    "dl_buffer_bytes",
    "ul_buffer_bytes",
    "sum_requested_prbs",
    "sum_granted_prbs",
    "tx_errors_dl_pct",
    "rx_errors_ul_pct",
    "slicing_enabled",
    "slice_id",
    "slice_prb",
    "scheduling_policy",
    "latency_ms",
    "jitter_ms",
    "payload_bytes",
]

_COL_ALIASES = {
    "nof_ue": "num_ues",
    "dl_brate": "enb_dl_brate",
    "tx_brate_downlink_mbps": "tx_brate_dl_mbps",
    "txbrate_downlink_mbps": "tx_brate_dl_mbps",
    "tx_brate_uplink_mbps": "tx_brate_ul_mbps",
    "tx_errors_dl": "tx_errors_dl_pct",
    "rx_errors_ul": "rx_errors_ul_pct",
    "ul_buffer": "ul_buffer_bytes",
    "dl_buffer": "dl_buffer_bytes",
}


def normalize_column_name(name: str) -> str:
    col = str(name).strip().lower()
    col = col.replace("%", "pct").replace("/", "_")
    col = col.replace("[", "_").replace("]", "")
    col = col.replace("(", "_").replace(")", "")
    col = col.replace("-", "_").replace(" ", "_")
    col = re.sub(r"_+", "_", col).strip("_")
    return _COL_ALIASES.get(col, col)


def _read_csv(path: Path, prefer_sep: str = ",") -> pd.DataFrame:
    if path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, sep=prefer_sep)
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    return pd.read_csv(path, sep=None, engine="python")


def _load_bs_metrics(exp_dir: Path) -> pd.DataFrame:
    bs_files = sorted((exp_dir / "bs").glob("*_metrics.csv"))
    frames: List[pd.DataFrame] = []
    for f in bs_files:
        df = _read_csv(f, prefer_sep=",")
        if df.empty:
            continue
        df = df.loc[:, [str(c).strip() != "" for c in df.columns]].copy()
        df.columns = [normalize_column_name(c) for c in df.columns]
        if "time" not in df.columns:
            continue
        df = df.loc[:, ~df.columns.str.startswith("unnamed")]
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time")
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["time_ms"])

    all_bs = pd.concat(frames, ignore_index=True, sort=False)
    numeric_cols = [c for c in all_bs.columns if c != "time"]
    for col in numeric_cols:
        all_bs[col] = pd.to_numeric(all_bs[col], errors="coerce")
    agg = all_bs.groupby("time", as_index=False).mean(numeric_only=True)
    agg = agg.rename(columns={"time": "time_ms"})
    return agg


def _load_enb_metrics(exp_dir: Path) -> pd.DataFrame:
    path = exp_dir / "bs" / "enb_metrics.csv"
    if not path.exists():
        return pd.DataFrame(columns=["time_ms"])
    df = _read_csv(path, prefer_sep=",")
    if df.empty:
        return pd.DataFrame(columns=["time_ms"])
    df.columns = [normalize_column_name(c) for c in df.columns]
    if "time" not in df.columns:
        return pd.DataFrame(columns=["time_ms"])
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    for col in df.columns:
        if col != "time":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.rename(columns={"time": "time_ms"})


def _load_ue_metrics(exp_dir: Path, start_time_ms: float) -> pd.DataFrame:
    ue_files = sorted(exp_dir.glob("ue_*/ue_metrics.csv"))
    frames: List[pd.DataFrame] = []
    for f in ue_files:
        df = _read_csv(f, prefer_sep=";")
        if df.empty:
            continue
        df.columns = [normalize_column_name(c) for c in df.columns]
        if "time" not in df.columns:
            continue
        keep = [c for c in ["time", "dl_mcs", "ul_mcs", "dl_cqi", "ul_sinr", "ul_rssi"] if c in df.columns]
        if len(keep) <= 1:
            continue
        sub = df[keep].copy()
        for col in keep:
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
        sub["time_ms"] = start_time_ms + sub["time"]
        sub = sub.drop(columns=["time"])
        frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=["time_ms"])
    ue = pd.concat(frames, ignore_index=True, sort=False)
    return ue.groupby("time_ms", as_index=False).mean(numeric_only=True)


def _load_flow_metrics(exp_dir: Path, start_time_ms: float) -> pd.DataFrame:
    flow_files = sorted(exp_dir.glob("**/*flow*.csv"))
    frames: List[pd.DataFrame] = []
    for f in flow_files:
        df = _read_csv(f, prefer_sep=",")
        if df.empty:
            continue
        df.columns = [normalize_column_name(c) for c in df.columns]
        sent_col = next((c for c in df.columns if "sent_time" in c), None)
        recv_col = next((c for c in df.columns if "received_time" in c), None)
        payload_col = next((c for c in df.columns if "payload" in c), None)
        if not sent_col or not recv_col:
            continue
        sent = pd.to_numeric(df[sent_col], errors="coerce")
        recv = pd.to_numeric(df[recv_col], errors="coerce")
        latency_ms = (recv - sent) * 1000.0
        time_ms = np.where(recv < 1e9, start_time_ms + recv * 1000.0, recv)
        sub = pd.DataFrame({"time_ms": time_ms, "latency_ms": latency_ms})
        if payload_col:
            sub["payload_bytes"] = pd.to_numeric(df[payload_col], errors="coerce")
        frames.append(sub)

    if not frames:
        return pd.DataFrame(columns=["time_ms", "latency_ms", "jitter_ms", "payload_bytes"])

    flow = pd.concat(frames, ignore_index=True, sort=False)
    flow = flow.dropna(subset=["time_ms"]).sort_values("time_ms")
    flow["jitter_ms"] = flow["latency_ms"].diff().abs()
    return flow.groupby("time_ms", as_index=False).mean(numeric_only=True)


def _scenario_context(exp_dir: Path) -> Dict[str, object]:
    scheduling_name = exp_dir.parent.name
    slicing_name = exp_dir.parent.parent.name if exp_dir.parent.parent else "slicing_unknown"
    slice_digits = re.findall(r"\d+", slicing_name)
    slice_prb = int(slice_digits[0]) if slice_digits else 0
    slice_id = int(hashlib.md5(slicing_name.encode("utf-8")).hexdigest()[:6], 16) % 1000
    return {
        "slicing_enabled": 1.0 if "slicing" in slicing_name else 0.0,
        "slice_id": float(slice_id),
        "slice_prb": float(slice_prb),
        "scheduling_policy": scheduling_name.replace("scheduling_", ""),
    }


def load_timeseries_from_kpm(root_dir: str | Path, n_steps: Optional[int] = None, verbose: bool = True) -> pd.DataFrame:
    root = Path(root_dir)
    reservations = sorted(root.glob("**/RESERVATION-*"))
    if not reservations:
        raise FileNotFoundError(f"No RESERVATION-* directories found under {root}")

    frames: List[pd.DataFrame] = []
    for exp in reservations:
        bs = _load_bs_metrics(exp)
        if bs.empty:
            continue
        start = float(bs["time_ms"].min())
        enb = _load_enb_metrics(exp)
        ue = _load_ue_metrics(exp, start)
        flow = _load_flow_metrics(exp, start)
        merged = bs.sort_values("time_ms").copy()
        merged["time_ms"] = pd.to_numeric(merged["time_ms"], errors="coerce").astype(float)

        if not enb.empty:
            enb["time_ms"] = pd.to_numeric(enb["time_ms"], errors="coerce").astype(float)
            merged = pd.merge_asof(
                merged.sort_values("time_ms"),
                enb.sort_values("time_ms"),
                on="time_ms",
                direction="nearest",
                tolerance=300,
                suffixes=("", "_enb"),
            )
        if not ue.empty:
            ue["time_ms"] = pd.to_numeric(ue["time_ms"], errors="coerce").astype(float)
            merged = pd.merge_asof(merged.sort_values("time_ms"), ue.sort_values("time_ms"), on="time_ms", direction="nearest", tolerance=300)
        if not flow.empty:
            flow["time_ms"] = pd.to_numeric(flow["time_ms"], errors="coerce").astype(float)
            merged = pd.merge_asof(merged.sort_values("time_ms"), flow.sort_values("time_ms"), on="time_ms", direction="nearest", tolerance=1000)

        ctx = _scenario_context(exp)
        for k, v in ctx.items():
            merged[k] = v

        merged["reservation"] = exp.name
        frames.append(merged)
        if verbose:
            print(f"[DATA] reservation={exp.name} rows={len(merged)}")

    if not frames:
        raise RuntimeError("No usable data parsed from reservations")

    df = pd.concat(frames, ignore_index=True, sort=False)
    df.columns = [normalize_column_name(c) for c in df.columns]

    if "tx_brate_dl_mbps" in df.columns:
        df["traffic_load"] = pd.to_numeric(df["tx_brate_dl_mbps"], errors="coerce")
    elif "enb_dl_brate" in df.columns:
        df["traffic_load"] = pd.to_numeric(df["enb_dl_brate"], errors="coerce")
    else:
        df["traffic_load"] = 0.0

    for col in REQUIRED_OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce")
    numeric_cols = [c for c in REQUIRED_OUTPUT_COLUMNS if c not in {"scheduling_policy"}]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("time_ms").reset_index(drop=True)
    df = df[df["time_ms"].notna()]
    df[numeric_cols] = df[numeric_cols].ffill().fillna(0.0)
    df["scheduling_policy"] = df["scheduling_policy"].ffill().fillna("unknown")
    df["time_ms"] = df["time_ms"].astype("int64")

    result_cols = REQUIRED_OUTPUT_COLUMNS + ["reservation"]
    df = df[result_cols]
    df = df.drop_duplicates(subset=["time_ms", "reservation"], keep="first")
    df = df.sort_values("time_ms").reset_index(drop=True)

    if n_steps is not None and len(df) > n_steps:
        df = df.iloc[:n_steps].copy()
    return df


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
