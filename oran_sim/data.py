from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from oran_sim.config import FEATURE_ORDER


def normalize_column_name(name: str) -> str:
    cleaned = name.strip().lower()
    cleaned = cleaned.replace("%", "pct")
    cleaned = cleaned.replace("[", "_").replace("]", "")
    cleaned = cleaned.replace("(", "_").replace(")", "")
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def _read_csv_robust(path: Path, prefer_sep: str | None = None) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    options = [prefer_sep, ",", ";", "\\t", None]
    for sep in options:
        if sep is None:
            df = pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip")
        else:
            try:
                df = pd.read_csv(path, sep=sep, on_bad_lines="skip")
            except Exception:
                continue
        if df.shape[1] > 0:
            df = df.loc[:, [str(c).strip() != "" and not str(c).startswith("Unnamed") for c in df.columns]]
            df.columns = [normalize_column_name(str(c)) for c in df.columns]
            return df
    return pd.DataFrame()


COLUMN_ALIASES = {
    "tx_brate_downlink_mbps": "tx_brate_dl_mbps",
    "tx_brate_downlink": "tx_brate_dl_mbps",
    "dl_brate": "tx_brate_dl_mbps",
    "dl_brate_mbps": "tx_brate_dl_mbps",
    "tx_brate_uplink_mbps": "tx_brate_ul_mbps",
    "ul_brate": "tx_brate_ul_mbps",
    "nof_ue": "num_ues",
    "ue_count": "num_ues",
    "dl_buffer_bytes": "dl_buffer_bytes",
    "dl_buffer": "dl_buffer_bytes",
    "ul_buffer": "ul_buffer_bytes",
    "ul_buff": "ul_buffer_bytes",
    "sum_requested_prb": "sum_requested_prbs",
    "sum_granted_prb": "sum_granted_prbs",
}


def _apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    rename = {c: COLUMN_ALIASES[c] for c in df.columns if c in COLUMN_ALIASES}
    return df.rename(columns=rename)


def _coalesce_col(df: pd.DataFrame, target: str, candidates: Iterable[str], default: float = 0.0) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def _load_flow_features(exp_dir: Path) -> pd.DataFrame:
    flow_files = sorted(exp_dir.glob("**/*flow*.csv")) + sorted(exp_dir.glob("**/*flows*.csv"))
    frames: list[pd.DataFrame] = []
    for flow_file in flow_files:
        fdf = _apply_aliases(_read_csv_robust(flow_file))
        if fdf.empty:
            continue
        time_col = "time_ms" if "time_ms" in fdf.columns else ("time" if "time" in fdf.columns else None)
        if time_col is None:
            continue
        payload = _coalesce_col(fdf, "payload_bytes", ["payload_bytes", "bytes", "pkt_size_bytes"], default=0.0)
        latency = _coalesce_col(fdf, "latency_ms", ["latency_ms", "delay_ms", "rtt_ms"], default=0.0)
        chunk = pd.DataFrame(
            {
                "time_ms": pd.to_numeric(fdf[time_col], errors="coerce"),
                "latency_ms": latency,
                "payload_bytes": payload,
            }
        ).dropna(subset=["time_ms"])
        chunk = chunk.sort_values("time_ms")
        chunk["jitter_ms"] = chunk["latency_ms"].diff().abs().fillna(0.0)
        frames.append(chunk[["time_ms", "latency_ms", "jitter_ms", "payload_bytes"]])

    if not frames:
        return pd.DataFrame(columns=["time_ms", "latency_ms", "jitter_ms", "payload_bytes"])

    merged = pd.concat(frames, ignore_index=True)
    return merged.groupby("time_ms", as_index=False).mean(numeric_only=True)


def load_timeseries_from_kpm(root_dir: str | Path, n_steps: int | None = None, verbose: bool = True) -> pd.DataFrame:
    root = Path(root_dir)
    reservations = sorted(root.glob("**/RESERVATION-*"))
    if not reservations:
        raise FileNotFoundError(f"No RESERVATION-* folders found under {root}")

    all_frames: list[pd.DataFrame] = []
    for res in reservations:
        bs_metrics = sorted(res.glob("bs/*_metrics.csv")) + sorted(res.glob("bs/enb_metrics.csv"))
        bs_frames: list[pd.DataFrame] = []
        for m in bs_metrics:
            bdf = _apply_aliases(_read_csv_robust(m, prefer_sep=","))
            if bdf.empty:
                continue
            time_col = "time_ms" if "time_ms" in bdf.columns else ("time" if "time" in bdf.columns else None)
            if time_col is None:
                continue
            bdf["time_ms"] = pd.to_numeric(bdf[time_col], errors="coerce")
            bs_frames.append(bdf.dropna(subset=["time_ms"]))

        if not bs_frames:
            if verbose:
                print(f"[WARN] skipping {res}: no bs metrics")
            continue

        bs_df = pd.concat(bs_frames, ignore_index=True)
        bs_df = bs_df.groupby("time_ms", as_index=False).mean(numeric_only=True)
        bs_df["time_ms"] = pd.to_numeric(bs_df["time_ms"], errors="coerce").astype(float)
        bs_df = bs_df.sort_values("time_ms")

        ue_files = sorted(res.glob("ue_*/ue_metrics.csv"))
        ue_chunks: list[pd.DataFrame] = []
        start_time = float(bs_df["time_ms"].min())
        for ue in ue_files:
            udf = _apply_aliases(_read_csv_robust(ue, prefer_sep=";"))
            if udf.empty:
                continue
            rel_col = "time" if "time" in udf.columns else ("time_ms" if "time_ms" in udf.columns else None)
            if rel_col is None:
                continue
            udf["time_ms"] = start_time + pd.to_numeric(udf[rel_col], errors="coerce")
            keep = [c for c in FEATURE_ORDER if c in udf.columns]
            ue_chunks.append(udf[["time_ms"] + keep])

        if ue_chunks:
            ue_df = pd.concat(ue_chunks, ignore_index=True)
            ue_df = ue_df.groupby("time_ms", as_index=False).mean(numeric_only=True)
            ue_df["time_ms"] = pd.to_numeric(ue_df["time_ms"], errors="coerce").astype(float)
            merged = pd.merge_asof(
                bs_df.sort_values("time_ms"),
                ue_df.sort_values("time_ms"),
                on="time_ms",
                direction="nearest",
                tolerance=200,
            )
        else:
            merged = bs_df

        flow_df = _load_flow_features(res)
        if not flow_df.empty:
            flow_df["time_ms"] = pd.to_numeric(flow_df["time_ms"], errors="coerce").astype(float)
            merged = pd.merge_asof(
                merged.sort_values("time_ms"),
                flow_df.sort_values("time_ms"),
                on="time_ms",
                direction="nearest",
                tolerance=200,
            )

        # Core feature synthesis
        merged["num_ues"] = _coalesce_col(merged, "num_ues", ["num_ues"], default=0.0)
        merged["sum_requested_prbs"] = _coalesce_col(
            merged, "sum_requested_prbs", ["sum_requested_prbs", "dl_requested_prbs"], default=0.0
        )
        merged["sum_granted_prbs"] = _coalesce_col(
            merged, "sum_granted_prbs", ["sum_granted_prbs", "dl_granted_prbs"], default=0.0
        )
        merged["dl_buffer_bytes"] = _coalesce_col(merged, "dl_buffer_bytes", ["dl_buffer_bytes"], default=0.0)
        merged["ul_buffer_bytes"] = _coalesce_col(merged, "ul_buffer_bytes", ["ul_buffer_bytes"], default=0.0)
        merged["tx_errors_dl_pct"] = _coalesce_col(merged, "tx_errors_dl_pct", ["tx_errors_dl_pct", "dl_bler"], default=0.0)
        merged["rx_errors_ul_pct"] = _coalesce_col(merged, "rx_errors_ul_pct", ["rx_errors_ul_pct", "ul_bler"], default=0.0)
        merged["dl_mcs"] = _coalesce_col(merged, "dl_mcs", ["dl_mcs"], default=0.0)
        merged["ul_mcs"] = _coalesce_col(merged, "ul_mcs", ["ul_mcs"], default=0.0)
        merged["dl_cqi"] = _coalesce_col(merged, "dl_cqi", ["dl_cqi", "cqi"], default=0.0)
        merged["ul_sinr"] = _coalesce_col(merged, "ul_sinr", ["ul_sinr", "ul_snr"], default=0.0)
        merged["ul_rssi"] = _coalesce_col(merged, "ul_rssi", ["ul_rssi", "rssi"], default=-95.0)

        if "slicing" in str(res).lower() or "slice" in str(res).lower():
            merged["slicing_enabled"] = 1.0
        else:
            merged["slicing_enabled"] = 0.0
        merged["slice_id"] = float(int(re.search(r"slice[_-]?(\d+)", str(res).lower()).group(1)) if re.search(r"slice[_-]?(\d+)", str(res).lower()) else 0)
        merged["slice_prb"] = _coalesce_col(merged, "slice_prb", ["slice_prb"], default=50.0)
        merged["scheduling_policy"] = 1.0 if "rr" in str(res).lower() else 0.0

        merged["latency_ms"] = _coalesce_col(merged, "latency_ms", ["latency_ms"], default=0.0)
        merged["jitter_ms"] = _coalesce_col(merged, "jitter_ms", ["jitter_ms"], default=0.0)
        merged["payload_bytes"] = _coalesce_col(merged, "payload_bytes", ["payload_bytes"], default=0.0)

        traffic_primary = _coalesce_col(merged, "traffic_load", ["tx_brate_dl_mbps"])
        traffic_fallback = _coalesce_col(merged, "traffic_load", ["dl_brate", "dl_brate_mbps"])
        merged["traffic_load"] = traffic_primary.where(traffic_primary > 0, traffic_fallback)

        for col in FEATURE_ORDER + ["ul_rssi", "traffic_load"]:
            if col not in merged.columns:
                merged[col] = 0.0

        merged = merged[["time_ms", "traffic_load", "ul_rssi"] + FEATURE_ORDER].copy()
        merged = merged.sort_values("time_ms")
        merged[FEATURE_ORDER + ["ul_rssi", "traffic_load"]] = merged[FEATURE_ORDER + ["ul_rssi", "traffic_load"]].ffill().fillna(0.0)
        merged["time_ms"] = merged["time_ms"].astype("int64")
        for col in ["traffic_load", "ul_rssi"] + FEATURE_ORDER:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0).astype("float32")

        all_frames.append(merged)

    if not all_frames:
        raise RuntimeError("No valid reservations could be parsed from KPM dataset")

    full = pd.concat(all_frames, ignore_index=True).sort_values("time_ms").reset_index(drop=True)
    if n_steps is not None:
        full = full.iloc[:n_steps].copy()
    return full


def split_by_time(df: pd.DataFrame, split: tuple[float, float, float] = (0.6, 0.3, 0.1)) -> pd.DataFrame:
    if abs(sum(split) - 1.0) > 1e-8:
        raise ValueError("split must sum to 1")
    n = len(df)
    n_train = int(n * split[0])
    n_val = int(n * split[1])
    labels = np.array(["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val), dtype=object)
    out = df.copy()
    out["split"] = labels
    return out


def save_dataframe(df: pd.DataFrame, output: str | Path) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out
