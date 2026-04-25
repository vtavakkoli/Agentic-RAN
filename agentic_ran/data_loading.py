from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS_GLOB = "**/*_metrics.csv"
METRIC_COLUMNS = [
    "Timestamp",
    "num_ues",
    "IMSI",
    "RNTI",
    "empty_1",
    "slicing_enabled",
    "slice_id",
    "slice_prb",
    "power_multiplier",
    "scheduling_policy",
    "empty_2",
    "dl_mcs",
    "dl_n_samples",
    "dl_buffer [bytes]",
    "tx_brate downlink [Mbps]",
    "tx_pkts downlink",
    "tx_errors downlink (%)",
    "dl_cqi",
    "empty_3",
    "ul_mcs",
    "ul_n_samples",
    "ul_buffer [bytes]",
    "rx_brate uplink [Mbps]",
    "rx_pkts uplink",
    "rx_errors uplink (%)",
    "ul_rssi",
    "ul_sinr",
    "phr",
    "empty_4",
    "sum_requested_prbs",
    "sum_granted_prbs",
    "empty_5",
    "dl_pmi",
    "dl_ri",
    "ul_n",
    "ul_turbo_iters",
]
DEFAULT_FEATURES = [
    "num_ues",
    "slice_id",
    "slice_prb",
    "power_multiplier",
    "dl_mcs",
    "dl_n_samples",
    "dl_buffer [bytes]",
    "dl_cqi",
    "ul_mcs",
    "ul_n_samples",
    "ul_buffer [bytes]",
    "rx_brate uplink [Mbps]",
    "ul_rssi",
    "ul_sinr",
    "phr",
    "sum_requested_prbs",
    "sum_granted_prbs",
    "ratio_granted_req",
    "dl_pmi",
    "dl_ri",
    "ul_n",
    "ul_turbo_iters",
]
DEFAULT_TARGET_COL = "tx_brate downlink [Mbps]"


def load_dataset(
    shared_data_dir: Path,
    max_rows: int = 4000,
    selected_features: list[str] | None = None,
    target_col: str = DEFAULT_TARGET_COL,
    keep_zero_requested_prbs: bool = False,
) -> tuple[pd.DataFrame, dict]:
    selected_features = selected_features or DEFAULT_FEATURES
    csv_files = sorted(shared_data_dir.glob(METRICS_GLOB)) if shared_data_dir.exists() else []

    if not csv_files:
        rng = np.random.default_rng(42)
        n_rows = 2500
        n_features = min(len(selected_features), 10)
        x = rng.normal(size=(n_rows, n_features))
        coeffs = rng.normal(size=n_features)
        noise = 0.15 * rng.normal(size=n_rows)
        y = x @ coeffs + noise
        cols = selected_features[:n_features]
        df = pd.DataFrame(x, columns=cols)
        df["target"] = y
        return df, {"source": "synthetic", "files_used": [], "rows": int(df.shape[0])}

    frames = []
    rows_per = max(200, max_rows // min(len(csv_files), 8))
    used_files = []

    for file in csv_files[:8]:
        try:
            part = pd.read_csv(file, names=METRIC_COLUMNS, header=0, usecols=METRIC_COLUMNS)
            part["sum_requested_prbs"] = pd.to_numeric(part["sum_requested_prbs"], errors="coerce")
            part["sum_granted_prbs"] = pd.to_numeric(part["sum_granted_prbs"], errors="coerce")
            part["ratio_granted_req"] = np.clip(
                np.nan_to_num(part["sum_granted_prbs"] / part["sum_requested_prbs"], nan=0.0, posinf=0.0, neginf=0.0),
                a_min=0.0,
                a_max=1.0,
            )
            if not keep_zero_requested_prbs:
                part = part.loc[part["sum_requested_prbs"] > 0].copy()
            part["dl_buffer [bytes]"] = pd.to_numeric(part["dl_buffer [bytes]"], errors="coerce") / 100000.0

            feature_cols = [c for c in selected_features if c in part.columns]
            if target_col not in part.columns or not feature_cols:
                continue

            narrow = part.loc[:, feature_cols + [target_col]].copy()
            narrow = narrow.apply(pd.to_numeric, errors="coerce").dropna()
            if narrow.empty:
                continue
            narrow = narrow.head(rows_per)
            narrow = narrow.rename(columns={target_col: "target"})
            frames.append(narrow)
            used_files.append(str(file))
        except Exception:
            continue

    if not frames:
        return load_dataset(Path("/nonexistent"), max_rows=max_rows)

    df = pd.concat(frames, axis=0, ignore_index=True)
    summary = {
        "source": "shared_data",
        "file_glob": METRICS_GLOB,
        "files_used": used_files,
        "metrics_file_count": len(used_files),
        "rows": int(df.shape[0]),
        "selected_features": [c for c in df.columns if c != "target"],
        "target_column": target_col,
    }
    return df, summary


def write_data_summary(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
