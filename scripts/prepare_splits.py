from __future__ import annotations

import argparse
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

LEGACY_FEATURE_NAMES = {"time", "nof_ue", "ul_brate"}
DEFAULT_TARGET_COL = "tx_brate downlink [Mbps]"


def _iter_metrics_files(input_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for directory in input_dirs:
        if not directory.exists():
            continue
        files.extend(sorted(directory.glob(METRICS_GLOB)))
    deduped = sorted(set(files))
    return deduped


def _validate_selected_features(selected_features: list[str]) -> None:
    unknown = [f for f in selected_features if f not in set(METRIC_COLUMNS) | {"ratio_granted_req"}]
    if unknown:
        raise ValueError(f"Selected features are not present in the srsLTE metrics schema: {unknown}")

    invalid_legacy = [f for f in selected_features if f in LEGACY_FEATURE_NAMES and f not in METRIC_COLUMNS]
    if invalid_legacy:
        raise ValueError(
            "Legacy feature names are not valid for *_metrics.csv schema: "
            f"{invalid_legacy}. Use schema-aligned names instead."
        )


def _read_metrics_frame(file: Path) -> pd.DataFrame:
    return pd.read_csv(file, names=METRIC_COLUMNS, header=0, usecols=METRIC_COLUMNS)


def _standardize_frame(
    df: pd.DataFrame,
    selected_features: list[str],
    target_col: str,
    keep_zero_requested_prbs: bool,
) -> pd.DataFrame:
    working = df.copy()

    working["sum_requested_prbs"] = pd.to_numeric(working["sum_requested_prbs"], errors="coerce")
    working["sum_granted_prbs"] = pd.to_numeric(working["sum_granted_prbs"], errors="coerce")

    working["ratio_granted_req"] = np.clip(
        np.nan_to_num(working["sum_granted_prbs"] / working["sum_requested_prbs"], nan=0.0, posinf=0.0, neginf=0.0),
        a_min=0.0,
        a_max=1.0,
    )

    if not keep_zero_requested_prbs:
        working = working.loc[working["sum_requested_prbs"] > 0].copy()

    if "dl_buffer [bytes]" in working.columns:
        working["dl_buffer [bytes]"] = pd.to_numeric(working["dl_buffer [bytes]"], errors="coerce") / 100000.0

    required_cols = sorted(set(selected_features + [target_col, "Timestamp"]))
    missing = [c for c in required_cols if c not in working.columns]
    if missing:
        raise ValueError(f"Missing required columns from metrics file: {missing}")

    standardized = working.loc[:, required_cols].copy()
    for col in selected_features + [target_col]:
        standardized[col] = pd.to_numeric(standardized[col], errors="coerce")

    standardized["Timestamp"] = pd.to_datetime(standardized["Timestamp"], errors="coerce")
    standardized = standardized.sort_values("Timestamp", kind="stable")
    standardized = standardized.dropna(subset=selected_features + [target_col]).reset_index(drop=True)

    if standardized.empty:
        return standardized

    out = standardized.loc[:, ["Timestamp", *selected_features, target_col]].copy()
    out = out.rename(columns={target_col: "target"})
    return out


def build_dataset(
    input_dirs: list[Path],
    max_files: int,
    rows_per_file: int,
    selected_features: list[str],
    target_col: str,
    keep_zero_requested_prbs: bool,
) -> tuple[list[tuple[str, pd.DataFrame]], dict]:
    _validate_selected_features(selected_features)
    metrics_files = _iter_metrics_files(input_dirs)
    per_file_frames: list[tuple[str, pd.DataFrame]] = []
    used_files: list[str] = []

    for file in metrics_files[:max_files]:
        try:
            part = _read_metrics_frame(file)
            standardized_df = _standardize_frame(
                part,
                selected_features=selected_features,
                target_col=target_col,
                keep_zero_requested_prbs=keep_zero_requested_prbs,
            )
        except Exception:
            continue

        if standardized_df.empty:
            continue

        per_file_frames.append((str(file), standardized_df.head(rows_per_file).copy()))
        used_files.append(str(file))

    if not per_file_frames:
        raise RuntimeError(
            "No valid *_metrics.csv files found with the requested schema/features under input folders."
        )

    total_rows = sum(len(frame) for _, frame in per_file_frames)
    summary = {
        "rows": int(total_rows),
        "files_used": used_files,
        "metrics_file_count": len(used_files),
        "max_files": max_files,
        "rows_per_file": rows_per_file,
        "feature_names": selected_features,
        "target_column": target_col,
        "keep_zero_requested_prbs": keep_zero_requested_prbs,
        "file_glob": METRICS_GLOB,
    }
    return per_file_frames, summary


def split_and_save(per_file_frames: list[tuple[str, pd.DataFrame]], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    per_file_rows: dict[str, dict[str, int]] = {}
    for filename, frame in per_file_frames:
        frame = frame.sort_values("Timestamp", kind="stable").reset_index(drop=True)
        n_rows = len(frame)
        train_end = int(n_rows * 0.60)
        val_end = train_end + int(n_rows * 0.10)

        train_df = frame.iloc[:train_end].copy()
        val_df = frame.iloc[train_end:val_end].copy()
        test_df = frame.iloc[val_end:].copy()

        train_parts.append(train_df)
        val_parts.append(val_df)
        test_parts.append(test_df)

        per_file_rows[filename] = {
            "train": int(train_df.shape[0]),
            "val": int(val_df.shape[0]),
            "test": int(test_df.shape[0]),
            "total": int(n_rows),
        }

    train_all = pd.concat(train_parts, ignore_index=True).drop(columns=["Timestamp"], errors="ignore")
    val_all = pd.concat(val_parts, ignore_index=True).drop(columns=["Timestamp"], errors="ignore")
    test_all = pd.concat(test_parts, ignore_index=True).drop(columns=["Timestamp"], errors="ignore")

    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    train_all.to_csv(train_path, index=False)
    val_all.to_csv(val_path, index=False)
    test_all.to_csv(test_path, index=False)

    return {
        "split_ratio": {"train": 0.60, "val": 0.10, "test": 0.30},
        "rows": {
            "train": int(train_all.shape[0]),
            "val": int(val_all.shape[0]),
            "test": int(test_all.shape[0]),
            "total": int(train_all.shape[0] + val_all.shape[0] + test_all.shape[0]),
        },
        "per_file_rows": per_file_rows,
        "files": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build time-aware train/val/test splits from srsLTE *_metrics.csv files.")
    parser.add_argument("--input-dir", action="append", dest="input_dirs", default=[])
    parser.add_argument("--output-dir", default="shared_data/splits")
    parser.add_argument("--target-col", default=DEFAULT_TARGET_COL, help="Explicit target column name in *_metrics.csv files.")
    parser.add_argument("--feature-col", action="append", dest="feature_cols", default=None, help="Feature columns to use; repeat the flag for multiple columns.")
    parser.add_argument("--max-files", type=int, default=240)
    parser.add_argument("--rows-per-file", type=int, default=300)
    parser.add_argument("--keep-zero-requested-prbs", action="store_true", help="Keep rows where sum_requested_prbs <= 0.")
    args = parser.parse_args()

    raw_input_dirs = args.input_dirs or ["dataset/slice_mixed", "dataset/slice_traffic", "dataset"]
    input_dirs = [Path(p) for p in raw_input_dirs]
    selected_features = args.feature_cols if args.feature_cols else DEFAULT_FEATURES
    if args.target_col not in set(METRIC_COLUMNS) | {"ratio_granted_req"}:
        raise ValueError(f"Invalid --target-col '{args.target_col}'. Must be a metrics schema column or ratio_granted_req.")

    per_file_frames, prep_summary = build_dataset(
        input_dirs=input_dirs,
        max_files=args.max_files,
        rows_per_file=args.rows_per_file,
        selected_features=selected_features,
        target_col=args.target_col,
        keep_zero_requested_prbs=args.keep_zero_requested_prbs,
    )
    split_summary = split_and_save(per_file_frames, output_dir=Path(args.output_dir))

    full_summary = {
        "inputs": [str(p) for p in input_dirs],
        "preprocessing": prep_summary,
        "split": split_summary,
    }

    summary_path = Path(args.output_dir) / "summary.json"
    summary_path.write_text(json.dumps(full_summary, indent=2), encoding="utf-8")

    print(json.dumps(full_summary, indent=2))


if __name__ == "__main__":
    main()
