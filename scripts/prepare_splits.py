from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _iter_csv_files(input_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for directory in input_dirs:
        if not directory.exists():
            continue
        files.extend(sorted(directory.rglob("*.csv")))
    return files


def _standardize_frame(
    df: pd.DataFrame,
    max_features: int,
    explicit_target_col: str | None = None,
) -> tuple[pd.DataFrame, dict] | None:
    numeric = df.select_dtypes(include=["number"]).dropna()
    if numeric.shape[1] < 2 or numeric.empty:
        return None

    target_candidates = [
        "target",
        "tx_brate downlink [Mbps]",
        "dl_brate",
        "rx_brate uplink [Mbps]",
        "ul_brate",
    ]
    target_col = explicit_target_col if explicit_target_col in numeric.columns else None
    if target_col is None:
        target_col = next((col for col in target_candidates if col in numeric.columns), numeric.columns[-1])

    features = [col for col in numeric.columns if col != target_col]
    features = features[:max_features]

    out = numeric.loc[:, features].astype(np.float32).copy()
    out["target"] = numeric[target_col].astype(np.float32)
    return out, {"source_target_col": target_col, "selected_features": features}


def build_dataset(
    input_dirs: list[Path],
    max_files: int,
    rows_per_file: int,
    max_features: int,
    target_col: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    csv_files = _iter_csv_files(input_dirs)
    frames: list[pd.DataFrame] = []
    used_files: list[str] = []
    feature_schema: list[str] | None = None
    file_target_columns: dict[str, str] = {}

    for file in csv_files[:max_files]:
        try:
            part = pd.read_csv(file)
        except Exception:
            continue
        standardized = _standardize_frame(part, max_features=max_features, explicit_target_col=target_col)
        if standardized is None:
            continue
        standardized_df, file_meta = standardized
        if feature_schema is None:
            feature_schema = file_meta["selected_features"]
        standardized_df = standardized_df.reindex(columns=(feature_schema + ["target"]), fill_value=0.0)

        frames.append(standardized_df.head(rows_per_file))
        used_files.append(str(file))
        file_target_columns[str(file)] = file_meta["source_target_col"]

    if not frames:
        raise RuntimeError(
            "No valid numeric CSV files found under input folders. "
            "Expected data under dataset/ (for example dataset/slice_mixed and dataset/slice_traffic)."
        )

    combined = pd.concat(frames, ignore_index=True)
    summary = {
        "rows": int(combined.shape[0]),
        "columns": list(combined.columns),
        "files_used": used_files,
        "max_files": max_files,
        "rows_per_file": rows_per_file,
        "max_features": max_features,
        "feature_names": [c for c in combined.columns if c != "target"],
        "target_column": "target",
        "source_target_columns": file_target_columns,
    }
    return combined, summary


def split_and_save(df: pd.DataFrame, output_dir: Path, seed: int = 42) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_rows = len(shuffled)

    train_end = int(n_rows * 0.60)
    val_end = train_end + int(n_rows * 0.10)

    train_df = shuffled.iloc[:train_end]
    val_df = shuffled.iloc[train_end:val_end]
    test_df = shuffled.iloc[val_end:]

    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    return {
        "split_ratio": {"train": 0.60, "val": 0.10, "test": 0.30},
        "rows": {
            "train": int(train_df.shape[0]),
            "val": int(val_df.shape[0]),
            "test": int(test_df.shape[0]),
            "total": int(n_rows),
        },
        "files": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/val/test splits from slice_mixed and slice_traffic datasets.")
    parser.add_argument("--input-dir", action="append", dest="input_dirs", default=[])
    parser.add_argument("--output-dir", default="shared_data/splits")
    parser.add_argument("--target-col", default=None, help="Optional explicit numeric target column name in raw CSVs.")
    parser.add_argument("--max-files", type=int, default=240)
    parser.add_argument("--rows-per-file", type=int, default=300)
    parser.add_argument("--max-features", type=int, default=10)
    args = parser.parse_args()

    raw_input_dirs = args.input_dirs or ["dataset/slice_mixed", "dataset/slice_traffic", "dataset"]
    input_dirs = [Path(p) for p in raw_input_dirs]
    df, prep_summary = build_dataset(
        input_dirs=input_dirs,
        max_files=args.max_files,
        rows_per_file=args.rows_per_file,
        max_features=args.max_features,
        target_col=args.target_col,
    )
    split_summary = split_and_save(df, output_dir=Path(args.output_dir))

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
