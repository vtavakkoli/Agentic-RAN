from __future__ import annotations

import argparse
import os
from pathlib import Path

from agentic_ran.data_loading import DEFAULT_FEATURES, DEFAULT_TARGET_COL
from scripts.prepare_splits import build_dataset, split_and_save


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate train/test-ready dataset splits.")
    parser.add_argument("--max-files", type=int, default=int(os.getenv("PREP_MAX_FILES", "240")))
    parser.add_argument("--rows-per-file", type=int, default=int(os.getenv("PREP_ROWS_PER_FILE", "300")))
    parser.add_argument("--feature-col", action="append", dest="feature_cols", default=None)
    parser.add_argument("--target-col", default=DEFAULT_TARGET_COL)
    parser.add_argument("--keep-zero-requested-prbs", action="store_true")
    args = parser.parse_args()

    input_dirs = [Path("dataset/slice_mixed"), Path("dataset/slice_traffic"), Path("dataset")]
    per_file_frames, _ = build_dataset(
        input_dirs=input_dirs,
        max_files=args.max_files,
        rows_per_file=args.rows_per_file,
        selected_features=args.feature_cols or DEFAULT_FEATURES,
        target_col=args.target_col,
        keep_zero_requested_prbs=args.keep_zero_requested_prbs,
    )
    split_and_save(per_file_frames, output_dir=Path("shared_data/splits"))
    print("[generate_data] Dataset prepared at shared_data/splits")


if __name__ == "__main__":
    main()
