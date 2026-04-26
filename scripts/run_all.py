from __future__ import annotations

import argparse
import os
from pathlib import Path

from agentic_ran.data_loading import DEFAULT_FEATURES, DEFAULT_TARGET_COL
from agentic_ran.scenarios import SCENARIOS
from scripts.aggregate_report import aggregate
from scripts.prepare_splits import build_dataset, split_and_save
from scripts.run_scenario import run


def prepare_data(
    max_files: int,
    rows_per_file: int,
    feature_cols: list[str],
    target_col: str,
    keep_zero_requested_prbs: bool,
) -> None:
    input_dirs = [Path("dataset/slice_mixed"), Path("dataset/slice_traffic"), Path("dataset")]
    per_file_frames, _ = build_dataset(
        input_dirs=input_dirs,
        max_files=max_files,
        rows_per_file=rows_per_file,
        selected_features=feature_cols,
        target_col=target_col,
        keep_zero_requested_prbs=keep_zero_requested_prbs,
    )
    split_and_save(per_file_frames, output_dir=Path("shared_data/splits"))


def run_all(
    max_files: int,
    rows_per_file: int,
    feature_cols: list[str],
    target_col: str,
    keep_zero_requested_prbs: bool,
    log_target: bool,
    loss: str | None = None,
    peak_weight: float | None = None,
) -> None:
    prepare_data(
        max_files=max_files,
        rows_per_file=rows_per_file,
        feature_cols=feature_cols,
        target_col=target_col,
        keep_zero_requested_prbs=keep_zero_requested_prbs,
    )
    for scenario_name in SCENARIOS:
        run(scenario_name, target_col=target_col, log_target=log_target, loss=loss, peak_weight=peak_weight)
    aggregate(Path("results"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data, run all scenarios, and aggregate report.")
    parser.add_argument("--max-files", type=int, default=int(os.getenv("PREP_MAX_FILES", "240")))
    parser.add_argument("--rows-per-file", type=int, default=int(os.getenv("PREP_ROWS_PER_FILE", "300")))
    parser.add_argument("--feature-col", action="append", dest="feature_cols", default=None)
    parser.add_argument("--target-col", default=DEFAULT_TARGET_COL)
    parser.add_argument("--keep-zero-requested-prbs", action="store_true")
    parser.add_argument("--log-target", action="store_true")
    parser.add_argument("--loss", choices=["mse", "huber", "weighted_huber"], default=None)
    parser.add_argument("--peak-weight", type=float, default=None)
    args = parser.parse_args()
    run_all(
        max_files=args.max_files,
        rows_per_file=args.rows_per_file,
        feature_cols=args.feature_cols or DEFAULT_FEATURES,
        target_col=args.target_col,
        keep_zero_requested_prbs=args.keep_zero_requested_prbs,
        log_target=args.log_target,
        loss=args.loss,
        peak_weight=args.peak_weight,
    )
