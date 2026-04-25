from __future__ import annotations

import argparse
import os
from pathlib import Path

from agentic_ran.scenarios import SCENARIOS
from scripts.aggregate_report import aggregate
from scripts.prepare_splits import build_dataset, split_and_save
from scripts.run_scenario import run


def prepare_data(max_files: int, rows_per_file: int, max_features: int) -> None:
    input_dirs = [Path("slice_mixed"), Path("slice_traffic")]
    dataset, _ = build_dataset(
        input_dirs=input_dirs,
        max_files=max_files,
        rows_per_file=rows_per_file,
        max_features=max_features,
    )
    split_and_save(dataset, output_dir=Path("shared_data/splits"))


def run_all(max_files: int, rows_per_file: int, max_features: int) -> None:
    prepare_data(max_files=max_files, rows_per_file=rows_per_file, max_features=max_features)
    for scenario_name in SCENARIOS:
        run(scenario_name)
    aggregate(Path("results"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data, run all scenarios, and aggregate report.")
    parser.add_argument("--max-files", type=int, default=int(os.getenv("PREP_MAX_FILES", "240")))
    parser.add_argument("--rows-per-file", type=int, default=int(os.getenv("PREP_ROWS_PER_FILE", "300")))
    parser.add_argument("--max-features", type=int, default=int(os.getenv("PREP_MAX_FEATURES", "10")))
    args = parser.parse_args()
    run_all(max_files=args.max_files, rows_per_file=args.rows_per_file, max_features=args.max_features)
