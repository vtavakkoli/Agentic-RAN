from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd

FIX = Path(__file__).parent / "fixtures"


def _copy_reservation(root: Path, reservation: str) -> None:
    exp_root = root / "cluster_1" / "slicing_1" / "scheduling_0" / reservation
    (exp_root / "bs").mkdir(parents=True, exist_ok=True)
    (exp_root / "ue_001010123456002").mkdir(parents=True, exist_ok=True)

    shutil.copy(FIX / "bs" / "1010123456002_metrics.csv", exp_root / "bs" / "1010123456002_metrics.csv")
    shutil.copy(FIX / "bs" / "enb_metrics.csv", exp_root / "enb_metrics.csv")
    shutil.copy(FIX / "bs" / "enb_metrics.csv", exp_root / "bs" / "enb_metrics.csv")
    shutil.copy(FIX / "ue_001010123456002" / "ue_metrics.csv", exp_root / "ue_001010123456002" / "ue_metrics.csv")
    shutil.copy(FIX / "ue_001010123456002" / "mgen.csv", exp_root / "ue_001010123456002" / "flow_log.csv")


def _build_kpm_tree(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "shared_data" / "dataset-kpm"
    _copy_reservation(dataset_root, "RESERVATION-142634")
    return dataset_root


def test_generate_data_exact_steps_and_splits(tmp_path: Path) -> None:
    input_root = _build_kpm_tree(tmp_path)
    out_csv = tmp_path / "shared_data" / "traffic_data.csv"

    cmd = [
        "python",
        "generate_data.py",
        "--steps",
        "50",
        "--input",
        str(input_root),
        "--output",
        str(out_csv),
        "--seed",
        "42",
    ]
    subprocess.run(cmd, check=True)

    full = pd.read_csv(out_csv)
    train = pd.read_csv(tmp_path / "shared_data" / "traffic_data_train.csv")
    val = pd.read_csv(tmp_path / "shared_data" / "traffic_data_val.csv")
    test = pd.read_csv(tmp_path / "shared_data" / "traffic_data_test.csv")

    assert len(full) == 50
    assert len(train) == 30
    assert len(val) == 15
    assert len(test) == 5

    first5 = full.head(5).copy()
    subprocess.run(cmd, check=True)
    full_again = pd.read_csv(out_csv)
    pd.testing.assert_frame_equal(first5.reset_index(drop=True), full_again.head(5).reset_index(drop=True))


def test_generate_data_stops_fetching_after_requested_steps(tmp_path: Path) -> None:
    input_root = tmp_path / "shared_data" / "dataset-kpm"
    _copy_reservation(input_root, "RESERVATION-142634")
    _copy_reservation(input_root, "RESERVATION-142635")

    out_csv = tmp_path / "shared_data" / "traffic_data.csv"
    cmd = [
        "python",
        "generate_data.py",
        "--steps",
        "3",
        "--input",
        str(input_root),
        "--output",
        str(out_csv),
        "--seed",
        "42",
    ]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)

    full = pd.read_csv(out_csv)
    assert len(full) == 3

    data_lines = [line for line in completed.stdout.splitlines() if line.startswith("[DATA] reservation=")]
    assert len(data_lines) == 1
    assert "reservation=RESERVATION-142634" in data_lines[0]
    assert "used_rows=4" in data_lines[0]
