from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd


def _build_kpm_tree(base: Path) -> None:
    for i in range(2):
        res = base / "cluster_a" / "slicing_on_20" / "scheduling_rr" / f"RESERVATION-{i+1}"
        (res / "bs").mkdir(parents=True, exist_ok=True)
        (res / "ue_1").mkdir(parents=True, exist_ok=True)
        (res / "bs" / "enb_metrics.csv").write_text(
            "time,nof_ue,dl_brate\n" + "\n".join([f"{1000+j*1000},{i+1},{10+j}" for j in range(5)]) + "\n",
            encoding="utf-8",
        )
        (res / "bs" / "cell_metrics.csv").write_text(
            "time,tx_brate downlink [Mbps],sum_requested_prbs,sum_granted_prbs\n"
            + "\n".join([f"{1000+j*1000},{10+j},{20+j},{15+j}" for j in range(5)])
            + "\n",
            encoding="utf-8",
        )
        (res / "ue_1" / "ue_metrics.csv").write_text(
            "time;dl_mcs;ul_mcs;dl_cqi;ul_sinr\n"
            + "\n".join([f"{j*1000},{1+j},{2+j},{3+j},{4+j}" for j in range(5)])
            + "\n",
            encoding="utf-8",
        )


def test_generate_exact_n_and_splits(tmp_path: Path) -> None:
    input_dir = tmp_path / "dataset-kpm"
    _build_kpm_tree(input_dir)
    out = tmp_path / "traffic_data.csv"

    subprocess.run(
        [
            "python",
            "generate_data.py",
            "--steps",
            "20",
            "--input",
            str(input_dir),
            "--output",
            str(out),
            "--seed",
            "42",
        ],
        check=True,
    )

    full = pd.read_csv(out)
    train = pd.read_csv(tmp_path / "traffic_data_train.csv")
    val = pd.read_csv(tmp_path / "traffic_data_val.csv")
    test = pd.read_csv(tmp_path / "traffic_data_test.csv")

    assert len(full) == 20
    assert len(train) == 12
    assert len(val) == 6
    assert len(test) == 2


def test_deterministic_split_seed(tmp_path: Path) -> None:
    input_dir = tmp_path / "dataset-kpm"
    _build_kpm_tree(input_dir)

    out1 = tmp_path / "a.csv"
    out2 = tmp_path / "b.csv"
    cmd1 = ["python", "generate_data.py", "--steps", "20", "--input", str(input_dir), "--output", str(out1), "--seed", "7"]
    cmd2 = ["python", "generate_data.py", "--steps", "20", "--input", str(input_dir), "--output", str(out2), "--seed", "7"]
    subprocess.run(cmd1, check=True)
    subprocess.run(cmd2, check=True)

    a_train = pd.read_csv(tmp_path / "a_train.csv")
    b_train = pd.read_csv(tmp_path / "b_train.csv")
    assert a_train.equals(b_train)
