from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from oran_sim.data import load_timeseries_from_kpm


ZONE_I = Path("dataset/madrid-lte-dataset/zoneI")


def test_load_timeseries_from_madrid_zone_i() -> None:
    df = load_timeseries_from_kpm(ZONE_I, n_steps=64, verbose=False)
    assert len(df) == 64
    assert "traffic_load" in df.columns
    assert "num_ues" in df.columns
    assert set(df["reservation"].unique()).issubset({"f796", "f1815", "f2650"})
    assert df["traffic_load"].astype(float).sum() > 0
    assert df["num_ues"].astype(float).max() > 0


def test_generate_data_with_madrid_zone_i(tmp_path: Path) -> None:
    out_csv = tmp_path / "traffic_data.csv"
    subprocess.run(
        [
            sys.executable,
            "generate_data.py",
            "--steps",
            "100",
            "--input",
            str(ZONE_I),
            "--output",
            str(out_csv),
            "--seed",
            "42",
        ],
        check=True,
    )

    df = pd.read_csv(out_csv)
    assert len(df) == 100
    assert {"traffic_load", "num_ues", "target"}.issubset(df.columns)
    assert df["traffic_load"].sum() > 0
