from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd

from oran_sim.config import FEATURE_ORDER


def test_train_respects_feature_count(tmp_path) -> None:
    n = 60
    rng = np.random.default_rng(0)
    data = {c: rng.normal(size=n) for c in FEATURE_ORDER if c != "scheduling_policy"}
    data["scheduling_policy"] = ["rr", "pf", "mt"] * (n // 3) + ["rr"] * (n % 3)
    data["target"] = rng.normal(size=n)

    csv_path = tmp_path / "toy.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)

    out_dir = tmp_path / "model"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.train",
            "--csv",
            str(csv_path),
            "--out_dir",
            str(out_dir),
            "--model",
            "ridge",
            "--feature_count",
            "10",
            "--epochs",
            "20",
        ],
        check=True,
    )

    features = json.loads((out_dir / "features.json").read_text(encoding="utf-8"))
    cfg = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))

    assert len(features) == 10
    assert features == FEATURE_ORDER[:10]
    assert cfg["epochs"] == 1
    assert cfg["requested_epochs"] == 20
