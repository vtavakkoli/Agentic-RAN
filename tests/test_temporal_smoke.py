from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd

from oran_sim.config import FEATURE_ORDER


def test_temporal_scenario_train_predict_smoke(tmp_path) -> None:
    n = 120
    rng = np.random.default_rng(123)
    data = {c: rng.normal(size=n) for c in FEATURE_ORDER if c != "scheduling_policy"}
    data["scheduling_policy"] = rng.choice(["rr", "pf"], size=n)
    data["time_ms"] = np.arange(n)
    data["target"] = 0.7 * data["dl_cqi"] + 0.2 * data["ul_sinr"] + rng.normal(scale=0.05, size=n)
    df = pd.DataFrame(data)

    csv = tmp_path / "toy.csv"
    df.to_csv(csv, index=False)
    df.iloc[:72].to_csv(tmp_path / "toy_train.csv", index=False)
    df.iloc[72:108].to_csv(tmp_path / "toy_val.csv", index=False)
    df.iloc[108:].to_csv(tmp_path / "toy_test.csv", index=False)

    out_dir = tmp_path / "model"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.train",
            "--csv",
            str(csv),
            "--out_dir",
            str(out_dir),
            "--model",
            "attention-baseline",
            "--epochs",
            "1",
            "--feature_count",
            "8",
        ],
        check=True,
    )

    cfg = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    assert cfg["temporal"] is True
    assert cfg["seq_len"] <= 16
    assert cfg["seq_len"] >= 2

    pred_path = tmp_path / "preds.csv"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.predict",
            "--model_dir",
            str(out_dir),
            "--csv",
            str(tmp_path / "toy_test.csv"),
            "--output",
            str(pred_path),
        ],
        check=True,
    )
    preds = pd.read_csv(pred_path)
    assert not preds.empty
