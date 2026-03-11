from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from oran_sim.config import FEATURE_ORDER
from oran_sim.feature_selection import rank_features_by_importance, select_top_k_features


def _toy_df(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    noise = rng.normal(scale=0.05, size=n)
    y = 10.0 * x1 + 0.1 * x2 + noise

    data = {c: rng.normal(size=n) for c in FEATURE_ORDER if c != "scheduling_policy"}
    data["dl_cqi"] = x1
    data["ul_sinr"] = x2
    data["scheduling_policy"] = ["rr", "pf"] * (n // 2) + (["rr"] if n % 2 else [])
    data["target"] = y
    return pd.DataFrame(data)


def test_top_k_feature_selection_uses_importance_ranking() -> None:
    df = _toy_df()
    ranked = rank_features_by_importance(df, FEATURE_ORDER, random_state=42)
    top2 = select_top_k_features(ranked, 2)
    assert "dl_cqi" in top2


def test_feature_importance_file_generation_and_selected_features_in_status(tmp_path) -> None:
    csv_path = tmp_path / "traffic_data_lightweight-32.csv"
    df = _toy_df(90)
    df.to_csv(csv_path, index=False)
    df.iloc[:54].to_csv(tmp_path / "traffic_data_lightweight-32_train.csv", index=False)
    df.iloc[54:81].to_csv(tmp_path / "traffic_data_lightweight-32_val.csv", index=False)
    df.iloc[81:].to_csv(tmp_path / "traffic_data_lightweight-32_test.csv", index=False)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_scenario",
            "--scenario",
            "lightweight-32",
            "--dataset",
            str(csv_path),
        ],
        check=True,
    )

    sdir = Path("results/scenarios/lightweight-32")
    status = json.loads((sdir / "status.json").read_text(encoding="utf-8"))
    assert status["success"] is True
    assert len(status.get("selected_features", [])) == 10
    assert status.get("model_backend") == "Ridge"
    assert status.get("logical_profile") == "tabular_baseline"
    assert (sdir / "model" / "feature_importance.json").exists()
    assert Path("results/feature_importance.json").exists()
