from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_flow_latency_jitter_features() -> None:
    flow_path = Path(__file__).parent / "fixtures" / "ue_001010123456002" / "mgen.csv"
    df = pd.read_csv(flow_path)

    latency_ms = (df["Received Time"] - df["Sent Time"]) * 1000.0
    jitter_ms = latency_ms.diff().abs()

    assert abs(float(latency_ms.iloc[0]) - 22.234) < 1e-3
    assert pd.isna(jitter_ms.iloc[0]) or float(jitter_ms.iloc[0]) == 0.0

    expected_jitter_1 = abs(float(latency_ms.iloc[1]) - float(latency_ms.iloc[0]))
    assert abs(float(jitter_ms.iloc[1]) - expected_jitter_1) < 1e-9
