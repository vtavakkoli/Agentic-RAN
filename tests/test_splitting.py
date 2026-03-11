from __future__ import annotations

import pandas as pd

from oran_sim.splitting import chronological_split


def test_chronological_split_order_and_counts() -> None:
    df = pd.DataFrame({"time_ms": [5, 1, 4, 2, 3], "target": [50, 10, 40, 20, 30]})
    train, val, test, meta = chronological_split(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)

    assert train["time_ms"].tolist() == [1, 2, 3]
    assert val["time_ms"].tolist() == [4]
    assert test["time_ms"].tolist() == [5]
    assert meta["train_rows"] == 3
    assert meta["val_rows"] == 1
    assert meta["test_rows"] == 1


def test_chronological_split_no_overlap() -> None:
    df = pd.DataFrame({"time_ms": list(range(10)), "target": list(range(10))})
    train, val, test, meta = chronological_split(df)
    assert train["time_ms"].max() < val["time_ms"].min()
    assert val["time_ms"].max() < test["time_ms"].min()
    assert meta["train_end_index"] < meta["val_start_index"]
    assert meta["val_end_index"] < meta["test_start_index"]
