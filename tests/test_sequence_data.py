from __future__ import annotations

import numpy as np
import pandas as pd

from oran_sim.sequence_data import make_sequences, sort_by_time


def test_make_sequences_and_alignment() -> None:
    x = np.arange(20, dtype=float).reshape(10, 2)
    y = np.arange(10, dtype=float)
    xs, ys = make_sequences(x, y, seq_len=4)
    assert xs.shape == (7, 4, 2)
    assert ys.tolist() == [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]


def test_temporal_sort_preserves_time_order() -> None:
    df = pd.DataFrame({"time_ms": [30, 10, 20], "target": [3, 1, 2]})
    out = sort_by_time(df)
    assert out["time_ms"].tolist() == [10, 20, 30]
