from __future__ import annotations

import numpy as np
import pandas as pd


def sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    if "time_ms" in df.columns:
        return df.sort_values("time_ms").reset_index(drop=True)
    return df.reset_index(drop=True)


def make_sequences(x: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")
    if len(x) < seq_len:
        raise ValueError(f"Not enough rows to build sequences: rows={len(x)} seq_len={seq_len}")

    xs = []
    ys = []
    for end_idx in range(seq_len - 1, len(x)):
        start = end_idx - seq_len + 1
        xs.append(x[start : end_idx + 1])
        ys.append(y[end_idx])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)
