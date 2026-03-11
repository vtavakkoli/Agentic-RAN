from __future__ import annotations

import pandas as pd


def _sorted(df: pd.DataFrame, time_col: str = "time_ms") -> pd.DataFrame:
    if time_col in df.columns:
        return df.sort_values(time_col).reset_index(drop=True)
    return df.reset_index(drop=True)


def build_split_metadata(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    time_col: str = "time_ms",
) -> dict:
    n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
    total = max(n_train + n_val + n_test, 1)

    def _time_bounds(df: pd.DataFrame) -> tuple[float | None, float | None]:
        if time_col not in df.columns or df.empty:
            return None, None
        return float(df[time_col].iloc[0]), float(df[time_col].iloc[-1])

    tr_s, tr_e = _time_bounds(train_df)
    va_s, va_e = _time_bounds(val_df)
    te_s, te_e = _time_bounds(test_df)

    return {
        "split_type": "chronological",
        "train_start_index": 0,
        "train_end_index": max(n_train - 1, -1),
        "val_start_index": n_train,
        "val_end_index": n_train + n_val - 1,
        "test_start_index": n_train + n_val,
        "test_end_index": n_train + n_val + n_test - 1,
        "train_start": tr_s,
        "train_end": tr_e,
        "val_start": va_s,
        "val_end": va_e,
        "test_start": te_s,
        "test_end": te_e,
        "train_rows": n_train,
        "val_rows": n_val,
        "test_rows": n_test,
        "train_pct": n_train / total,
        "val_pct": n_val / total,
        "test_pct": n_test / total,
    }


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    *,
    time_col: str = "time_ms",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1")

    ordered = _sorted(df, time_col=time_col)
    n = len(ordered)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train = ordered.iloc[:n_train].copy()
    val = ordered.iloc[n_train : n_train + n_val].copy()
    test = ordered.iloc[n_train + n_val : n_train + n_val + n_test].copy()
    metadata = build_split_metadata(train, val, test, time_col=time_col)
    return train, val, test, metadata


def sort_split(df: pd.DataFrame, time_col: str = "time_ms") -> pd.DataFrame:
    return _sorted(df, time_col=time_col)
