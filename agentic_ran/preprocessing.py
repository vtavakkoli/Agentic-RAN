from __future__ import annotations

import numpy as np
import pandas as pd


def _make_sequences(x: np.ndarray, y: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    if sequence_length <= 1:
        return x, y
    if len(x) <= sequence_length:
        return np.empty((0, sequence_length, x.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=sequence_length, axis=0)[:-1]
    return np.ascontiguousarray(windows, dtype=np.float32), np.ascontiguousarray(y[sequence_length:], dtype=np.float32)


def build_features(df: pd.DataFrame, sequence_length: int) -> tuple[np.ndarray, np.ndarray, dict]:
    num_df = df.select_dtypes(include=["number"]).dropna().copy()
    if "target" not in num_df.columns:
        num_df = num_df.rename(columns={num_df.columns[-1]: "target"})

    feature_cols = [c for c in num_df.columns if c != "target"]
    x = num_df[feature_cols].to_numpy(dtype=np.float32)
    y = num_df["target"].to_numpy(dtype=np.float32)

    x_mean = x.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True) + 1e-6
    x_scaled = (x - x_mean) / x_std

    x_final, y_final = _make_sequences(x_scaled, y, sequence_length)

    metadata = {
        "dataset_rows": int(num_df.shape[0]),
        "num_features": len(feature_cols),
        "selected_features": feature_cols,
        "sequence_length": sequence_length,
    }
    return x_final, y_final, metadata


def split_dataset(x: np.ndarray, y: np.ndarray, train_ratio: float = 0.6, val_ratio: float = 0.1):
    n = len(x)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return (
        (x[:train_end], y[:train_end]),
        (x[train_end:val_end], y[train_end:val_end]),
        (x[val_end:], y[val_end:]),
    )


def build_features_for_pre_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sequence_length: int,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], dict]:
    train_num = train_df.select_dtypes(include=["number"]).dropna().copy()
    val_num = val_df.select_dtypes(include=["number"]).dropna().copy()
    test_num = test_df.select_dtypes(include=["number"]).dropna().copy()

    if "target" not in train_num.columns:
        train_num = train_num.rename(columns={train_num.columns[-1]: "target"})
    if "target" not in val_num.columns:
        val_num = val_num.rename(columns={val_num.columns[-1]: "target"})
    if "target" not in test_num.columns:
        test_num = test_num.rename(columns={test_num.columns[-1]: "target"})

    feature_cols = [c for c in train_num.columns if c != "target"]
    val_num = val_num.reindex(columns=feature_cols + ["target"], fill_value=0.0)
    test_num = test_num.reindex(columns=feature_cols + ["target"], fill_value=0.0)

    x_train = train_num[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_num["target"].to_numpy(dtype=np.float32)
    x_val = val_num[feature_cols].to_numpy(dtype=np.float32)
    y_val = val_num["target"].to_numpy(dtype=np.float32)
    x_test = test_num[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_num["target"].to_numpy(dtype=np.float32)

    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True) + 1e-6

    def _scale(x: np.ndarray) -> np.ndarray:
        return (x - x_mean) / x_std

    x_train = _scale(x_train)
    x_val = _scale(x_val)
    x_test = _scale(x_test)

    def _to_sequences(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _make_sequences(x, y, sequence_length)

    train_set = _to_sequences(x_train, y_train)
    val_set = _to_sequences(x_val, y_val)
    test_set = _to_sequences(x_test, y_test)

    metadata = {
        "dataset_rows": int(train_num.shape[0] + val_num.shape[0] + test_num.shape[0]),
        "num_features": len(feature_cols),
        "selected_features": feature_cols,
        "sequence_length": sequence_length,
        "preset_split": {
            "train": int(train_num.shape[0]),
            "val": int(val_num.shape[0]),
            "test": int(test_num.shape[0]),
        },
    }
    return train_set, val_set, test_set, metadata
