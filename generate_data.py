#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from oran_sim.config import FEATURE_ORDER
from oran_sim.data import load_timeseries_from_kpm, write_json


def _build_target(df: pd.DataFrame, target: str, horizon_steps: int) -> pd.DataFrame:
    horizon = max(1, horizon_steps)
    shifted = df["traffic_load"].shift(-horizon)
    if target == "traffic_load":
        df["target"] = shifted
    else:
        df["target"] = shifted - df["traffic_load"]
    return df.dropna(subset=["target"]).reset_index(drop=True)


def _expand_to_exact_rows(df: pd.DataFrame, steps: int, seed: int) -> tuple[pd.DataFrame, List[str]]:
    logs: List[str] = []
    if len(df) >= steps:
        logs.append(f"strategy=truncate first {steps} rows from {len(df)}")
        return df.iloc[:steps].copy(), logs

    rng = np.random.default_rng(seed)
    parts = [df.copy()]
    remaining = steps - len(df)
    logs.append(f"strategy=resample because available_rows={len(df)} < required_rows={steps}")

    grouped = [g for _, g in df.groupby("reservation", sort=True)]
    if not grouped:
        sampled = df.sample(n=remaining, replace=True, random_state=seed)
        parts.append(sampled)
    else:
        idx = 0
        while remaining > 0:
            g = grouped[idx % len(grouped)]
            take = min(len(g), remaining)
            pick = g.iloc[rng.choice(len(g), size=take, replace=len(g) < take)]
            parts.append(pick)
            logs.append(f"used reservation={g['reservation'].iloc[0]} sampled_rows={take}")
            remaining -= take
            idx += 1

    expanded = pd.concat(parts, ignore_index=True)
    return expanded.iloc[:steps].copy(), logs


def _det_split(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    n_train = int(n * 0.6)
    n_val = int(n * 0.3)
    n_test = n - n_train - n_val
    train = shuffled.iloc[:n_train].copy()
    val = shuffled.iloc[n_train : n_train + n_val].copy()
    test = shuffled.iloc[n_train + n_val : n_train + n_val + n_test].copy()
    return train, val, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate unified KPM dataset with deterministic splits")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", choices=["traffic_load", "delta_traffic_load"], default="traffic_load")
    parser.add_argument("--horizon_steps", type=int, default=1)
    args = parser.parse_args()

    base = load_timeseries_from_kpm(args.input, verbose=True)
    base = _build_target(base, args.target, args.horizon_steps)

    keep_cols = ["time_ms", "reservation", "traffic_load"] + FEATURE_ORDER + ["target"]
    for c in keep_cols:
        if c not in base.columns:
            base[c] = 0
    base = base[keep_cols]

    exact_df, sampling_logs = _expand_to_exact_rows(base, args.steps, args.seed)
    train_df, val_df, test_df = _det_split(exact_df, args.seed)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    exact_df.to_csv(output, index=False)
    train_path = output.with_name(f"{output.stem}_train.csv")
    val_path = output.with_name(f"{output.stem}_val.csv")
    test_path = output.with_name(f"{output.stem}_test.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    summary = {
        "total_rows": int(len(exact_df)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "time_range": {
            "min_time_ms": int(exact_df["time_ms"].min()) if len(exact_df) else None,
            "max_time_ms": int(exact_df["time_ms"].max()) if len(exact_df) else None,
        },
        "missing_value_counts": {k: int(v) for k, v in exact_df.isna().sum().to_dict().items()},
        "features_present": [c for c in FEATURE_ORDER if c in exact_df.columns],
        "target_definition": {"name": args.target, "horizon_steps": int(args.horizon_steps)},
        "sampling_logs": sampling_logs,
    }
    summary_path = Path("results/data/traffic_data_summary.json")
    write_json(summary_path, summary)

    print(f"rows: total={len(exact_df)} train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"time_range_ms: {summary['time_range']}")
    print(f"summary_json: {summary_path}")
    for line in sampling_logs:
        print(f"sampling: {line}")


if __name__ == "__main__":
    main()
