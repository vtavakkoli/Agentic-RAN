#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from oran_sim.config import FEATURE_ORDER
from oran_sim.data import load_timeseries_from_kpm, write_json
from oran_sim.splitting import chronological_split


def _is_madrid_zone_layout(root: Path) -> bool:
    freq_dirs = [p for p in root.glob("f*") if p.is_dir()]
    if not freq_dirs:
        return False
    return any(list(fd.glob("downlink_*.csv")) for fd in freq_dirs)


def _load_madrid_zone_wide(root: Path) -> pd.DataFrame:
    freq_dirs = sorted([p for p in root.glob("f*") if p.is_dir()])
    rows = []
    per_freq = {}

    for freq_dir in freq_dirs:
        freq = freq_dir.name
        dl_path = next(iter(sorted(freq_dir.glob("downlink_*.csv"))), None)
        ul_path = next(iter(sorted(freq_dir.glob("uplink_*.csv"))), None)
        users_path = next(iter(sorted(freq_dir.glob("users_*.csv"))), None)
        if dl_path is None:
            continue

        dl = pd.read_csv(dl_path)
        dl["second"] = np.floor(pd.to_numeric(dl["timestamp"], errors="coerce")).astype("Int64")
        dl["tbs_sum"] = pd.to_numeric(dl["tbs_sum"], errors="coerce")
        dl = dl.dropna(subset=["second"]).groupby("second", as_index=False)["tbs_sum"].sum()
        dl = dl.rename(columns={"tbs_sum": f"downlink_{freq}"})

        ul = pd.DataFrame(columns=["second", f"uplink_{freq}"])
        if ul_path is not None:
            ul_tmp = pd.read_csv(ul_path)
            ul_tmp["second"] = np.floor(pd.to_numeric(ul_tmp["timestamp"], errors="coerce")).astype("Int64")
            ul_tmp["tbs_sum"] = pd.to_numeric(ul_tmp["tbs_sum"], errors="coerce")
            ul_tmp = ul_tmp.dropna(subset=["second"]).groupby("second", as_index=False)["tbs_sum"].sum()
            ul = ul_tmp.rename(columns={"tbs_sum": f"uplink_{freq}"})

        users = pd.DataFrame(columns=["second", f"users_{freq}"])
        if users_path is not None:
            user_tmp = pd.read_csv(users_path)
            user_tmp["second"] = np.floor(pd.to_numeric(user_tmp["timestamp"], errors="coerce")).astype("Int64")
            user_tmp["user_unique"] = pd.to_numeric(user_tmp["user_unique"], errors="coerce")
            user_tmp = user_tmp.dropna(subset=["second"]).groupby("second", as_index=False)["user_unique"].mean()
            users = user_tmp.rename(columns={"user_unique": f"users_{freq}"})

        merged = dl.merge(ul, on="second", how="outer").merge(users, on="second", how="outer").sort_values("second")
        per_freq[freq] = merged
        rows.extend(merged["second"].dropna().astype(int).tolist())

    if not per_freq:
        raise RuntimeError(f"No usable frequency data found in {root}")

    all_seconds = pd.DataFrame({"second": sorted(set(rows))})
    base = all_seconds.copy()
    for freq in sorted(per_freq.keys()):
        base = base.merge(per_freq[freq], on="second", how="left")

    for c in base.columns:
        if c != "second":
            base[c] = pd.to_numeric(base[c], errors="coerce")
    base = base.sort_values("second").reset_index(drop=True)
    feature_cols = [c for c in base.columns if c != "second"]
    base[feature_cols] = base[feature_cols].ffill().fillna(0.0)

    down_cols = [c for c in base.columns if c.startswith("downlink_f")]
    up_cols = [c for c in base.columns if c.startswith("uplink_f")]
    user_cols = [c for c in base.columns if c.startswith("users_f")]

    base["timestamp"] = base["second"].astype(float)
    base["time_ms"] = (base["second"].astype(float) * 1000.0).astype("int64")
    base["traffic_load"] = base[down_cols].sum(axis=1) if down_cols else 0.0
    base["num_ues"] = base[user_cols].sum(axis=1) if user_cols else 0.0
    base["ul_buffer_bytes"] = base[up_cols].sum(axis=1) if up_cols else 0.0
    base["dl_buffer_bytes"] = base["traffic_load"]
    base["scheduling_policy"] = root.name
    base["reservation"] = root.name

    return base


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




def main() -> None:
    parser = argparse.ArgumentParser(description="Generate unified KPM dataset with deterministic splits")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", choices=["traffic_load", "delta_traffic_load"], default="traffic_load")
    parser.add_argument("--horizon_steps", type=int, default=1)
    args = parser.parse_args()

    required_rows = args.steps + max(1, int(args.horizon_steps))
    input_path = Path(args.input)
    if _is_madrid_zone_layout(input_path):
        base = _load_madrid_zone_wide(input_path)
    else:
        base = load_timeseries_from_kpm(args.input, n_steps=required_rows, verbose=True)
    base = _build_target(base, args.target, args.horizon_steps)

    keep_cols = ["timestamp", "time_ms", "reservation", "traffic_load"]
    keep_cols += sorted([c for c in base.columns if c.startswith("downlink_f") or c.startswith("uplink_f") or c.startswith("users_f")])
    keep_cols += FEATURE_ORDER + ["target"]
    for c in keep_cols:
        if c not in base.columns:
            base[c] = 0
    base = base[keep_cols]

    exact_df, sampling_logs = _expand_to_exact_rows(base, args.steps, args.seed)
    train_df, val_df, test_df, split_meta = chronological_split(exact_df, train_ratio=0.6, val_ratio=0.3, test_ratio=0.1)

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
        "split_metadata": split_meta,
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
