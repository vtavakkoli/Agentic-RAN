#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from oran_sim.data import save_dataframe, split_by_time
from oran_sim.seed import SEED, set_global_seed


def _parse_split(raw: str) -> tuple[float, float, float]:
    vals = tuple(float(x.strip()) for x in raw.split(","))
    if len(vals) != 3:
        raise ValueError("--split must contain 3 comma-separated values")
    if abs(sum(vals) - 1.0) > 1e-8:
        raise ValueError("--split values must sum to 1.0")
    return vals


def generate_synthetic_dataset(steps: int, seed: int = SEED, split: tuple[float, float, float] = (0.6, 0.3, 0.1)) -> pd.DataFrame:
    if steps <= 0:
        raise ValueError("steps must be > 0")
    rng = np.random.default_rng(seed)
    set_global_seed(seed)

    time_ms = np.arange(steps, dtype=np.int64) * 100
    t = np.arange(steps, dtype=float)

    regime = np.piecewise(
        t,
        [t < steps * 0.25, (t >= steps * 0.25) & (t < steps * 0.65), t >= steps * 0.65],
        [0.35, 0.7, 0.5],
    )
    wave = 0.12 * np.sin(t / 37.0) + 0.06 * np.sin(t / 9.0)
    noise = rng.normal(0, 0.03, steps)
    traffic_load = np.clip(regime + wave + noise, 0.05, 1.2)

    num_ues = np.clip((8 + traffic_load * 35 + rng.normal(0, 2, steps)).round(), 1, 100)
    dl_cqi = np.clip(5 + traffic_load * 8 + rng.normal(0, 1.2, steps), 1, 15)
    dl_mcs = np.clip(dl_cqi * 1.7 + rng.normal(0, 1.5, steps), 0, 28)
    ul_mcs = np.clip(dl_mcs - rng.normal(1.5, 1.2, steps), 0, 28)
    ul_sinr = np.clip(4 + dl_cqi * 1.2 + rng.normal(0, 1.0, steps), -5, 35)
    ul_rssi = np.clip(-108 + ul_sinr * 0.9 + rng.normal(0, 1.5, steps), -120, -50)

    sum_requested_prbs = np.clip(35 + traffic_load * 210 + rng.normal(0, 12, steps), 10, 275)
    sum_granted_prbs = np.clip(sum_requested_prbs * (0.8 + 0.15 * (dl_cqi / 15)) + rng.normal(0, 6, steps), 5, 275)

    dl_buffer_bytes = np.clip((sum_requested_prbs - sum_granted_prbs + 5) * 420 + rng.normal(0, 900, steps), 0, None)
    ul_buffer_bytes = np.clip(dl_buffer_bytes * 0.65 + rng.normal(0, 700, steps), 0, None)

    tx_errors_dl_pct = np.clip(14 - dl_cqi * 0.7 + rng.normal(0, 0.8, steps), 0, 35)
    rx_errors_ul_pct = np.clip(16 - ul_sinr * 0.4 + rng.normal(0, 0.7, steps), 0, 35)

    slicing_enabled = (t > steps * 0.3).astype(float)
    slice_id = np.where(slicing_enabled > 0, ((t // 700) % 3) + 1, 0)
    slice_prb = np.where(slicing_enabled > 0, 30 + 10 * (slice_id % 3), 0)
    scheduling_policy = np.where((t // 800) % 2 == 0, 0, 1)

    latency_ms = np.clip(40 + dl_buffer_bytes / 1500 + tx_errors_dl_pct * 0.8 + rng.normal(0, 1.5, steps), 5, 400)
    jitter_ms = np.clip(np.abs(np.diff(latency_ms, prepend=latency_ms[0])) + rng.normal(0, 0.5, steps), 0, 80)
    payload_bytes = np.clip(600 + traffic_load * 1600 + rng.normal(0, 120, steps), 64, 9000)

    df = pd.DataFrame(
        {
            "time_ms": time_ms,
            "traffic_load": traffic_load,
            "num_ues": num_ues,
            "dl_mcs": dl_mcs,
            "ul_mcs": ul_mcs,
            "dl_cqi": dl_cqi,
            "ul_sinr": ul_sinr,
            "ul_rssi": ul_rssi,
            "dl_buffer_bytes": dl_buffer_bytes,
            "ul_buffer_bytes": ul_buffer_bytes,
            "sum_requested_prbs": sum_requested_prbs,
            "sum_granted_prbs": sum_granted_prbs,
            "tx_errors_dl_pct": tx_errors_dl_pct,
            "rx_errors_ul_pct": rx_errors_ul_pct,
            "slicing_enabled": slicing_enabled,
            "slice_id": slice_id,
            "slice_prb": slice_prb,
            "scheduling_policy": scheduling_policy,
            "latency_ms": latency_ms,
            "jitter_ms": jitter_ms,
            "payload_bytes": payload_bytes,
        }
    )
    for col in df.columns:
        if col == "time_ms":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    return split_by_time(df, split)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic traffic dataset")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--split", type=str, default="0.6,0.3,0.1")
    args = parser.parse_args()

    split = _parse_split(args.split)
    df = generate_synthetic_dataset(args.steps, seed=args.seed, split=split)
    out = save_dataframe(df, args.output)

    counts = df["split"].value_counts().to_dict()
    print(f"Wrote {len(df)} rows to {Path(out).resolve()}")
    print(f"split_counts={counts}")
    print(
        "traffic_load_stats="
        f"min={df['traffic_load'].min():.4f},max={df['traffic_load'].max():.4f},mean={df['traffic_load'].mean():.4f}"
    )


if __name__ == "__main__":
    main()
