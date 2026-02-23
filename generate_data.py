#!/usr/bin/env python3
"""Generate synthetic multivariate O-RAN-like traffic KPI data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    # Physical/network KPI style features (8)
    "rsrp",  # Reference Signal Received Power
    "rsrq",  # Reference Signal Received Quality
    "sinr",  # Signal to Interference plus Noise Ratio
    "prb_utilization",  # Physical Resource Block usage
    "ue_count",  # active users
    "handover_rate",  # handovers / unit time
    "packet_loss",  # packet drop ratio
    "latency_ms",  # latency proxy
    # Temporal features (4)
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "is_weekend",
    # Context/API style features (4)
    "mobility_index",
    "video_demand",
    "gaming_demand",
    "iot_demand",
]


def generate_timeseries(n_steps: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)

    hour = t % 24
    day_of_week = (t // 24) % 7
    is_weekend = (day_of_week >= 5).astype(float)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    # Daily and weekly traffic cycles
    daily_cycle = 0.6 + 0.3 * (np.sin(2 * np.pi * (hour - 6) / 24) + 1) / 2
    weekly_cycle = 0.85 + 0.2 * (1 - is_weekend) + 0.05 * is_weekend

    mobility_index = np.clip(0.3 + 0.5 * daily_cycle + rng.normal(0, 0.08, n_steps), 0, 1)
    video_demand = np.clip(0.45 + 0.4 * daily_cycle + 0.08 * is_weekend + rng.normal(0, 0.07, n_steps), 0, 1.4)
    gaming_demand = np.clip(0.3 + 0.35 * daily_cycle + 0.15 * (hour >= 18) + rng.normal(0, 0.08, n_steps), 0, 1.4)
    iot_demand = np.clip(0.35 + 0.05 * hour_cos + rng.normal(0, 0.03, n_steps), 0, 1)

    demand_signal = 0.35 * video_demand + 0.25 * gaming_demand + 0.15 * iot_demand + 0.25 * mobility_index

    ue_count = np.clip(40 + 180 * demand_signal * weekly_cycle + rng.normal(0, 8, n_steps), 10, None)
    prb_utilization = np.clip(0.2 + 0.7 * demand_signal + rng.normal(0, 0.05, n_steps), 0.05, 1)
    sinr = np.clip(22 - 9 * prb_utilization - 2 * mobility_index + rng.normal(0, 1.2, n_steps), -5, 30)
    rsrp = np.clip(-96 + 8 * (1 - mobility_index) - 3 * prb_utilization + rng.normal(0, 2.0, n_steps), -125, -70)
    rsrq = np.clip(-13 + 2.5 * (1 - prb_utilization) + rng.normal(0, 1.0, n_steps), -20, -3)
    handover_rate = np.clip(0.05 + 0.7 * mobility_index + rng.normal(0, 0.04, n_steps), 0.01, 1.3)
    packet_loss = np.clip(0.002 + 0.06 * prb_utilization + 0.02 * np.maximum(0, 0.5 - sinr / 20) + rng.normal(0, 0.002, n_steps), 0, 0.2)
    latency_ms = np.clip(8 + 45 * prb_utilization + 15 * packet_loss + rng.normal(0, 2.5, n_steps), 3, 120)

    # Target traffic load with autocorrelation and non-linear interactions
    traffic_load = np.zeros(n_steps)
    for i in range(n_steps):
        base = (
            0.30 * prb_utilization[i]
            + 0.15 * (ue_count[i] / 220)
            + 0.15 * video_demand[i]
            + 0.12 * gaming_demand[i]
            + 0.08 * iot_demand[i]
            + 0.08 * (1 - np.clip((sinr[i] + 5) / 35, 0, 1))
            + 0.12 * daily_cycle[i]
        )
        if i == 0:
            traffic_load[i] = base + rng.normal(0, 0.03)
        else:
            traffic_load[i] = 0.68 * traffic_load[i - 1] + 0.32 * base + rng.normal(0, 0.02)

    traffic_load = np.clip(traffic_load, 0, 1.2)

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n_steps, freq="H"),
            "rsrp": rsrp,
            "rsrq": rsrq,
            "sinr": sinr,
            "prb_utilization": prb_utilization,
            "ue_count": ue_count,
            "handover_rate": handover_rate,
            "packet_loss": packet_loss,
            "latency_ms": latency_ms,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "mobility_index": mobility_index,
            "video_demand": video_demand,
            "gaming_demand": gaming_demand,
            "iot_demand": iot_demand,
            "traffic_load": traffic_load,
        }
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic O-RAN traffic time series data")
    parser.add_argument("--steps", type=int, default=5000, help="Number of time steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        type=str,
        default="shared_data/traffic_data.csv",
        help="Path to output CSV",
    )
    args = parser.parse_args()

    df = generate_timeseries(args.steps, args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows to {out_path}")
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    main()
