#!/usr/bin/env python3
from __future__ import annotations

import argparse

from oran_sim.data import generate_timeseries, save_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic O-RAN traffic time series")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="shared_data/traffic_data.csv")
    args = parser.parse_args()

    df = generate_timeseries(args.steps, args.seed)
    out = save_dataframe(df, args.output)
    print(f"Generated {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
