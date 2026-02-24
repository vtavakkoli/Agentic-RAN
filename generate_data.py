#!/usr/bin/env python3
from __future__ import annotations

import argparse

from oran_sim.data import generate_timeseries, load_timeseries_from_kpm, save_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic O-RAN traffic time series OR build it from O-RAN KPM dataset folders."
    )

    # Keep your requested CLI shape
    parser.add_argument("--step", type=int, default=5000, help="Number of rows to generate/load (max).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (synthetic only).")
    parser.add_argument("--input", type=str, default=None, help="Path to dataset-kpm root folder (real data mode).")
    parser.add_argument("--output", type=str, default="shared_data/traffic_data.csv")

    args = parser.parse_args()

    if args.input:
        df = load_timeseries_from_kpm(args.input, n_steps=args.step)
        mode = f"real dataset from {args.input}"
    else:
        df = generate_timeseries(args.step, args.seed)
        mode = "synthetic"

    out = save_dataframe(df, args.output)
    print(f"Generated {len(df)} rows ({mode}) to {out}")


if __name__ == "__main__":
    main()
