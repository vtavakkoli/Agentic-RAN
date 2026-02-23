#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from oran_sim.reporting import create_charts, write_markdown_report


def minmax_norm(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    return np.ones_like(arr) if abs(hi - lo) < 1e-12 else (arr - lo) / (hi - lo)


def main() -> None:
    p = argparse.ArgumentParser(description="Compute NAS efficiency report")
    p.add_argument("--input", default="shared_data/results.csv")
    p.add_argument("--output", default="shared_data/nas_efficiency.csv")
    p.add_argument("--report", default="shared_data/nas_report.md")
    p.add_argument("--chart_dir", default="shared_data/charts")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    if df.empty:
        raise ValueError("Input results CSV is empty")

    c_model = df["general_complexity"].astype(float).values
    c_norm = c_model / max(c_model)
    p_norm = np.clip(minmax_norm(df["r2"].astype(float).values), 1e-6, None)
    efficiency = c_norm / p_norm

    out = df.copy()
    out["c_norm"] = c_norm
    out["p_norm"] = p_norm
    out["efficiency_E"] = efficiency
    out = out.sort_values(["efficiency_E", "rmse"], ascending=[True, True]).reset_index(drop=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    create_charts(out, args.chart_dir)
    report = write_markdown_report(out, args.report)

    print("\n=== NAS SUMMARY TABLE ===")
    print(out[["model_type", "r2", "mae", "rmse", "params", "general_complexity", "efficiency_E"]].to_string(index=False))
    print(f"\nSaved CSV: {output}")
    print(f"Saved report: {report}")
    print(f"Saved charts: {args.chart_dir}")


if __name__ == "__main__":
    main()
