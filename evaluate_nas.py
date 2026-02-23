#!/usr/bin/env python3
"""Post-process NAS results and compute normalized efficiency scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def minmax_norm(arr: np.ndarray) -> np.ndarray:
    mn, mx = float(arr.min()), float(arr.max())
    if abs(mx - mn) < 1e-12:
        return np.ones_like(arr)
    return (arr - mn) / (mx - mn)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NAS model efficiency")
    parser.add_argument("--input", type=str, default="shared_data/results.csv")
    parser.add_argument("--output", type=str, default="shared_data/nas_efficiency.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if df.empty:
        raise ValueError("Input results.csv is empty.")

    # C_norm = C_model / max(C_model)
    c_model = df["general_complexity"].values.astype(float)
    c_norm = c_model / max(c_model)

    # P_norm derived from R^2. Higher is better -> min-max normalization in [0,1].
    # Stabilized to avoid divide-by-zero in E.
    p_norm = np.clip(minmax_norm(df["r2"].values.astype(float)), 1e-6, None)

    # E = C_norm / P_norm (lower is better: less complexity per normalized performance unit).
    efficiency = c_norm / p_norm

    out = df.copy()
    out["c_norm"] = c_norm
    out["p_norm"] = p_norm
    out["efficiency_E"] = efficiency
    out = out.sort_values("efficiency_E", ascending=True).reset_index(drop=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    print("NAS Efficiency ranking (lower E is better):")
    print(out[["model_type", "r2", "general_complexity", "c_norm", "p_norm", "efficiency_E"]])
    print(f"Saved evaluation to {output}")


if __name__ == "__main__":
    main()
