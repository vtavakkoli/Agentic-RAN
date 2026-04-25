from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_dataset(shared_data_dir: Path, max_rows: int = 4000) -> tuple[pd.DataFrame, dict]:
    csv_files = sorted(shared_data_dir.rglob("*.csv")) if shared_data_dir.exists() else []

    if not csv_files:
        rng = np.random.default_rng(42)
        n_rows = 2500
        n_features = 10
        x = rng.normal(size=(n_rows, n_features))
        coeffs = np.array([0.8, -1.1, 0.4, 0.6, -0.2, 1.3, -0.5, 0.7, 0.9, -0.3])
        noise = 0.15 * rng.normal(size=n_rows)
        y = x @ coeffs + 0.6 * np.sin(x[:, 0] * x[:, 1]) + noise
        cols = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(x, columns=cols)
        df["target"] = y
        return df, {"source": "synthetic", "files_used": [], "rows": int(df.shape[0])}

    frames = []
    rows_per = max(200, max_rows // min(len(csv_files), 8))
    used_files = []
    for file in csv_files[:8]:
        try:
            part = pd.read_csv(file).select_dtypes(include=["number"]).dropna()
            if part.empty:
                continue
            part = part.head(rows_per)
            frames.append(part)
            used_files.append(str(file))
        except Exception:
            continue

    if not frames:
        return load_dataset(Path("/nonexistent"), max_rows=max_rows)

    df = pd.concat(frames, axis=0, ignore_index=True)
    df = df.loc[:, df.nunique() > 1]
    if df.shape[1] < 2:
        return load_dataset(Path("/nonexistent"), max_rows=max_rows)

    if "target" not in df.columns:
        target_col = df.columns[-1]
        df = df.rename(columns={target_col: "target"})

    summary = {"source": "shared_data", "files_used": used_files, "rows": int(df.shape[0]), "columns": list(df.columns)}
    return df, summary


def write_data_summary(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
