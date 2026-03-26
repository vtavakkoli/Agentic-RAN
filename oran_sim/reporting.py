from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def write_markdown_report(df: pd.DataFrame, output_md: str) -> Path:
    out = Path(output_md)
    out.parent.mkdir(parents=True, exist_ok=True)

    top_acc = df.sort_values("r2", ascending=False)[["model_type", "r2", "mae", "rmse", "params"]].head(5)
    top_eff = df.sort_values("efficiency_E", ascending=True)[["model_type", "efficiency_E", "c_norm", "p_norm"]].head(5)

    md = [
        "# NAS Evaluation Report",
        "",
        "## Top-5 by R² (higher is better)",
        top_acc.to_markdown(index=False),
        "",
        "## Top-5 by Efficiency E (lower is better)",
        top_eff.to_markdown(index=False),
        "",
        "## Full Metrics",
        df[["model_type", "mae", "rmse", "mape", "r2", "params", "model_size_mb", "general_complexity", "efficiency_E"]].to_markdown(index=False),
        "",
        "## Charts",
        "- ![R2 vs Complexity](./charts/r2_vs_complexity.png)",
        "- ![Efficiency Ranking](./charts/efficiency_ranking.png)",
    ]

    out.write_text("\n".join(md), encoding="utf-8")
    return out


def create_charts(df: pd.DataFrame, chart_dir: str) -> None:
    charts = Path(chart_dir)
    charts.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(9, 5))
    scatter = sns.scatterplot(data=df, x="general_complexity", y="r2", size="params", hue="model_type", legend=False)
    scatter.set_title("Model Accuracy vs. Complexity")
    scatter.set_xlabel("General Complexity")
    scatter.set_ylabel("R²")
    for _, row in df.iterrows():
        scatter.text(row["general_complexity"], row["r2"], row["model_type"], fontsize=8)
    plt.tight_layout()
    plt.savefig(charts / "r2_vs_complexity.png", dpi=200)
    plt.close()

    ranked = df.sort_values("efficiency_E", ascending=True)
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=ranked, x="model_type", y="efficiency_E", palette="viridis")
    ax.set_title("Normalized Efficiency Ranking (Lower is Better)")
    ax.set_xlabel("Model Scenario")
    ax.set_ylabel("Efficiency E")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(charts / "efficiency_ranking.png", dpi=200)
    plt.close()
