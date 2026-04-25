from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from agentic_ran.scenarios import SCENARIOS


METRIC_COLS = ["r2", "rmse", "mae", "mape", "smape", "wmape", "composite_score"]
HIGHER_IS_BETTER = {"r2", "composite_score"}


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _metric_agg(expr: str) -> dict:
    return {metric: (metric, expr) for metric in METRIC_COLS}


def _plot_group_metric_heatmap(group_summary: pd.DataFrame, output_path: Path) -> None:
    plot_df = group_summary.set_index("scenario_type")[METRIC_COLS]
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 4 + 0.55 * len(plot_df)))
    im = ax.imshow(plot_df.values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(METRIC_COLS)))
    ax.set_xticklabels(METRIC_COLS, rotation=30, ha="right")
    ax.set_yticks(range(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index)
    ax.set_title("Average Metric Performance by Scenario Type")
    for i in range(plot_df.shape[0]):
        for j in range(plot_df.shape[1]):
            value = plot_df.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8, color="#0f172a")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _plot_group_composite(group_summary: pd.DataFrame, output_path: Path) -> None:
    plot_df = group_summary.sort_values("composite_score", ascending=False)
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4 + 0.5 * len(plot_df)))
    bars = ax.barh(plot_df["scenario_type"], plot_df["composite_score"], color="#1d4ed8")
    ax.invert_yaxis()
    ax.set_xlabel("Average Composite Score")
    ax.set_title("Cumulative Composite Score by Scenario Type")
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height() / 2, f"{width:.3f}", va="center", fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _plot_metric_rankings(success_df: pd.DataFrame, output_path: Path) -> None:
    if success_df.empty:
        return
    rankings = {}
    for metric in METRIC_COLS:
        ascending = metric not in HIGHER_IS_BETTER
        winner_idx = success_df[metric].sort_values(ascending=ascending).index[0]
        rankings[metric] = success_df.loc[winner_idx, "scenario"]
    winners = pd.Series(rankings).value_counts()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(winners.index, winners.values, color="#0f766e")
    ax.set_ylabel("Number of Metrics Won")
    ax.set_title("Metric Leader Distribution Across Scenarios")
    ax.set_xticklabels(winners.index, rotation=20, ha="right")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{int(bar.get_height())}", ha="center")
    ax.set_ylim(0, max(winners.values) + 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _to_html_table(df: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
    if columns is not None:
        df = df.loc[:, list(columns)]
    return df.to_html(index=False, escape=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "")


def aggregate(results_root: Path = Path("results")) -> Path:
    results_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for name, scenario in SCENARIOS.items():
        folder = results_root / name
        metrics = _read_json(folder / "metrics.json")
        metadata = _read_json(folder / "model_metadata.json")
        status = _read_json(folder / "status.json") or {}
        data_summary = _read_json(folder / "data_summary.json") or {}

        ok = bool(metrics and metadata and status.get("status") == "success")
        row = {
            "scenario": name,
            "scenario_type": scenario.logical_profile,
            "status": "success" if ok else "failure",
            "model_type": (metadata or {}).get("model_type", scenario.model_type),
            "backend": (metadata or {}).get("backend", scenario.backend),
            "logical_profile": (metadata or {}).get("logical_profile", scenario.logical_profile),
            "sequence_length": (metadata or {}).get("sequence_length", scenario.sequence_length),
            "epochs": (metadata or {}).get("epochs", "n/a"),
            "dataset_rows": (metadata or {}).get("dataset_rows", "n/a"),
            "num_features": (metadata or {}).get("num_features", "n/a"),
            "selected_features": ", ".join((metadata or {}).get("selected_features", [])),
            "target_column": (metadata or {}).get("target_column", data_summary.get("target_column", "n/a")),
            "log_target": (metadata or {}).get("log_target", data_summary.get("log_target", False)),
            "source_files_used": ", ".join(data_summary.get("source_files_used", data_summary.get("files_used", []))),
            "rows_train": data_summary.get("rows_per_split", {}).get("train", "n/a"),
            "rows_val": data_summary.get("rows_per_split", {}).get("val", "n/a"),
            "rows_test": data_summary.get("rows_per_split", {}).get("test", "n/a"),
            "metrics_file_count": data_summary.get("num_metrics_files_used", data_summary.get("metrics_file_count", "n/a")),
            "notes_or_errors": status.get("error") or "",
        }
        for m in METRIC_COLS:
            row[m] = metrics.get(m) if metrics else None
        rows.append(row)

    df = pd.DataFrame(rows)
    for col in ["dataset_rows", "epochs", "rows_train", "rows_val", "rows_test", "metrics_file_count", *METRIC_COLS]:
        df[col] = _coerce_numeric(df[col])

    leaderboard = df[df["status"] == "success"].sort_values("composite_score", ascending=False).copy()
    best = leaderboard.iloc[0]["scenario"] if not leaderboard.empty else "n/a"
    score_mean = leaderboard["composite_score"].mean() if not leaderboard.empty else float("nan")
    score_std = leaderboard["composite_score"].std() if len(leaderboard) > 1 else float("nan")
    best_r2 = float(leaderboard.iloc[0]["r2"]) if (not leaderboard.empty and pd.notna(leaderboard.iloc[0]["r2"])) else float("nan")

    group_success = leaderboard.groupby("scenario_type", as_index=False).agg(
        scenario_count=("scenario", "count"),
        **_metric_agg("mean"),
        dataset_rows=("dataset_rows", "sum"),
    )
    group_success = group_success.sort_values("composite_score", ascending=False)

    status_by_group = (
        df.groupby("scenario_type", as_index=False)
        .agg(total_scenarios=("scenario", "count"), successful_runs=("status", lambda x: int((x == "success").sum())))
        .assign(success_rate=lambda x: (100 * x["successful_runs"] / x["total_scenarios"]))
    )
    cumulative_metrics = leaderboard.agg({m: "mean" for m in METRIC_COLS}).to_frame(name="overall_average").T

    figures_dir = results_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = figures_dir / "scenario_type_metric_heatmap.png"
    composite_path = figures_dir / "scenario_type_composite.png"
    ranking_path = figures_dir / "metric_leaders.png"
    _plot_group_metric_heatmap(group_success, heatmap_path)
    _plot_group_composite(group_success, composite_path)
    _plot_metric_rankings(leaderboard, ranking_path)

    score_std_display = f"{score_std:.3f}" if pd.notna(score_std) else "n/a"

    if pd.notna(best_r2) and best_r2 < 0:
        conclusion = (
            f"<p>The top-ranked scenario by composite score is <strong>{best}</strong>, but its R2 is negative ({best_r2:.3f}). "
            "This is not yet satisfactory and requires better feature/target handling before claiming strong predictive quality.</p>"
            "<p>At scenario-type granularity, the grouped tables still provide useful relative comparisons across model families.</p>"
        )
    else:
        conclusion = (
            f"<p>Based on cumulative composite ranking, <strong>{best}</strong> is the strongest individual scenario in this run. "
            "Interpret this alongside per-metric tables because different baselines may outperform on specific metrics.</p>"
            "<p>At scenario-type granularity, the grouped tables and charts show how robust each design family is across metrics.</p>"
        )

    summary = (
        "<p>This benchmark compares the <strong>Liquid Dynamics</strong> scenario family against lightweight MLP, balanced MLP, "
        "deep MLP, ultra-performance MLP, attention-based sequence modeling, and xLSTM baselines. Metrics are evaluated "
        "on a held-out test set. R2 and composite score are higher-is-better; RMSE/MAE/MAPE/sMAPE/wMAPE are lower-is-better.</p>"
    )

    html = [
        "<html><head><meta charset='utf-8'><title>Final Benchmark Report</title>",
        "<style>body{font-family:Inter,Segoe UI,Arial,sans-serif;margin:28px;background:#f1f5f9;color:#0f172a;line-height:1.4;}"
        ".container{max-width:1280px;margin:0 auto;}h1,h2{color:#0b3a75;margin-top:22px;}"
        ".cards{display:flex;gap:14px;flex-wrap:wrap;margin:16px 0 20px 0;}"
        ".card{background:white;border:1px solid #cbd5e1;border-radius:10px;padding:14px 16px;min-width:220px;box-shadow:0 1px 4px rgba(15,23,42,.08);}"
        ".card .label{font-size:12px;color:#475569;text-transform:uppercase;letter-spacing:.04em;}"
        ".card .value{font-size:22px;font-weight:700;color:#0b3a75;margin-top:6px;}"
        "table{border-collapse:collapse;width:100%;margin:10px 0 22px 0;background:white;box-shadow:0 1px 4px rgba(15,23,42,.08);}"
        "th,td{border:1px solid #cbd5e1;padding:8px 9px;font-size:13px;vertical-align:top;}"
        "th{background:#dbeafe;color:#1e3a8a;} .section{background:white;padding:14px 18px;border-radius:12px;border:1px solid #cbd5e1;margin-bottom:14px;}"
        ".figure-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px;margin:10px 0 20px;}"
        ".figure{background:white;border:1px solid #cbd5e1;border-radius:12px;padding:10px;box-shadow:0 1px 4px rgba(15,23,42,.08);}"
        ".figure img{width:100%;height:auto;border-radius:8px;} .caption{font-size:12px;color:#475569;margin-top:6px;}"
        ".prediction-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px;margin-top:10px;}"
        ".prediction-card{background:white;border:1px solid #cbd5e1;border-radius:12px;padding:10px;box-shadow:0 1px 4px rgba(15,23,42,.08);}"
        ".prediction-card img{width:100%;height:auto;border-radius:8px;}"
        "</style></head><body>",
        "<div class='container'><h1>Final Benchmark Report</h1>",
        summary,
        "<div class='cards'>",
        f"<div class='card'><div class='label'>Best Scenario</div><div class='value'>{best}</div></div>",
        f"<div class='card'><div class='label'>Successful Runs</div><div class='value'>{len(leaderboard)}/{len(df)}</div></div>",
        f"<div class='card'><div class='label'>Avg Composite</div><div class='value'>{score_mean:.3f}</div></div>",
        f"<div class='card'><div class='label'>Composite Std. Dev.</div><div class='value'>{score_std_display}</div></div>",
        "</div>",
        "<div class='section'><h2>Evaluation protocol</h2><p>Time-order/precomputed split with 60% train, 10% validation, 30% test.</p></div>",
        "<div class='section'><h2>Cumulative results grouped by scenario type</h2>",
        _to_html_table(group_success, ["scenario_type", "scenario_count", *METRIC_COLS, "dataset_rows"]),
        "<h2>Scenario-type reliability summary</h2>",
        _to_html_table(status_by_group, ["scenario_type", "total_scenarios", "successful_runs", "success_rate"]),
        "<h2>Global cumulative benchmark averages</h2>",
        _to_html_table(cumulative_metrics, METRIC_COLS),
        "</div>",
        "<div class='section'><h2>Scenario-level detailed comparison</h2>",
        _to_html_table(df),
        "<h2>Leaderboard (successful scenarios)</h2>",
        _to_html_table(leaderboard[["scenario", "scenario_type", *METRIC_COLS]]),
        "</div>",
        "<div class='section'><h2>Feature mapping used by each scenario</h2>",
        _to_html_table(
            df[
                [
                    "scenario",
                    "dataset_rows",
                    "num_features",
                    "target_column",
                    "selected_features",
                    "log_target",
                    "rows_train",
                    "rows_val",
                    "rows_test",
                    "metrics_file_count",
                    "source_files_used",
                ]
            ]
        ),
        "</div>",
        "<div class='section'><h2>Comparative visualizations</h2><div class='figure-grid'>",
        f"<div class='figure'><img src='figures/{heatmap_path.name}' alt='Scenario type metric heatmap'><div class='caption'>Average metric profile by scenario type.</div></div>",
        f"<div class='figure'><img src='figures/{composite_path.name}' alt='Scenario type composite scores'><div class='caption'>Cumulative composite scores, ranked.</div></div>",
        f"<div class='figure'><img src='figures/{ranking_path.name}' alt='Metric winner counts'><div class='caption'>How often each scenario wins across all metrics.</div></div>",
        "</div></div>",
        "<div class='section'><h2>Scientific conclusion</h2>",
        conclusion,
        "</div>",
        "<div class='section'><h2>Limitations and reproducibility notes</h2>"
        "<ul>"
        "<li>Composite score provides one ranking view but can hide metric-specific trade-offs.</li>"
        "<li>Results depend on data sampling and split composition; rerun with fixed seeds and the same prepared splits for reproducibility.</li>"
        "<li>Input feature availability can vary by source files, so feature mapping should be reviewed before cross-run comparisons.</li>"
        "</ul></div>",
        "<div class='section'><h2>Scenario prediction plots (top-5 scenarios)</h2><div class='prediction-grid'>",
    ]

    for name in leaderboard["scenario"].tolist()[:5]:
        rel = f"{name}/plots/predictions_vs_truth.png"
        if (results_root / rel).exists():
            html.append(
                f"<div class='prediction-card'>"
                f"<h3>{name}</h3>"
                f"<img src='{rel}' alt='{name} predictions vs truth'>"
                f"<div class='caption'>{rel}</div>"
                f"</div>"
            )
        else:
            html.append(
                f"<div class='prediction-card'>"
                f"<h3>{name}</h3>"
                f"<div class='caption'>Missing plot: {rel}</div>"
                f"</div>"
            )

    html.extend(["</div></div></div></body></html>"])
    report_path = results_root / "report.html"
    report_path.write_text("\n".join(html), encoding="utf-8")
    return report_path


if __name__ == "__main__":
    path = aggregate()
    print(f"Wrote {path}")
