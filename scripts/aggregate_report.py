from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from agentic_ran.scenarios import SCENARIOS


METRIC_COLS = ["r2", "rmse", "mae", "mape", "smape", "wmape", "composite_score", "action_accuracy", "action_macro_f1"]
PEAK_METRIC_COLS = ["peak_mae", "normal_mae", "peak_rmse", "peak_r2"]
HIGHER_IS_BETTER = {"r2", "composite_score", "peak_r2"}


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
    return {metric: (metric, expr) for metric in [*METRIC_COLS, *PEAK_METRIC_COLS]}


def _to_html_table(df: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
    if columns is not None:
        df = df.loc[:, list(columns)]
    return df.to_html(index=False, escape=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "")


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
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _plot_residual_vs_mlp(success_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = success_df[success_df["scenario"].str.contains("residual|lightweight|balanced", regex=True)].copy()
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values("r2", ascending=False)

    metrics = ["r2", "rmse", "mae", "wmape"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.bar(plot_df["scenario"], plot_df[metric], color=["#1d4ed8" if "residual" in s else "#64748b" for s in plot_df["scenario"]])
        ax.set_title(metric.upper())
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.suptitle("Residual and temporal models vs MLP baselines")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _plot_base_case_temperature_by_policy(results_root: Path, output_path: Path) -> bool:
    seed_files = sorted((results_root / "policies").glob("seed_*/predictions.csv"))
    frames: list[pd.DataFrame] = []
    for file in seed_files:
        try:
            df = pd.read_csv(file)
        except Exception:
            continue
        required = {"scenario", "policy_name", "temperature_c"}
        if not required.issubset(set(df.columns)):
            continue
        base = df.loc[df["scenario"] == "base_case"].copy()
        if base.empty:
            continue
        base["step"] = range(len(base))
        frames.append(base)
    if not frames:
        return False

    data = pd.concat(frames, ignore_index=True)
    grouped = data.groupby(["policy_name", "step"], as_index=False)["temperature_c"].mean()
    if grouped.empty:
        return False

    fig, ax = plt.subplots(figsize=(11, 5))
    for policy in sorted(grouped["policy_name"].unique().tolist()):
        part = grouped[grouped["policy_name"] == policy]
        ax.plot(part["step"], part["temperature_c"], label=policy, linewidth=1.6)
    ax.set_title("Base-case temperature trajectory by policy")
    ax.set_xlabel("Step")
    ax.set_ylabel("Temperature proxy (°C)")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return True


def aggregate(results_root: Path = Path("results")) -> Path:
    results_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for name, scenario in SCENARIOS.items():
        folder = results_root / name
        metrics = _read_json(folder / "metrics.json")
        metadata = _read_json(folder / "model_metadata.json")
        status = _read_json(folder / "status.json") or {}
        data_summary = _read_json(folder / "data_summary.json") or {}
        agentic_summary = _read_json(folder / "agentic_summary.json") or {}

        source_files = data_summary.get("source_files_used", [])
        ok = bool(metrics and metadata and status.get("status") == "success")
        row = {
            "scenario": name,
            "scenario_type": scenario.logical_profile,
            "status": "success" if ok else "failure",
            "model_type": (metadata or {}).get("model_type", scenario.model_type),
            "backend": (metadata or {}).get("backend", scenario.backend),
            "logical_profile": (metadata or {}).get("logical_profile", scenario.logical_profile),
            "sequence_length": (metadata or {}).get("sequence_length", scenario.sequence_length),
            "residual": int("residual" in str((metadata or {}).get("model_type", scenario.model_type))),
            "temporal": int((metadata or {}).get("sequence_length", scenario.sequence_length) > 1),
            "epochs": (metadata or {}).get("epochs", "n/a"),
            "dataset_rows": (metadata or {}).get("dataset_rows", "n/a"),
            "num_features": (metadata or {}).get("num_features", "n/a"),
            "target_column": (metadata or {}).get("target_column", data_summary.get("target_column", "n/a")),
            "log_target": (metadata or {}).get("log_target", data_summary.get("log_target", False)),
            "rows_train": data_summary.get("rows_per_split", {}).get("train", "n/a"),
            "rows_val": data_summary.get("rows_per_split", {}).get("val", "n/a"),
            "rows_test": data_summary.get("rows_per_split", {}).get("test", "n/a"),
            "metrics_file_count": data_summary.get("num_metrics_files_used", data_summary.get("metrics_file_count", "n/a")),
            "source_root": data_summary.get("source_root", "dataset"),
            "source_files_first5": ", ".join(source_files[:5]),
            "avg_decision_confidence": agentic_summary.get("average_confidence", float("nan")),
            "notes_or_errors": status.get("error") or "",
        }
        for m in [*METRIC_COLS, *PEAK_METRIC_COLS]:
            row[m] = metrics.get(m) if metrics else None
        rows.append(row)

    df = pd.DataFrame(rows)
    for col in ["dataset_rows", "epochs", "rows_train", "rows_val", "rows_test", "metrics_file_count", *METRIC_COLS, *PEAK_METRIC_COLS]:
        df[col] = _coerce_numeric(df[col])

    leaderboard = df[df["status"] == "success"].sort_values("composite_score", ascending=False).copy()
    best = leaderboard.iloc[0]["scenario"] if not leaderboard.empty else "n/a"
    score_mean = leaderboard["composite_score"].mean() if not leaderboard.empty else float("nan")

    group_success = leaderboard.groupby("scenario_type", as_index=False).agg(
        scenario_count=("scenario", "count"),
        **_metric_agg("mean"),
        dataset_rows=("dataset_rows", "sum"),
    )

    figures_dir = results_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = figures_dir / "scenario_type_metric_heatmap.png"
    residual_vs_mlp_path = figures_dir / "residual_vs_mlp.png"
    base_case_temperature_path = figures_dir / "base_case_temperature_trajectory_by_policy.png"
    _plot_group_metric_heatmap(group_success, heatmap_path)
    _plot_residual_vs_mlp(leaderboard, residual_vs_mlp_path)
    has_temperature_plot = _plot_base_case_temperature_by_policy(results_root, base_case_temperature_path)

    model_family_comparison = leaderboard[
        ["scenario", "model_type", "sequence_length", "residual", "temporal", "r2", "rmse", "mae", "smape", "wmape", "composite_score", "action_accuracy", "action_macro_f1"]
    ].sort_values("composite_score", ascending=False)

    peak_eval = leaderboard[["scenario", "peak_mae", "normal_mae", "peak_rmse", "peak_r2"]].sort_values("peak_mae")

    def _best(metric: str, lower: bool = False) -> str:
        if leaderboard.empty or leaderboard[metric].dropna().empty:
            return "n/a"
        ranked = leaderboard.sort_values(metric, ascending=lower)
        return f"{ranked.iloc[0]['scenario']} ({ranked.iloc[0][metric]:.4f})"

    conclusion = (
        "<p><strong>Best by R2:</strong> " + _best("r2", lower=False) + "</p>"
        "<p><strong>Best by RMSE:</strong> " + _best("rmse", lower=True) + "</p>"
        "<p><strong>Best by MAE:</strong> " + _best("mae", lower=True) + "</p>"
        "<p><strong>Best by wMAPE:</strong> " + _best("wmape", lower=True) + "</p>"
        "<p><strong>Best by Peak MAE:</strong> " + _best("peak_mae", lower=True) + "</p>"
        "<p>Composite score remains useful, but these per-metric winners should drive deployment choices.</p>"
    )

    prediction_frames = []
    for scenario_name in leaderboard["scenario"].tolist():
        pred_path = results_root / scenario_name / "predictions.csv"
        if pred_path.exists():
            prediction_frames.append(pd.read_csv(pred_path))
    if prediction_frames:
        ref = prediction_frames[0][["sample_id", "y_true"]].reset_index(drop=True)
        for cur in prediction_frames[1:]:
            chk = cur[["sample_id", "y_true"]].reset_index(drop=True)
            if not ref.equals(chk):
                raise ValueError("Prediction plots cannot be compared because test samples differ.")

    drl_metrics_path = results_root / "tables" / "drl_seed_metrics.csv"
    drl_df = pd.read_csv(drl_metrics_path) if drl_metrics_path.exists() else pd.DataFrame()

    html = [
        "<html><head><meta charset='utf-8'><title>Final Benchmark Report</title>",
        "<style>body{font-family:Inter,Segoe UI,Arial,sans-serif;margin:28px;background:#f1f5f9;color:#0f172a;line-height:1.4;}"
        ".container{max-width:1280px;margin:0 auto;}h1,h2{color:#0b3a75;margin-top:22px;}"
        "table{border-collapse:collapse;width:100%;margin:10px 0 22px 0;background:white;}th,td{border:1px solid #cbd5e1;padding:8px 9px;font-size:13px;}"
        "th{background:#dbeafe;color:#1e3a8a;} .section{background:white;padding:14px 18px;border-radius:12px;border:1px solid #cbd5e1;margin-bottom:14px;}"
        ".figure img{width:100%;max-width:1100px;height:auto;border-radius:8px;}</style></head><body>",
        "<div class='container'><h1>Final Benchmark Report</h1>",
        "<div class='section'><h2>Executive summary</h2><p>This report benchmarks forecasting and agentic DRL control for slice-aware RAN scheduling/resource actions across eMBB, MTC, and URLLC.</p></div>",
        f"<p><strong>Best scenario by composite:</strong> {best} | <strong>Average composite:</strong> {score_mean:.4f}</p>",
        "<div class='section'><h2>Model family comparison</h2>",
        _to_html_table(model_family_comparison),
        "</div>",
        "<div class='section'><h2>Scenario-level detailed comparison</h2>",
        _to_html_table(df[["scenario", "status", "model_type", "sequence_length", *METRIC_COLS, *PEAK_METRIC_COLS]]),
        "</div>",
        "<div class='section'><h2>Dataset/source summary</h2>",
        _to_html_table(df[["scenario", "metrics_file_count", "source_files_first5", "source_root", "rows_train", "rows_val", "rows_test"]]),
        "</div>",
        "<div class='section'><h2>Time-aware feature section</h2><p>Timestamp is parsed with pandas datetime conversion, sorted chronologically, and transformed into hour/day cyclic signals, elapsed seconds, experiment_second, and time-index features.</p></div>",
        "<div class='section'><h2>Traffic-aware feature section</h2><p>UE identifiers are mapped to eMBB/MTC/URLLC classes, slice semantics are inferred (slice 0/1/2), and match/mismatch signals are generated for traffic-vs-slice consistency.</p></div>",
        "<div class='section'><h2>Peak evaluation table</h2>",
        _to_html_table(peak_eval),
        "</div>",
        "<div class='section'><h2>DRL policy leaderboard</h2>",
        _to_html_table(drl_df) if not drl_df.empty else "<p>No DRL policy metrics found.</p>",
        "</div>",
        "<div class='section'><h2>Agentic decision section</h2>",
        _to_html_table(df[["scenario", "action_accuracy", "action_macro_f1", "avg_decision_confidence"]]),
        "</div>",
        "<div class='section'><h2>Ablation table</h2>",
        _to_html_table(
            leaderboard[leaderboard["scenario"].isin(["lightweight-32", "with_time_features", "with_time_and_traffic_features", "residual-mlp-128", "agentic_residual_mlp", "agentic_liquid_residual"])][
                ["scenario", "model_type", "r2", "rmse", "mae", "wmape", "action_accuracy", "action_macro_f1", "composite_score"]
            ]
        ),
        "</div>",
        "<div class='section'><h2>Residual and temporal models vs MLP baselines</h2>",
        f"<div class='figure'><img src='figures/{residual_vs_mlp_path.name}' alt='Residual and temporal models vs MLP baselines'></div>",
        "</div>",
        "<div class='section'><h2>Scenario-type heatmap</h2>",
        f"<div class='figure'><img src='figures/{heatmap_path.name}' alt='Scenario type metric heatmap'></div>",
        "</div>",
        "<div class='section'><h2>Agent benchmark: temperature trajectory by policy (base_case)</h2>",
        (
            f"<div class='figure'><img src='figures/{base_case_temperature_path.name}' alt='Base case temperature trajectory by policy'></div>"
            if has_temperature_plot
            else "<p>No base_case policy-temperature trajectory data found.</p>"
        ),
        "</div>",
        "<div class='section'><h2>Scientific conclusion</h2>",
        conclusion,
        "</div></div></body></html>",
    ]

    prediction_section = ["<div class='section'><h2>Scenario prediction plots</h2>"]
    successful_scenarios = leaderboard["scenario"].tolist() if not leaderboard.empty else []
    if successful_scenarios:
        prediction_section.append(
            "<p>Predictions vs. ground truth for each successful scenario, useful for qualitative model-behavior inspection.</p>"
        )
        prediction_section.append("<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px;'>")
        for scenario_name in successful_scenarios:
            pred_rel = f"{scenario_name}/plots/predictions_vs_truth.png"
            pred_abs = results_root / pred_rel
            prediction_section.append("<div style='border:1px solid #cbd5e1;border-radius:10px;padding:10px;background:#fff;'>")
            prediction_section.append(f"<h3 style='margin:4px 0 8px 0'>{scenario_name}</h3>")
            if pred_abs.exists():
                prediction_section.append(
                    f"<img src='{pred_rel}' alt='{scenario_name} predictions vs truth' "
                    "style='width:100%;height:auto;border-radius:8px;'>"
                )
            else:
                prediction_section.append(f"<p>Missing plot: {pred_rel}</p>")
            prediction_section.append("</div>")
        prediction_section.append("</div>")
    else:
        prediction_section.append("<p>No successful scenarios were found, so prediction plots are unavailable.</p>")
    prediction_section.append("</div>")

    html.insert(
        -1,
        "<div class='section'><h2>Scientific wording note</h2><p>The previous action-decision metrics were pseudo-label based and are not sufficient to prove real control quality. The updated benchmark evaluates agentic RAN control through offline DRL reward, slice-specific operational KPIs, and policy behavior.</p></div>",
    )
    html.insert(-1, "\n".join(prediction_section))

    report_path = results_root / "report.html"
    report_path.write_text("\n".join(html), encoding="utf-8")
    return report_path


if __name__ == "__main__":
    path = aggregate()
    print(f"Wrote {path}")
