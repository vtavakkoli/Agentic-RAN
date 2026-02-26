from __future__ import annotations

import base64
import io
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oran_sim.config import FEATURE_ORDER, supported_scenarios


HIGHER_IS_BETTER = {"R2_test", "R2_val"}


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_true - y_pred
    denom = np.clip(np.abs(y_true), 1e-6, None)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(err) / denom) * 100.0)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def _metric_value(row: dict, primary_metric: str) -> float | None:
    value = row.get(primary_metric)
    if value is None or pd.isna(value):
        return None
    return float(value)


def _is_better(current: float, best: float | None, primary_metric: str) -> bool:
    if best is None:
        return True
    if primary_metric in HIGHER_IS_BETTER:
        return current > best
    return current < best


def _build_timeseries_chart(preds: pd.DataFrame, scenario: str) -> str:
    fig, ax = plt.subplots(figsize=(9, 3))
    x = preds["time_ms"] if "time_ms" in preds.columns else preds.index
    ax.plot(x.values, preds["y_true"].values, label="y_true", linewidth=1.6)
    ax.plot(x.values, preds["y_pred"].values, label="y_pred", linewidth=1.2)
    ax.set_title(f"{scenario}: y_true/y_pred vs timestamp")
    ax.set_xlabel("timestamp")
    ax.legend(loc="best")
    img = _fig_to_base64(fig)
    plt.close(fig)
    return img


def _build_naive_chart(preds: pd.DataFrame, scenario: str) -> tuple[str, dict]:
    y_true = preds["y_true"].to_numpy(dtype=float)
    y_naive = np.roll(y_true, 1)
    y_naive[0] = y_true[0]
    naive_m = _metrics(y_true, y_naive)

    fig, ax = plt.subplots(figsize=(9, 3))
    x = preds["time_ms"] if "time_ms" in preds.columns else preds.index
    ax.plot(x.values, y_true, label="y_true", linewidth=1.6)
    ax.plot(x.values, y_naive, label="y_pred_naive", linewidth=1.2)
    ax.set_title(f"{scenario}: naive baseline vs timestamp")
    ax.set_xlabel("timestamp")
    ax.legend(loc="best")
    img = _fig_to_base64(fig)
    plt.close(fig)
    return img, naive_m


def _dataset_summary(dataset_path: Path, scenario: str) -> tuple[pd.DataFrame, str]:
    if not dataset_path.exists():
        return pd.DataFrame(), "<p>Dataset not found for this scenario.</p>"

    df = pd.read_csv(dataset_path)
    feature_cols = [c for c in FEATURE_ORDER if c in df.columns]
    if not feature_cols:
        return pd.DataFrame(), "<p>No known feature columns available.</p>"

    stats = df[feature_cols].describe().T[["mean", "std", "min", "max"]].reset_index().rename(columns={"index": "feature"})
    summary = pd.DataFrame(
        [
            {
                "scenario": scenario,
                "samples": int(len(df)),
                "num_features": int(len(feature_cols)),
                "feature_names": ", ".join(feature_cols),
            }
        ]
    )
    combined = summary.merge(stats, how="cross")

    violin_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])][: min(8, len(feature_cols))]
    if not violin_features:
        return combined, "<p>No numeric features available for violin plots.</p>"

    data = [df[c].dropna().to_numpy() for c in violin_features]
    fig, ax = plt.subplots(figsize=(max(10, len(violin_features) * 1.3), 3.8))
    ax.violinplot(data, showmeans=True, showmedians=True)
    ax.set_xticks(range(1, len(violin_features) + 1))
    ax.set_xticklabels(violin_features, rotation=30, ha="right")
    ax.set_title(f"{scenario}: feature distributions (violin)")
    img = _fig_to_base64(fig)
    plt.close(fig)
    return combined, f"<img src='data:image/png;base64,{img}'/>"


HIGHER_IS_BETTER = {"R2_test", "R2_val"}


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _metric_value(row: dict, primary_metric: str) -> float | None:
    value = row.get(primary_metric)
    if value is None or pd.isna(value):
        return None
    return float(value)


def _is_better(current: float, best: float | None, primary_metric: str) -> bool:
    if best is None:
        return True
    if primary_metric in HIGHER_IS_BETTER:
        return current > best
    return current < best


def main() -> None:
    scenarios = supported_scenarios()
    status_paths = [Path("results/scenarios") / s / "status.json" for s in scenarios]

    print("[aggregator] waiting for scenario statuses", flush=True)
    deadline = time.time() + 60 * 60
    while time.time() < deadline:
        if all(p.exists() for p in status_paths):
            break
        time.sleep(5)

    out_dir = Path("results/final")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    model_chart_sections: list[str] = []
    naive_chart_sections: list[str] = []
    dataset_table_sections: list[str] = []
    violin_sections: list[str] = []

    primary_metric = os.getenv("PRIMARY_METRIC", "R2_test")

    for scenario in scenarios:
        sdir = Path("results/scenarios") / scenario
        status_path = sdir / "status.json"
        status = {"scenario_name": scenario, "success": False, "error": "missing status"}
        if status_path.exists():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except Exception:
                status = {"scenario_name": scenario, "success": False, "error": "invalid status"}

        row = {
            "scenario": scenario,
            "success": bool(status.get("success", False)),
            "error": status.get("error", ""),
            "metrics_path": status.get("metrics_path", str(sdir / "model" / "metrics.json")),
            "preds_path": status.get("preds_path", str(sdir / "preds.csv")),
            "dataset_path": status.get("dataset_path", ""),
            "epochs": status.get("epochs"),
        }

        metrics_path = Path(row["metrics_path"])
        preds_path = Path(row["preds_path"])
        cfg_path = metrics_path.parent / "config.json"
        epoch_metrics_path = metrics_path.parent / "epoch_metrics.csv"

        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            row["model_type"] = cfg.get("model_type")
            row["num_features"] = len(cfg.get("features", []))
            row["epochs"] = cfg.get("epochs", row.get("epochs"))
        else:
            row["model_type"] = None

        if epoch_metrics_path.exists():
            epoch_df = pd.read_csv(epoch_metrics_path)
            row["epoch_metrics_path"] = str(epoch_metrics_path)
            row["epochs_logged"] = int(len(epoch_df))

        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            test_metrics = metrics.get("test", {})
            val_metrics = metrics.get("val", {})
            for key in ["MAE", "RMSE", "MAPE", "R2"]:
                row[f"{key}_test"] = test_metrics.get(key)
                row[f"{key}_val"] = val_metrics.get(key)

        if preds_path.exists():
            preds = pd.read_csv(preds_path)
            row["rows"] = int(len(preds))
            row["mean_abs_error"] = float(preds["abs_error"].mean())
            row["mean_abs_pct_error"] = float(preds["pct_error"].abs().mean())

            model_chart = _build_timeseries_chart(preds, scenario)
            model_chart_sections.append(f"<h3>{scenario}</h3><img src='data:image/png;base64,{model_chart}'/>")

            naive_chart, naive_m = _build_naive_chart(preds, scenario)
            row["naive_MAE"] = naive_m["MAE"]
            row["naive_RMSE"] = naive_m["RMSE"]
            row["naive_MAPE"] = naive_m["MAPE"]
            row["naive_R2"] = naive_m["R2"]
            row["model_beats_naive_MAE"] = row.get("MAE_test", np.inf) < naive_m["MAE"]
            naive_chart_sections.append(
                f"<h3>{scenario}</h3><p>Naive metrics: {json.dumps(naive_m)}</p><img src='data:image/png;base64,{naive_chart}'/>"
            )

        dpath_str = str(row.get("dataset_path", "")).strip()
        if dpath_str:
            stats_df, violin_html = _dataset_summary(Path(dpath_str), scenario)
            if not stats_df.empty:
                dataset_table_sections.append(f"<h3>{scenario}</h3>{stats_df.to_html(index=False)}")
            violin_sections.append(f"<h3>{scenario}</h3>{violin_html}")

        rows.append(row)

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(out_dir / "scenario_status.csv", index=False)

    best_scenario = None
    best_metric_value = None
    for row in rows:
        v = _metric_value(row, primary_metric)
        if v is None:
            continue
        if _is_better(v, best_metric_value, primary_metric):
            best_metric_value = v
            best_scenario = row["scenario"]

    table_cols = [
        "scenario",
        "success",
        "model_type",
        "epochs",
        "epochs_logged",
        "rows",
        "num_features",
        "MAE_test",
        "RMSE_test",
        "MAPE_test",
        "R2_test",
        "naive_MAE",
        "naive_RMSE",
        "naive_MAPE",
        "naive_R2",
        "model_beats_naive_MAE",
        "mean_abs_error",
        "mean_abs_pct_error",
        "error",
    ]
    present_cols = [c for c in table_cols if c in comp_df.columns]
    table_df = comp_df[present_cols].copy() if present_cols else comp_df.copy()

    if best_scenario and not table_df.empty and "scenario" in table_df.columns:
        table_df["best"] = table_df["scenario"].apply(lambda s: "⭐" if s == best_scenario else "")

    best_text = (
        f"Best scenario by {primary_metric}: <strong>{best_scenario}</strong> ({best_metric_value:.6f})"
        if best_scenario is not None
        else f"Best scenario by {primary_metric}: unavailable (no valid metric values found)."
    )

    html = f"""
    <html><body>
    <h1>KPM Final Report</h1>
    <h2>Scenario Comparison</h2>
    <p>{best_text}</p>
    {table_df.to_html(index=False)}
    <h2>Model Predictions vs Ground Truth (timestamp axis)</h2>
    {''.join(model_chart_sections) if model_chart_sections else '<p>No model charts available.</p>'}
    <h2>Naive Baseline Comparison</h2>
    {''.join(naive_chart_sections) if naive_chart_sections else '<p>No naive baseline charts available.</p>'}
    <h2>Dataset/Feature Statistics</h2>
    {''.join(dataset_table_sections) if dataset_table_sections else '<p>No dataset stats available.</p>'}
    <h2>Feature Violin Plots</h2>
    {''.join(violin_sections) if violin_sections else '<p>No violin plots available.</p>'}
    </body></html>
    """

    (out_dir / "report.html").write_text(html, encoding="utf-8")
    print("[aggregator] final report generated", flush=True)


if __name__ == "__main__":
    main()
