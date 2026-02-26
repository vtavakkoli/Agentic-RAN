from __future__ import annotations

import base64
import io
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from oran_sim.config import supported_scenarios


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
    charts: list[tuple[str, str]] = []

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
        }

        metrics_path = Path(row["metrics_path"])
        preds_path = Path(row["preds_path"])
        cfg_path = metrics_path.parent / "config.json"

        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                row["model_type"] = cfg.get("model_type")
            except Exception:
                row["model_type"] = None
        else:
            row["model_type"] = None

        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                test_metrics = metrics.get("test", {})
                val_metrics = metrics.get("val", {})
                for key in ["MAE", "RMSE", "MAPE", "R2"]:
                    row[f"{key}_test"] = test_metrics.get(key)
                    row[f"{key}_val"] = val_metrics.get(key)
            except Exception:
                pass

        if preds_path.exists():
            try:
                preds = pd.read_csv(preds_path)
                row["rows"] = int(len(preds))
                row["mean_abs_error"] = float(preds["abs_error"].mean())
                row["mean_abs_pct_error"] = float(preds["pct_error"].abs().mean())

                fig, ax = plt.subplots(figsize=(8, 3))
                x = preds["time_ms"] if "time_ms" in preds.columns else preds.index
                ax.plot(x.values[:300], preds["y_true"].values[:300], label="y_true", linewidth=1.5)
                ax.plot(x.values[:300], preds["y_pred"].values[:300], label="y_pred", linewidth=1.2)
                ax.set_title(f"{scenario}: prediction vs ground truth")
                ax.legend(loc="best")
                charts.append((scenario, _fig_to_base64(fig)))
                plt.close(fig)
            except Exception as exc:
                row["error"] = (str(row.get("error", "")) + f"; chart_error={exc}").strip("; ")

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
        "rows",
        "MAE_test",
        "RMSE_test",
        "MAPE_test",
        "R2_test",
        "mean_abs_error",
        "mean_abs_pct_error",
        "error",
    ]
    present_cols = [c for c in table_cols if c in comp_df.columns]
    table_df = comp_df[present_cols].copy() if present_cols else comp_df.copy()

    if best_scenario and not table_df.empty and "scenario" in table_df.columns:
        table_df["best"] = table_df["scenario"].apply(lambda s: "⭐" if s == best_scenario else "")

    chart_sections = "\n".join(
        f"<h3>{scenario}</h3><img src='data:image/png;base64,{img}'/>" for scenario, img in charts
    )
    if not chart_sections:
        chart_sections = "<p>No prediction charts available.</p>"

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
    <h2>Per-Scenario Prediction vs Ground Truth</h2>
    {chart_sections}
    </body></html>
    """

    (out_dir / "report.html").write_text(html, encoding="utf-8")
    print("[aggregator] final report generated", flush=True)


if __name__ == "__main__":
    main()
