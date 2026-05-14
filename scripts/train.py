from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pandas as pd

from agentic_ran.data_loading import DEFAULT_TARGET_COL
from agentic_ran.scenarios import SCENARIOS
from scripts.run_scenario import run


ARTIFACT_FILES = [
    "best_model.pt",
    "model_metadata.json",
    "metrics.json",
    "training_log.csv",
    "data_summary.json",
    "agentic_summary.json",
    "agentic_decisions.csv",
    "predictions.csv",
]


def _remove_old_root_model_outputs(results_root: Path) -> None:
    """Remove old /results/<model_name> folders so training artifacts exist only under results/train/artifacts."""
    for scenario_name in SCENARIOS:
        old_dir = results_root / scenario_name
        if old_dir.exists() and old_dir.is_dir():
            shutil.rmtree(old_dir)


def _manifest_for_artifact(scenario_name: str, artifact_dir: Path) -> dict:
    metadata_path = artifact_dir / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    copied = [filename for filename in ARTIFACT_FILES if (artifact_dir / filename).exists()]
    manifest = {
        "scenario": scenario_name,
        "artifact_folder": str(artifact_dir),
        "stored_under_train_artifacts_only": True,
        "files": copied,
        "model_type": metadata.get("model_type"),
        "sequence_length": metadata.get("sequence_length"),
        "is_agentic_model": metadata.get("is_agentic_model"),
        "use_agentic_policy": metadata.get("use_agentic_policy"),
        "has_action_head": metadata.get("has_action_head"),
    }
    (artifact_dir / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _write_train_report(summary: dict, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in summary.get("artifacts", []):
        artifact_folder = Path(item["artifact_folder"])
        metrics_path = artifact_folder / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        rows.append(
            {
                "scenario": item.get("scenario"),
                "model_type": item.get("model_type"),
                "has_action_head": item.get("has_action_head"),
                "agentic_policy": item.get("use_agentic_policy"),
                "r2": metrics.get("r2"),
                "rmse": metrics.get("rmse"),
                "mae": metrics.get("mae"),
                "action_accuracy": metrics.get("action_accuracy"),
                "action_macro_f1": metrics.get("action_macro_f1"),
                "artifact_folder": item.get("artifact_folder"),
            }
        )
    if rows:
        table = pd.DataFrame(rows).to_html(index=False, escape=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "")
    else:
        table = "<p>No models trained.</p>"
    html = f"""
<html><head><meta charset='utf-8'><title>Training Report</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:28px;background:#f8fafc;color:#0f172a;}}h1,h2{{color:#0b3a75;}}table{{border-collapse:collapse;width:100%;background:white;}}th,td{{border:1px solid #cbd5e1;padding:8px;font-size:13px;}}th{{background:#dbeafe;color:#1e3a8a;}}.section{{background:white;border:1px solid #cbd5e1;border-radius:10px;padding:14px 18px;margin-bottom:16px;}}code{{background:#e2e8f0;padding:2px 5px;border-radius:4px;}}</style>
</head><body>
<h1>Training Report</h1>
<div class='section'><p><strong>Artifacts folder:</strong> <code>results/train/artifacts/&lt;model_name&gt;</code></p><p>Training no longer writes model outputs to <code>results/&lt;model_name&gt;</code>. Only action-head agentic models are in the main study.</p></div>
<div class='section'><h2>Trained agentic action models</h2>{table}</div>
</body></html>
"""
    report_path.write_text(html, encoding="utf-8")


def main() -> None:
    results_root = Path("results")
    out_dir = results_root / "train"
    artifacts_root = out_dir / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_root.mkdir(parents=True, exist_ok=True)
    _remove_old_root_model_outputs(results_root)

    target_col = os.getenv("TARGET_COL", DEFAULT_TARGET_COL)
    log_target = os.getenv("LOG_TARGET", "0") == "1"
    loss = os.getenv("LOSS") or None
    peak_weight = os.getenv("PEAK_WEIGHT")
    peak_weight_value = float(peak_weight) if peak_weight else None

    completed: list[str] = []
    failed: dict[str, str] = {}
    artifacts: list[dict] = []

    for scenario_name in SCENARIOS:
        print(f"[train] running action model: {scenario_name}")
        artifact_dir = artifacts_root / scenario_name
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        try:
            run(
                scenario_name,
                target_col=target_col,
                log_target=log_target,
                loss=loss,
                peak_weight=peak_weight_value,
                results_dir=artifact_dir,
            )
            completed.append(scenario_name)
            artifacts.append(_manifest_for_artifact(scenario_name, artifact_dir))
        except Exception as exc:  # keep training report useful even if one model fails
            failed[scenario_name] = repr(exc)
            (artifact_dir / "status.json").write_text(json.dumps({"status": "failure", "error": repr(exc)}, indent=2), encoding="utf-8")
            print(f"[train] warning: model failed: {scenario_name}: {exc!r}")

    summary = {
        "target_col": target_col,
        "log_target": log_target,
        "loss": loss,
        "peak_weight": peak_weight_value,
        "study_scope": "agentic action-head models only",
        "completed_scenarios": completed,
        "failed_scenarios": failed,
        "artifacts_root": str(artifacts_root),
        "artifacts": artifacts,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_train_report(summary, out_dir / "report.html")
    print("[train] report saved to results/train/report.html")
    print("[train] artifacts saved to results/train/artifacts/<model_name>")


if __name__ == "__main__":
    main()
