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
]


def _copy_training_artifacts(scenario_name: str, artifacts_root: Path) -> dict:
    src_dir = Path("results") / scenario_name
    dst_dir = artifacts_root / scenario_name
    dst_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for filename in ARTIFACT_FILES:
        src = src_dir / filename
        if src.exists():
            shutil.copy2(src, dst_dir / filename)
            copied.append(filename)

    metadata_path = src_dir / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    artifact_manifest = {
        "scenario": scenario_name,
        "source_folder": str(src_dir),
        "artifact_folder": str(dst_dir),
        "copied_files": copied,
        "model_type": metadata.get("model_type"),
        "sequence_length": metadata.get("sequence_length"),
        "is_agentic_model": metadata.get("is_agentic_model"),
        "use_agentic_policy": metadata.get("use_agentic_policy"),
    }
    (dst_dir / "artifact_manifest.json").write_text(json.dumps(artifact_manifest, indent=2), encoding="utf-8")
    return artifact_manifest


def _write_train_report(summary: dict, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in summary.get("artifacts", []):
        metrics_path = Path(item["source_folder"]) / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        rows.append(
            {
                "scenario": item.get("scenario"),
                "model_type": item.get("model_type"),
                "agentic_model": item.get("is_agentic_model"),
                "agentic_policy": item.get("use_agentic_policy"),
                "r2": metrics.get("r2"),
                "rmse": metrics.get("rmse"),
                "mae": metrics.get("mae"),
                "action_accuracy": metrics.get("action_accuracy"),
                "artifact_folder": item.get("artifact_folder"),
            }
        )
    table = pd.DataFrame(rows).to_html(index=False, escape=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "") if rows else "<p>No models trained.</p>"
    html = f"""
<html><head><meta charset='utf-8'><title>Training Report</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:28px;background:#f8fafc;color:#0f172a;}}h1,h2{{color:#0b3a75;}}table{{border-collapse:collapse;width:100%;background:white;}}th,td{{border:1px solid #cbd5e1;padding:8px;font-size:13px;}}th{{background:#dbeafe;color:#1e3a8a;}}.section{{background:white;border:1px solid #cbd5e1;border-radius:10px;padding:14px 18px;margin-bottom:16px;}}code{{background:#e2e8f0;padding:2px 5px;border-radius:4px;}}</style>
</head><body>
<h1>Training Report</h1>
<div class='section'><p><strong>Artifacts folder:</strong> <code>results/train/artifacts</code></p><p>Every trained model stores its weights, metadata, metrics, training log, data summary, and agentic-policy summary when available.</p></div>
<div class='section'><h2>Trained models</h2>{table}</div>
</body></html>
"""
    report_path.write_text(html, encoding="utf-8")


def main() -> None:
    out_dir = Path("results/train")
    artifacts_root = out_dir / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    target_col = os.getenv("TARGET_COL", DEFAULT_TARGET_COL)
    log_target = os.getenv("LOG_TARGET", "0") == "1"
    loss = os.getenv("LOSS") or None
    peak_weight = os.getenv("PEAK_WEIGHT")
    peak_weight_value = float(peak_weight) if peak_weight else None

    completed: list[str] = []
    failed: dict[str, str] = {}
    artifacts: list[dict] = []

    for scenario_name in SCENARIOS:
        print(f"[train] running scenario/model: {scenario_name}")
        try:
            run(scenario_name, target_col=target_col, log_target=log_target, loss=loss, peak_weight=peak_weight_value)
            completed.append(scenario_name)
            artifacts.append(_copy_training_artifacts(scenario_name, artifacts_root))
        except Exception as exc:  # keep training report useful even if one model fails
            failed[scenario_name] = repr(exc)
            status_dir = Path("results") / scenario_name
            status_dir.mkdir(parents=True, exist_ok=True)
            (status_dir / "status.json").write_text(json.dumps({"status": "failure", "error": repr(exc)}, indent=2), encoding="utf-8")
            print(f"[train] warning: scenario failed: {scenario_name}: {exc!r}")

    summary = {
        "target_col": target_col,
        "log_target": log_target,
        "loss": loss,
        "peak_weight": peak_weight_value,
        "completed_scenarios": completed,
        "failed_scenarios": failed,
        "artifacts_root": str(artifacts_root),
        "artifacts": artifacts,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_train_report(summary, out_dir / "report.html")
    print("[train] report saved to results/train/report.html")
    print("[train] artifacts saved to results/train/artifacts")


if __name__ == "__main__":
    main()
