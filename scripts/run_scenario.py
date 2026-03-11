from __future__ import annotations

import argparse
import json
import os
import subprocess

from oran_sim.config import SCENARIOS
from datetime import datetime, timezone
from pathlib import Path


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(cmd: list[str]) -> None:
    print(f"[CMD] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _resolve_dataset_path(scenario: str, dataset: str | None) -> Path:
    if dataset:
        return Path(dataset)

    candidates = [
        Path("shared_data") / f"traffic_data_{scenario}.csv",
        Path("shared_data") / "traffic_data.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No dataset found for scenario '{scenario}'. Expected one of: "
        + ", ".join(str(c) for c in candidates)
        + ". Pre-generate data outside docker and mount it in shared_data/."
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True)
    p.add_argument(
        "--dataset",
        default=None,
        help="Path to existing dataset CSV. If omitted, tries shared_data/traffic_data_<scenario>.csv then shared_data/traffic_data.csv.",
    )
    args = p.parse_args()

    scenario = args.scenario
    epochs = int(os.getenv("EPOCHS", "5"))
    sdir = Path("results/scenarios") / scenario
    sdir.mkdir(parents=True, exist_ok=True)
    status = {
        "scenario_name": scenario,
        "success": False,
        "start_time": _iso_now(),
        "end_time": None,
        "metrics_path": str(sdir / "model" / "metrics.json"),
        "preds_path": str(sdir / "preds.csv"),
        "epoch_metrics_path": str(sdir / "model" / "epoch_metrics.csv"),
        "dataset_path": None,
        "epochs": epochs,
        "selected_features": [],
        "feature_importance_path": str(sdir / "model" / "feature_importance.json"),
        "split_metadata_path": str(sdir / "model" / "split_metadata.json"),
    }

    try:
        print(f"[{scenario}] scenario started", flush=True)
        csv = _resolve_dataset_path(scenario, args.dataset)
        status["dataset_path"] = str(csv)
        print(f"[{scenario}] using existing dataset: {csv}", flush=True)
        print(f"[{scenario}] epochs={epochs}", flush=True)

        print(f"[{scenario}] training started", flush=True)
        run(
            [
                "python",
                "-m",
                "scripts.train",
                "--csv",
                str(csv),
                "--out_dir",
                str(sdir / "model"),
                "--seed",
                "42",
                "--model",
                scenario,
                "--epochs",
                str(epochs),
                "--feature_count",
                str(SCENARIOS[scenario].features),
            ]
        )
        print(f"[{scenario}] training done", flush=True)

        features_path = sdir / "model" / "features.json"
        if features_path.exists():
            status["selected_features"] = json.loads(features_path.read_text(encoding="utf-8"))

        split_meta_path = sdir / "model" / "split_metadata.json"
        if split_meta_path.exists():
            status["split_metadata"] = json.loads(split_meta_path.read_text(encoding="utf-8"))

        cfg_path = sdir / "model" / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            status["model_backend"] = cfg.get("model_backend")
            status["logical_profile"] = cfg.get("logical_profile")
            status["profile_note"] = cfg.get("profile_note")
            status["seq_len"] = cfg.get("seq_len")

        print(f"[{scenario}] prediction started", flush=True)
        run(
            [
                "python",
                "-m",
                "scripts.predict",
                "--model_dir",
                str(sdir / "model"),
                "--csv",
                str(csv.with_name(f"{csv.stem}_test.csv")),
                "--output",
                str(sdir / "preds.csv"),
            ]
        )
        print(f"[{scenario}] prediction done", flush=True)
        status["success"] = True
    except Exception as exc:
        status["error"] = str(exc)
        print(f"[{scenario}] failed: {exc}", flush=True)
    finally:
        status["end_time"] = _iso_now()
        (sdir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"[{scenario}] report generated status={status['success']}", flush=True)


if __name__ == "__main__":
    main()
