from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(cmd: list[str]) -> None:
    print(f"[CMD] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True)
    p.add_argument("--input", default="shared_data/dataset-kpm")
    p.add_argument("--steps", type=int, default=5000)
    args = p.parse_args()

    scenario = args.scenario
    sdir = Path("results/scenarios") / scenario
    sdir.mkdir(parents=True, exist_ok=True)
    status = {
        "scenario_name": scenario,
        "success": False,
        "start_time": _iso_now(),
        "end_time": None,
        "metrics_path": str(sdir / "model" / "metrics.json"),
        "preds_path": str(sdir / "preds.csv"),
    }

    try:
        print(f"[{scenario}] scenario started", flush=True)
        csv = Path("shared_data") / f"traffic_data_{scenario}.csv"
        print(f"[{scenario}] data generation started", flush=True)
        run(["python", "generate_data.py", "--steps", str(args.steps), "--input", args.input, "--output", str(csv)])
        print(f"[{scenario}] data generation done", flush=True)

        print(f"[{scenario}] training started", flush=True)
        model_type = "ridge" if scenario in {"lightweight-32", "lightweight-64", "attention-baseline", "liquid-baseline"} else "hgb"
        run(["python", "-m", "scripts.train", "--csv", str(csv), "--out_dir", str(sdir / "model"), "--seed", "42", "--model", model_type])
        print(f"[{scenario}] training done", flush=True)

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
