from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from oran_sim.config import SCENARIOS
from scripts.train import main as train_main  # noqa: F401


def run_cmd(cmd: str) -> None:
    import subprocess

    print(f"[SCENARIO] Running: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one scenario end-to-end")
    parser.add_argument("--scenario", required=True, choices=list(SCENARIOS.keys()))
    parser.add_argument("--data", default="shared_data/traffic_data.csv")
    args = parser.parse_args()

    cfg = SCENARIOS[args.scenario]
    out = Path("results") / args.scenario
    if out.exists():
        shutil.rmtree(out)
    (out / "final").mkdir(parents=True, exist_ok=True)

    print(f"[SCENARIO] started {args.scenario}", flush=True)
    run_cmd(
        "python -m scripts.train "
        f"--data_root {args.data} --out_dir {out / 'model'} --seq_len 32 --feature_count {cfg.feature_count} "
        f"--model {cfg.model} --seed 42"
    )
    run_cmd(
        "python -m scripts.predict "
        f"--model_dir {out / 'model'} --input {args.data} --output {out / 'predictions' / 'preds.csv'}"
    )
    run_cmd(
        "python -m scripts.report "
        f"--metrics {out / 'model' / 'metrics.json'} --preds {out / 'predictions' / 'preds.csv'} --out {out / 'final' / 'report.html'}"
    )
    (out / "finished.marker").write_text("done\n", encoding="utf-8")
    print(f"[SCENARIO] finished {args.scenario}", flush=True)


if __name__ == "__main__":
    main()
