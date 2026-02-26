from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from oran_sim.config import supported_scenarios


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

    rows = []
    for p in status_paths:
        if p.exists():
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        else:
            rows.append({"scenario_name": p.parent.name, "success": False, "error": "missing status"})

    status_df = pd.DataFrame(rows)
    status_df.to_csv(out_dir / "scenario_status.csv", index=False)

    available = [r for r in rows if r.get("success")]
    if available:
        first = available[0]
        metrics = Path(first["metrics_path"])
        preds = Path(first["preds_path"])
        cfg = metrics.parent / "config.json"
        if metrics.exists() and preds.exists():
            import subprocess

            subprocess.run(
                [
                    "python",
                    "-m",
                    "scripts.report",
                    "--preds",
                    str(preds),
                    "--metrics",
                    str(metrics),
                    "--config",
                    str(cfg),
                    "--out",
                    str(out_dir / "report.html"),
                ],
                check=True,
            )
            print("[aggregator] final report generated", flush=True)
            return

    (out_dir / "report.html").write_text(
        "<html><body><h1>KPM Final Report</h1><p>No successful scenarios.</p></body></html>", encoding="utf-8"
    )
    print("[aggregator] generated fallback report with failures", flush=True)


if __name__ == "__main__":
    main()
