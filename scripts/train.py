from __future__ import annotations

import json
import os
from pathlib import Path

from agentic_ran.data_loading import DEFAULT_TARGET_COL
from agentic_ran.scenarios import SCENARIOS
from scripts.run_scenario import run


def main() -> None:
    out_dir = Path("results/train")
    out_dir.mkdir(parents=True, exist_ok=True)

    target_col = os.getenv("TARGET_COL", DEFAULT_TARGET_COL)
    log_target = os.getenv("LOG_TARGET", "0") == "1"
    loss = os.getenv("LOSS") or None
    peak_weight = os.getenv("PEAK_WEIGHT")
    peak_weight_value = float(peak_weight) if peak_weight else None

    completed = []
    for scenario_name in SCENARIOS:
        print(f"[train] running scenario: {scenario_name}")
        run(scenario_name, target_col=target_col, log_target=log_target, loss=loss, peak_weight=peak_weight_value)
        completed.append(scenario_name)

    summary = {
        "target_col": target_col,
        "log_target": log_target,
        "loss": loss,
        "peak_weight": peak_weight_value,
        "scenarios": completed,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    html = """<html><body><h1>Training Report</h1>
<p>Training completed for all configured scenarios.</p>
<p>Check generated artifacts in <code>results/</code> and weights in <code>ml_models/</code>.</p>
<ul>{items}</ul>
</body></html>""".format(items="".join([f"<li>{s}</li>" for s in completed]))
    (out_dir / "train.html").write_text(html, encoding="utf-8")
    print("[train] report saved to results/train/train.html")


if __name__ == "__main__":
    main()
