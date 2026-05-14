from __future__ import annotations

from pathlib import Path
import shutil

import pandas as pd

from scripts.aggregate_report import aggregate


def _assert_prediction_alignment(results_root: Path) -> None:
    hashes = set()
    artifacts_root = results_root / "train" / "artifacts"
    candidate_dirs = sorted(artifacts_root.iterdir()) if artifacts_root.exists() else []
    if not candidate_dirs:
        skip_names = {"train", "test", "generate_data", "figures", "tables", "policies"}
        candidate_dirs = [p for p in sorted(results_root.iterdir()) if p.is_dir() and p.name not in skip_names]
    for scenario_dir in candidate_dirs:
        if not scenario_dir.is_dir():
            continue
        status = scenario_dir / "status.json"
        pred = scenario_dir / "predictions.csv"
        if not status.exists() or not pred.exists():
            continue
        if '"status": "success"' not in status.read_text(encoding="utf-8"):
            continue
        df = pd.read_csv(pred)
        if "global_index" not in df.columns:
            raise AssertionError(f"Missing global_index in {pred}")
        hashes.add(tuple(sorted(df["global_index"].tolist())))
    if len(hashes) > 1:
        print(
            "[test] warning: successful scenarios do not share identical "
            "effective test_global_index values; global ranking fairness is disabled."
        )


def main() -> None:
    report_path = aggregate(Path("results"))
    _assert_prediction_alignment(Path("results"))

    test_dir = Path("results/test")
    test_dir.mkdir(parents=True, exist_ok=True)
    target = test_dir / "report.html"
    if report_path.exists():
        shutil.copy2(report_path, target)
    else:
        target.write_text("<html><body><h1>Test Report</h1><p>No report generated.</p></body></html>", encoding="utf-8")
    print("[test] report saved to results/test/report.html")


if __name__ == "__main__":
    main()
