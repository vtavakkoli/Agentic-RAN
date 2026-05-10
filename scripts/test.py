from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.aggregate_report import aggregate


def _assert_prediction_alignment(results_root: Path) -> None:
    hashes = set()
    for scenario_dir in sorted(results_root.iterdir()):
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
        # Compare the effective sample universe, not row order.
        # Different scenarios may emit predictions in a different ordering,
        # while still evaluating on the exact same test split.
        hashes.add(tuple(sorted(df["global_index"].tolist())))
    if len(hashes) > 1:
        raise AssertionError("Successful scenarios do not share identical effective test_global_index values.")


def main() -> None:
    aggregate(Path("results"))
    _assert_prediction_alignment(Path("results"))

    test_dir = Path("results/test")
    test_dir.mkdir(parents=True, exist_ok=True)

    src = Path("results/report.html")
    content = src.read_text(encoding="utf-8") if src.exists() else "<html><body><h1>Test Report</h1><p>No report generated.</p></body></html>"
    wrapped = f"<html><body><h1>Comprehensive Test Report</h1>{content}</body></html>"
    (test_dir / "test.html").write_text(wrapped, encoding="utf-8")
    print("[test] report saved to results/test/test.html")


if __name__ == "__main__":
    main()
