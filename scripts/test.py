from __future__ import annotations

from pathlib import Path

from scripts.aggregate_report import aggregate


def main() -> None:
    aggregate(Path("results"))

    test_dir = Path("results/test")
    test_dir.mkdir(parents=True, exist_ok=True)

    src = Path("results/report.html")
    content = src.read_text(encoding="utf-8") if src.exists() else "<html><body><h1>Test Report</h1><p>No report generated.</p></body></html>"
    wrapped = f"<html><body><h1>Comprehensive Test Report</h1>{content}</body></html>"
    (test_dir / "test.html").write_text(wrapped, encoding="utf-8")
    print("[test] report saved to results/test/test.html")


if __name__ == "__main__":
    main()
