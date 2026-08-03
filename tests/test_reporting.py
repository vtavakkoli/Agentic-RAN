from __future__ import annotations

from pathlib import Path

from agentic_ran.data import generate_dataset, write_dataset
from agentic_ran.reporting import benchmark
from agentic_ran.service import PolicyService


def test_benchmark_writes_json_and_html(tmp_path: Path, trained_service: PolicyService) -> None:
    dataset = write_dataset(generate_dataset(rows=180, seed=17), tmp_path / "data.csv")
    metrics = benchmark(trained_service, dataset, tmp_path / "results", limit=80)
    assert metrics["samples"] == 80
    assert metrics["latency_ms"]["p95"] >= 0
    assert metrics["throughput_decisions_per_second"] > 0
    assert metrics["guardrail_checks_passed"] == metrics["guardrail_checks_total"]
    assert (tmp_path / "results/benchmark.json").exists()
    html = (tmp_path / "results/benchmark.html").read_text(encoding="utf-8")
    assert "Agentic-RAN policy engine" in html
    assert "Policy distribution" in html
