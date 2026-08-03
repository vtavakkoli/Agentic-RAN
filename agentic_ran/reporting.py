"""Benchmark and HTML reporting utilities."""

from __future__ import annotations

import html
import json
import time
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from agentic_ran.data import load_dataset
from agentic_ran.domain import NetworkObservation
from agentic_ran.service import PolicyService


def benchmark(
    service: PolicyService,
    dataset_path: Path | str,
    output_dir: Path | str,
    limit: int = 400,
) -> dict[str, Any]:
    frame = load_dataset(dataset_path).head(limit)
    latencies_ms: list[float] = []
    selected: Counter[str] = Counter()
    safety_overrides = 0
    agreement = 0

    for row in frame.to_dict(orient="records"):
        observation = NetworkObservation(**{key: value for key, value in row.items() if key not in {"policy_label"}})
        started = time.perf_counter_ns()
        decision = service.decide(observation)
        latencies_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        selected[decision.selected_policy] += 1
        safety_overrides += int(decision.safety_override)
        agreement += int(decision.selected_policy == row["policy_label"])

    guardrail_scenarios = _run_guardrail_scenarios(service)
    metrics = {
        "samples": len(frame),
        "model_version": service.proposer.metadata.version,
        "training_accuracy": service.proposer.metadata.accuracy,
        "training_macro_f1": service.proposer.metadata.macro_f1,
        "expert_policy_agreement": agreement / max(len(frame), 1),
        "safety_override_rate": safety_overrides / max(len(frame), 1),
        "latency_ms": {
            "mean": float(np.mean(latencies_ms)),
            "median": float(median(latencies_ms)),
            "p95": float(np.percentile(latencies_ms, 95)),
            "max": float(max(latencies_ms, default=0.0)),
        },
        "throughput_decisions_per_second": 1000.0 / max(float(np.mean(latencies_ms)), 0.001),
        "policy_distribution": dict(sorted(selected.items())),
        "guardrail_checks": guardrail_scenarios,
        "guardrail_checks_passed": sum(int(item["passed"]) for item in guardrail_scenarios),
        "guardrail_checks_total": len(guardrail_scenarios),
        "notes": [
            "Expert-policy agreement compares the final guarded decision with transparent bootstrap labels; it is not operator ground truth.",
            "Latency measures the complete local propose-evaluate-guard-select loop on the current machine.",
        ],
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "benchmark.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (destination / "benchmark.html").write_text(_render_html(metrics), encoding="utf-8")
    return metrics


def _run_guardrail_scenarios(service: PolicyService) -> list[dict[str, Any]]:
    """Exercise adversarial snapshots that require the deterministic critic."""

    scenarios = [
        {
            "name": "block-energy-saving-near-urllc-sla",
            "observation": NetworkObservation(
                cell_id="guard-urllc",
                slice_type="URLLC",
                prb_utilization=0.25,
                active_users=24,
                downlink_mbps=60.0,
                uplink_mbps=8.0,
                latency_ms=9.5,
                jitter_ms=2.0,
                packet_loss_pct=0.15,
                throughput_demand_mbps=45.0,
                energy_load=0.30,
                handover_failure_pct=0.2,
                rsrp_dbm=-90.0,
                sinr_db=22.0,
            ),
            "must_reject": "energy_saver",
        },
        {
            "name": "block-throughput-boost-during-critical-congestion",
            "observation": NetworkObservation(
                cell_id="guard-congestion",
                slice_type="eMBB",
                prb_utilization=1.08,
                active_users=480,
                downlink_mbps=70.0,
                uplink_mbps=15.0,
                latency_ms=88.0,
                jitter_ms=24.0,
                packet_loss_pct=4.8,
                throughput_demand_mbps=280.0,
                energy_load=1.02,
                handover_failure_pct=1.8,
                rsrp_dbm=-101.0,
                sinr_db=6.0,
            ),
            "must_reject": "throughput_boost",
        },
    ]
    results: list[dict[str, Any]] = []
    for scenario in scenarios:
        decision = service.decide(scenario["observation"])
        rejected = {candidate.policy for candidate in decision.candidates if not candidate.safe}
        required = str(scenario["must_reject"])
        results.append(
            {
                "name": scenario["name"],
                "must_reject": required,
                "selected_policy": decision.selected_policy,
                "rejected_policies": sorted(rejected),
                "passed": required in rejected and decision.selected_policy != required,
            }
        )
    return results


def _render_html(metrics: dict[str, Any]) -> str:
    distribution_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{count}</td><td>{100*count/metrics['samples']:.1f}%</td></tr>"
        for name, count in metrics["policy_distribution"].items()
    )
    latency = metrics["latency_ms"]
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Agentic-RAN Benchmark</title><style>
body{{margin:0;background:#f3f6f9;color:#17212b;font:15px/1.5 Inter,Segoe UI,sans-serif}}main{{max-width:1100px;margin:auto;padding:38px 20px}}h1{{font-size:42px;margin:0}}.muted{{color:#627386}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:24px 0}}.card,section{{background:white;border:1px solid #dce3ea;border-radius:14px;padding:18px;box-shadow:0 8px 30px #2030400c}}.value{{font-size:29px;font-weight:800;color:#0a6c63}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid #e7edf2;text-align:left}}th{{color:#4b6073}}code{{background:#eef3f6;padding:2px 5px;border-radius:5px}}</style></head><body><main>
<p class='muted'>Reproducible local benchmark</p><h1>Agentic-RAN policy engine</h1><p class='muted'>Model <code>{html.escape(metrics['model_version'])}</code> evaluated on {metrics['samples']} KPI observations.</p>
<div class='cards'><div class='card'><div class='muted'>Expert agreement</div><div class='value'>{metrics['expert_policy_agreement']*100:.1f}%</div></div><div class='card'><div class='muted'>p95 decision latency</div><div class='value'>{latency['p95']:.3f} ms</div></div><div class='card'><div class='muted'>Throughput</div><div class='value'>{metrics['throughput_decisions_per_second']:.0f}/s</div></div><div class='card'><div class='muted'>Safety overrides</div><div class='value'>{metrics['safety_override_rate']*100:.1f}%</div></div><div class='card'><div class='muted'>Guardrail checks</div><div class='value'>{metrics['guardrail_checks_passed']}/{metrics['guardrail_checks_total']}</div></div></div>
<section><h2>Policy distribution</h2><table><thead><tr><th>Policy</th><th>Selections</th><th>Share</th></tr></thead><tbody>{distribution_rows}</tbody></table></section>
<section><h2>Latency details</h2><table><tr><th>Mean</th><td>{latency['mean']:.4f} ms</td></tr><tr><th>Median</th><td>{latency['median']:.4f} ms</td></tr><tr><th>p95</th><td>{latency['p95']:.4f} ms</td></tr><tr><th>Maximum</th><td>{latency['max']:.4f} ms</td></tr></table></section>
<section><h2>Guardrail scenarios</h2><pre>{html.escape(json.dumps(metrics['guardrail_checks'], indent=2))}</pre></section>
<section><h2>Interpretation</h2><ul>{''.join(f'<li>{html.escape(note)}</li>' for note in metrics['notes'])}</ul></section>
</main></body></html>"""
