"""Generate a self-contained HTML report for real-data shadow readiness."""

from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agentic_ran.config import FEATURES
from agentic_ran.domain import NetworkObservation
from agentic_ran.reporting import benchmark
from agentic_ran.service import PolicyService


def _real_metrics(service: PolicyService, path: Path | str, limit: int) -> dict[str, Any]:
    frame = pd.read_csv(path).head(limit)
    policies: Counter[str] = Counter()
    agreements = 0
    overrides = 0
    safe = 0
    confidences = []
    ood_scores = []
    uncertainties = []
    for row in frame.to_dict(orient="records"):
        observation = NetworkObservation(**{name: row[name] for name in FEATURES}, cell_id=str(row["cell_id"]), timestamp=row["timestamp"])
        decision = service.decide(observation)
        policies[decision.selected_policy] += 1
        agreements += int(decision.selected_policy == row["policy_label"])
        overrides += int(decision.safety_override)
        selected = next(item for item in decision.candidates if item.policy == decision.selected_policy)
        safe += int(selected.safe)
        confidences.append(decision.confidence)
        ood_scores.append(decision.ood_score)
        if decision.uncertainty is not None:
            uncertainties.append(decision.uncertainty.combined)
    samples = max(len(frame), 1)
    return {
        "samples": len(frame),
        "expert_reference_agreement": agreements / samples,
        "safe_selection_rate": safe / samples,
        "safety_override_rate": overrides / samples,
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "mean_ood_score": float(np.mean(ood_scores)) if ood_scores else 0.0,
        "p95_ood_score": float(np.percentile(ood_scores, 95)) if ood_scores else 0.0,
        "mean_uncertainty": float(np.mean(uncertainties)) if uncertainties else 0.0,
        "policy_distribution": dict(sorted(policies.items())),
        "mean_realism_score": float(frame["realism_score"].mean()) if "realism_score" in frame else 0.0,
    }


def _read_json(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verdict(selection: dict[str, Any], real: dict[str, Any], synthetic: dict[str, Any]) -> tuple[str, list[str]]:
    checks = {
        "selected model synthetic macro-F1 >= 0.75": float(selection["winner"]["synthetic_macro_f1"]) >= 0.75,
        "all deterministic guardrail checks pass": synthetic["guardrail_checks_passed"] == synthetic["guardrail_checks_total"],
        "safe policy selected on every real observation": real["safe_selection_rate"] == 1.0,
        "mean OOD score <= 0.55": real["mean_ood_score"] <= 0.55,
        "expert-reference agreement >= 0.70": real["expert_reference_agreement"] >= 0.70,
    }
    passed = sum(checks.values())
    if passed == len(checks):
        verdict = "SHADOW-READY"
    elif passed >= len(checks) - 1:
        verdict = "RESEARCH-READY"
    else:
        verdict = "NEEDS CALIBRATION"
    details = [f"{'PASS' if value else 'FAIL'} — {name}" for name, value in checks.items()]
    return verdict, details


def generate_production_report(
    service: PolicyService,
    synthetic_path: Path | str,
    real_path: Path | str,
    selection_path: Path | str,
    provenance_path: Path | str,
    output_path: Path | str = "results/report.html",
    limit: int = 1000,
) -> dict[str, Any]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    synthetic = benchmark(service, synthetic_path, output.parent, limit=min(limit, 400))
    real = _real_metrics(service, real_path, limit)
    selection = _read_json(selection_path)
    provenance = _read_json(provenance_path)
    verdict, checks = _verdict(selection, real, synthetic)
    summary = {"verdict": verdict, "checks": checks, "selection": selection, "real": real, "synthetic": synthetic, "provenance": provenance}
    output.write_text(_render(summary), encoding="utf-8")
    (output.parent / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _render(payload: dict[str, Any]) -> str:
    selection = payload["selection"]
    real = payload["real"]
    synthetic = payload["synthetic"]
    provenance = payload["provenance"]
    candidate_rows = "".join(
        "<tr>"
        f"<td>{html.escape(item['model'])}</td>"
        f"<td>{item['synthetic_macro_f1']:.3f}</td>"
        f"<td>{item['real_expert_macro_f1']:.3f}</td>"
        f"<td>{item['robustness_stability']:.3f}</td>"
        f"<td>{item['log_loss']:.3f}</td>"
        f"<td>{item['p95_inference_ms']:.3f}</td>"
        f"<td><strong>{item['production_score']:.3f}</strong></td>"
        "</tr>"
        for item in selection["candidates"]
    )
    source_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item['id']))}</td>"
        f"<td>{html.escape(str(item.get('doi', '')))}</td>"
        f"<td>{html.escape(str(item.get('license', 'source-record')))}</td>"
        f"<td>{item['rows']}</td>"
        "</tr>"
        for item in provenance["sources"]
    )
    check_rows = "".join(f"<li>{html.escape(item)}</li>" for item in payload["checks"])
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Agentic-RAN Real-Data Readiness Report</title><style>
body{{margin:0;background:#f4f7fa;color:#15202b;font:15px/1.55 Inter,system-ui,sans-serif}}main{{max-width:1180px;margin:auto;padding:40px 22px}}h1{{font-size:42px;margin:4px 0}}.eyebrow{{letter-spacing:.12em;text-transform:uppercase;color:#506579;font-weight:700}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:24px 0}}.card,section{{background:white;border:1px solid #dce4eb;border-radius:14px;padding:18px;margin:14px 0;box-shadow:0 8px 30px #2030400b}}.value{{font-size:28px;font-weight:800;color:#086b61}}.verdict{{color:#064e3b;background:#d1fae5;border-radius:10px;padding:10px 14px;display:inline-block;font-weight:800}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;border-bottom:1px solid #e7edf2;text-align:left}}th{{color:#536879}}.warn{{background:#fff7ed;border-left:4px solid #f59e0b;padding:12px}}code{{background:#edf2f6;padding:2px 5px;border-radius:5px}}</style></head><body><main>
<div class='eyebrow'>Real-data model selection · reproducible Docker workflow</div><h1>Agentic-RAN readiness report</h1><p class='verdict'>{payload['verdict']}</p>
<div class='grid'><div class='card'><div>Selected model</div><div class='value'>{html.escape(selection['selected_model'])}</div></div><div class='card'><div>Real observations</div><div class='value'>{real['samples']}</div></div><div class='card'><div>Safe selections</div><div class='value'>{real['safe_selection_rate']*100:.1f}%</div></div><div class='card'><div>Real expert agreement</div><div class='value'>{real['expert_reference_agreement']*100:.1f}%</div></div><div class='card'><div>Mean OOD</div><div class='value'>{real['mean_ood_score']:.3f}</div></div><div class='card'><div>p95 decision latency</div><div class='value'>{synthetic['latency_ms']['p95']:.3f} ms</div></div></div>
<section><h2>Readiness gates</h2><ul>{check_rows}</ul><p class='warn'><strong>Important:</strong> SHADOW-READY does not mean production-certified. Real operator actions are absent from the public datasets. Active/canary control still requires calibrated RIC/gNB lab validation, authenticated transport, operator approval and rollback exercises.</p></section>
<section><h2>Model comparison</h2><p>{html.escape(selection['selection_formula'])}</p><table><thead><tr><th>Model</th><th>Synthetic F1</th><th>Real reference F1</th><th>Robustness</th><th>Log loss</th><th>p95 ms</th><th>Score</th></tr></thead><tbody>{candidate_rows}</tbody></table></section>
<section><h2>Real-data provenance</h2><table><thead><tr><th>Source</th><th>DOI</th><th>License</th><th>Rows</th></tr></thead><tbody>{source_rows}</tbody></table><p>Prepared dataset SHA-256: <code>{html.escape(provenance['model_ready_sha256'])}</code></p><p>Mean measured-feature realism score: {real['mean_realism_score']:.3f}. Missing RAN counters are explicitly derived and tracked in the prepared CSV.GZ.</p></section>
<section><h2>Real-data shadow behavior</h2><pre>{html.escape(json.dumps(real, indent=2))}</pre></section>
<section><h2>Synthetic safety benchmark</h2><p>Guardrails: {synthetic['guardrail_checks_passed']}/{synthetic['guardrail_checks_total']} passed · expert agreement {synthetic['expert_policy_agreement']*100:.1f}%.</p><pre>{html.escape(json.dumps(synthetic['guardrail_checks'], indent=2))}</pre></section>
<section><h2>Interpretation</h2><p>The public measurements improve external realism for radio strength, throughput and latency. They do not expose the complete internal RAN state used by the controller, and they do not provide operator policy labels. Therefore this report measures shadow-readiness and model robustness, not causal production performance.</p></section>
</main></body></html>"""
