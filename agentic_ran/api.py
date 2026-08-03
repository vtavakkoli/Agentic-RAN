"""FastAPI application and built-in demonstration UI."""

from __future__ import annotations

import time
from collections import Counter
from threading import Lock
from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse

from agentic_ran import __version__
from agentic_ran.config import Settings
from agentic_ran.domain import BatchDecisionRequest, NetworkObservation, PolicyDecision
from agentic_ran.service import PolicyService


class RuntimeMetrics:
    """Dependency-free in-process counters suitable for demos and tests."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.requests = 0
        self.errors = 0
        self.total_seconds = 0.0
        self.policies: Counter[str] = Counter()

    def observe(self, policy: str | None, duration: float, error: bool = False) -> None:
        with self._lock:
            self.requests += 1
            self.total_seconds += duration
            self.errors += int(error)
            if policy:
                self.policies[policy] += 1

    def prometheus(self) -> str:
        with self._lock:
            lines = [
                "# HELP agentic_ran_decisions_total Total policy decisions.",
                "# TYPE agentic_ran_decisions_total counter",
                f"agentic_ran_decisions_total {self.requests}",
                "# HELP agentic_ran_decision_errors_total Total failed decisions.",
                "# TYPE agentic_ran_decision_errors_total counter",
                f"agentic_ran_decision_errors_total {self.errors}",
                "# HELP agentic_ran_decision_duration_seconds_sum Sum of decision latency.",
                "# TYPE agentic_ran_decision_duration_seconds_sum counter",
                f"agentic_ran_decision_duration_seconds_sum {self.total_seconds:.9f}",
            ]
            for name, value in sorted(self.policies.items()):
                lines.append(f'agentic_ran_selected_policy_total{{policy="{name}"}} {value}')
            return "\n".join(lines) + "\n"


def create_app(service: PolicyService | None = None) -> FastAPI:
    app = FastAPI(
        title="Agentic-RAN Policy Engine",
        summary="Safe and explainable network-policy selection",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.service = service
    app.state.metrics = RuntimeMetrics()

    def get_service() -> PolicyService:
        if app.state.service is None:
            app.state.service = PolicyService.load(Settings.from_env())
        return app.state.service

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home() -> str:
        return _demo_page()

    @app.get("/healthz", tags=["operations"])
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/readyz", tags=["operations"])
    def readiness(policy_service: PolicyService = Depends(get_service)) -> dict[str, str | int]:
        return {
            "status": "ready",
            "model_version": policy_service.proposer.metadata.version,
            "policy_count": len(policy_service.policies),
        }

    @app.get("/metrics", tags=["operations"])
    def metrics() -> Response:
        return Response(app.state.metrics.prometheus(), media_type="text/plain; version=0.0.4")

    @app.get("/v1/policies", tags=["policies"])
    def list_policies(policy_service: PolicyService = Depends(get_service)) -> dict[str, object]:
        return {
            "model_version": policy_service.proposer.metadata.version,
            "policies": [
                {
                    "name": policy.name,
                    "description": policy.description,
                    "action": policy.action,
                }
                for policy in policy_service.policies.values()
            ],
        }

    @app.post("/v1/decisions", response_model=PolicyDecision, tags=["decisions"])
    def decide(observation: NetworkObservation, policy_service: PolicyService = Depends(get_service)) -> PolicyDecision:
        started = time.perf_counter()
        try:
            decision = policy_service.decide(observation)
        except Exception as exc:  # pragma: no cover - defensive API boundary
            app.state.metrics.observe(None, time.perf_counter() - started, error=True)
            raise HTTPException(status_code=500, detail="Policy decision failed") from exc
        app.state.metrics.observe(decision.selected_policy, time.perf_counter() - started)
        return decision

    @app.post("/v1/decisions/batch", response_model=list[PolicyDecision], tags=["decisions"])
    def decide_batch(request: BatchDecisionRequest, policy_service: PolicyService = Depends(get_service)) -> list[PolicyDecision]:
        decisions: list[PolicyDecision] = []
        for observation in request.observations:
            started = time.perf_counter()
            decision = policy_service.decide(observation)
            app.state.metrics.observe(decision.selected_policy, time.perf_counter() - started)
            decisions.append(decision)
        return decisions

    return app


app = create_app()


def _demo_page() -> str:
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agentic-RAN Policy Engine</title>
<style>
:root{--bg:#07111f;--card:#101d2d;--line:#26394d;--ink:#e7f0f8;--muted:#93a6b8;--accent:#53d6b3;--warn:#ffcb6b}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top,#12314a 0,var(--bg) 48%);color:var(--ink);font:15px/1.55 Inter,system-ui,Segoe UI,sans-serif}
main{max-width:1120px;margin:auto;padding:48px 22px}.hero{display:grid;grid-template-columns:1.3fr .7fr;gap:22px;align-items:end}.eyebrow{color:var(--accent);text-transform:uppercase;letter-spacing:.12em;font-weight:700}.hero h1{font-size:clamp(36px,6vw,70px);line-height:1;margin:8px 0 16px}.hero p{font-size:18px;color:var(--muted);max-width:720px}.badge{border:1px solid var(--line);border-radius:18px;padding:18px;background:#0b1725bb}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:28px}.card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:22px;box-shadow:0 18px 50px #0005}.fields{display:grid;grid-template-columns:1fr 1fr;gap:12px}label{display:grid;gap:5px;color:var(--muted);font-size:13px}input,select{width:100%;border:1px solid var(--line);border-radius:9px;padding:10px;background:#081421;color:var(--ink)}button{border:0;border-radius:10px;padding:12px 18px;background:var(--accent);color:#05241d;font-weight:800;cursor:pointer;margin-top:16px}.result{white-space:pre-wrap;background:#081421;border-radius:12px;padding:16px;min-height:320px;overflow:auto;color:#cfe8df}.steps{display:flex;gap:8px;flex-wrap:wrap;margin-top:18px}.steps span{border:1px solid var(--line);padding:7px 10px;border-radius:999px;color:var(--muted)}a{color:var(--accent)}@media(max-width:800px){.hero,.grid{grid-template-columns:1fr}.fields{grid-template-columns:1fr}}
</style></head><body><main>
<section class="hero"><div><div class="eyebrow">Safe closed-loop control</div><h1>Agentic-RAN</h1><p>A compact policy-selection agent for eMBB, URLLC, and mMTC. It proposes candidate policies, predicts their effects, rejects unsafe actions, and explains the final choice.</p><div class="steps"><span>1 Observe</span><span>2 Propose</span><span>3 Simulate</span><span>4 Guard</span><span>5 Select</span></div></div><div class="badge"><strong>No GPU. No external LLM.</strong><br><span style="color:var(--muted)">Fast, reproducible, explainable, and deployable on CPU-only edge systems.</span><p><a href="/docs">Open API documentation →</a></p></div></section>
<section class="grid"><div class="card"><h2>Try a decision</h2><div class="fields">
<label>Slice<select id="slice_type"><option>eMBB</option><option>URLLC</option><option>mMTC</option></select></label>
<label>PRB utilization<input id="prb_utilization" type="number" step="0.01" value="0.91"></label>
<label>Active users<input id="active_users" type="number" value="210"></label>
<label>Downlink Mbps<input id="downlink_mbps" type="number" value="85"></label>
<label>Demand Mbps<input id="throughput_demand_mbps" type="number" value="210"></label>
<label>Latency ms<input id="latency_ms" type="number" value="48"></label>
<label>Jitter ms<input id="jitter_ms" type="number" value="9"></label>
<label>Packet loss %<input id="packet_loss_pct" type="number" step="0.01" value="2.4"></label>
<label>Energy load<input id="energy_load" type="number" step="0.01" value="0.86"></label>
<label>Handover failures %<input id="handover_failure_pct" type="number" step="0.01" value="1.2"></label>
<label>RSRP dBm<input id="rsrp_dbm" type="number" value="-101"></label>
<label>SINR dB<input id="sinr_db" type="number" value="8"></label>
</div><button onclick="runDecision()">Select policy</button></div><div class="card"><h2>Decision trace</h2><div id="result" class="result">Submit a KPI snapshot to inspect the agent's reasoning.</div></div></section>
</main><script>
async function runDecision(){const n=id=>Number(document.getElementById(id).value);const payload={cell_id:'demo-cell',slice_type:document.getElementById('slice_type').value,prb_utilization:n('prb_utilization'),active_users:n('active_users'),downlink_mbps:n('downlink_mbps'),uplink_mbps:12,latency_ms:n('latency_ms'),jitter_ms:n('jitter_ms'),packet_loss_pct:n('packet_loss_pct'),throughput_demand_mbps:n('throughput_demand_mbps'),energy_load:n('energy_load'),handover_failure_pct:n('handover_failure_pct'),rsrp_dbm:n('rsrp_dbm'),sinr_db:n('sinr_db')};const box=document.getElementById('result');box.textContent='Evaluating candidates…';try{const r=await fetch('/v1/decisions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const d=await r.json();box.textContent=`SELECTED: ${d.selected_policy}\nCONFIDENCE: ${(d.confidence*100).toFixed(1)}%\nSAFETY OVERRIDE: ${d.safety_override}\n\n${d.explanation}\n\nACTION\n${JSON.stringify(d.action,null,2)}\n\nEXPECTED KPIs\n${JSON.stringify(d.expected_kpis,null,2)}\n\nREJECTED\n${d.trace.rejected_policies.join(', ')||'none'}`;}catch(e){box.textContent='Request failed: '+e;}}
</script></body></html>"""
