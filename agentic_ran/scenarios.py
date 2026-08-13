"""Fault injection and resilience benchmark scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from statistics import mean

from agentic_ran.domain import NetworkObservation


class FaultScenario(str, Enum):
    TRAFFIC_SPIKE = "traffic_spike"; CELL_OUTAGE = "cell_outage"; BACKHAUL_DEGRADATION = "backhaul_degradation"; POOR_RADIO = "poor_radio"; HANDOVER_STORM = "handover_storm"; PACKET_LOSS_BURST = "packet_loss_burst"; STALE_TELEMETRY = "stale_telemetry"; CORRUPTED_KPI = "corrupted_kpi"; MODEL_DRIFT = "model_drift"; NEIGHBOR_INTERFERENCE = "neighbor_interference"


def apply_fault(observation: NetworkObservation, scenario: FaultScenario) -> NetworkObservation:
    updates: dict[str, object] = {"source": f"fault:{scenario.value}"}
    if scenario == FaultScenario.TRAFFIC_SPIKE: updates.update(prb_utilization=min(1.5, observation.prb_utilization + 0.35), active_users=int(observation.active_users * 1.8), throughput_demand_mbps=observation.throughput_demand_mbps * 1.7)
    elif scenario == FaultScenario.CELL_OUTAGE: updates.update(downlink_mbps=0.0, uplink_mbps=0.0, packet_loss_pct=max(80.0, observation.packet_loss_pct), latency_ms=max(1_500.0, observation.latency_ms))
    elif scenario == FaultScenario.BACKHAUL_DEGRADATION: updates.update(latency_ms=observation.latency_ms * 3.5, jitter_ms=observation.jitter_ms * 4.0, downlink_mbps=observation.downlink_mbps * 0.55)
    elif scenario == FaultScenario.POOR_RADIO: updates.update(rsrp_dbm=-121.0, sinr_db=-4.0, handover_failure_pct=max(7.5, observation.handover_failure_pct))
    elif scenario == FaultScenario.HANDOVER_STORM: updates.update(handover_failure_pct=max(12.0, observation.handover_failure_pct * 4.0), jitter_ms=observation.jitter_ms * 2.0)
    elif scenario == FaultScenario.PACKET_LOSS_BURST: updates.update(packet_loss_pct=min(100.0, observation.packet_loss_pct + 12.0), latency_ms=observation.latency_ms * 1.7)
    elif scenario == FaultScenario.STALE_TELEMETRY: updates.update(telemetry_age_ms=12_000.0, telemetry_completeness=0.78)
    elif scenario == FaultScenario.CORRUPTED_KPI: updates.update(prb_utilization=1.49, telemetry_completeness=0.55, sinr_db=60.0)
    elif scenario == FaultScenario.MODEL_DRIFT: updates.update(active_users=min(100_000, observation.active_users * 8), throughput_demand_mbps=observation.throughput_demand_mbps * 4.0)
    elif scenario == FaultScenario.NEIGHBOR_INTERFERENCE: updates.update(sinr_db=max(-30.0, observation.sinr_db - 15.0), packet_loss_pct=min(100.0, observation.packet_loss_pct + 3.0))
    return observation.model_copy(update=updates)


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    scenario: str; selected_policy: str; confidence: float; approved: bool; ood_score: float; unsafe_candidates: int


class ResilienceBenchmark:
    def __init__(self, service: object): self.service = service
    def run(self, baseline: NetworkObservation):
        results=[]
        for scenario in FaultScenario:
            decision=self.service.decide(apply_fault(baseline, scenario)); results.append(ScenarioResult(scenario.value,decision.selected_policy,decision.confidence,decision.approved_for_execution,decision.ood_score,sum(1 for candidate in decision.candidates if not candidate.safe)))
        return results,{"scenarios":float(len(results)),"autonomous_approval_rate":mean([1.0 if item.approved else 0.0 for item in results]),"mean_ood_score":mean([item.ood_score for item in results]),"mean_confidence":mean([item.confidence for item in results])}
