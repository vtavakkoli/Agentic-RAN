"""Intent compilation, slice-aware multi-objective scoring, and Pareto analysis."""

from __future__ import annotations

from dataclasses import dataclass

from agentic_ran.domain import CandidateEvaluation, SLAConstraint, SLAIntent


DEFAULT_INTENTS: dict[str, SLAIntent] = {
    "balanced": SLAIntent(
        name="balanced",
        weights={"latency": 0.22, "loss": 0.20, "throughput": 0.28, "stability": 0.15, "energy": 0.15},
        risk_tolerance=0.30,
    ),
    "urllc-strict": SLAIntent(
        name="urllc-strict",
        slice_type="URLLC",
        constraints=[
            SLAConstraint(metric="latency_ms", operator="<=", value=10.0),
            SLAConstraint(metric="packet_loss_pct", operator="<=", value=0.10),
        ],
        weights={"latency": 0.43, "loss": 0.27, "throughput": 0.12, "stability": 0.10, "energy": 0.08},
        risk_tolerance=0.15,
    ),
    "embb-capacity": SLAIntent(
        name="embb-capacity",
        slice_type="eMBB",
        weights={"latency": 0.16, "loss": 0.12, "throughput": 0.45, "stability": 0.10, "energy": 0.17},
        risk_tolerance=0.32,
    ),
    "mmtc-reliability": SLAIntent(
        name="mmtc-reliability",
        slice_type="mMTC",
        weights={"latency": 0.10, "loss": 0.31, "throughput": 0.12, "stability": 0.25, "energy": 0.22},
        risk_tolerance=0.22,
    ),
    "green-ran": SLAIntent(
        name="green-ran",
        constraints=[SLAConstraint(metric="energy_load", operator="<=", value=0.72, hard=False)],
        weights={"latency": 0.12, "loss": 0.15, "throughput": 0.18, "stability": 0.15, "energy": 0.40},
        risk_tolerance=0.20,
        energy_budget=0.82,
    ),
}


class IntentCompiler:
    def __init__(self, profiles: dict[str, SLAIntent] | None = None):
        self.profiles = profiles or DEFAULT_INTENTS

    def compile(self, intent: SLAIntent | str | None, slice_type: str) -> SLAIntent:
        if isinstance(intent, SLAIntent):
            return intent
        if isinstance(intent, str):
            if intent not in self.profiles:
                raise KeyError(f"unknown intent profile: {intent}")
            return self.profiles[intent].model_copy(deep=True)
        if slice_type == "URLLC":
            return self.profiles["urllc-strict"].model_copy(deep=True)
        if slice_type == "mMTC":
            return self.profiles["mmtc-reliability"].model_copy(deep=True)
        return self.profiles["embb-capacity"].model_copy(deep=True)


@dataclass(frozen=True, slots=True)
class ParetoPoint:
    policy: str
    latency: float
    loss: float
    throughput: float
    energy: float
    stability: float


class ParetoOptimizer:
    @staticmethod
    def _point(candidate: CandidateEvaluation) -> ParetoPoint:
        kpi = candidate.predicted_kpis
        return ParetoPoint(
            policy=candidate.policy,
            latency=kpi.get("latency_ms", float("inf")),
            loss=kpi.get("packet_loss_pct", float("inf")),
            throughput=kpi.get("downlink_mbps", 0.0),
            energy=kpi.get("energy_load", float("inf")),
            stability=kpi.get("handover_failure_pct", float("inf")),
        )

    @staticmethod
    def _dominates(left: ParetoPoint, right: ParetoPoint) -> bool:
        no_worse = (
            left.latency <= right.latency
            and left.loss <= right.loss
            and left.throughput >= right.throughput
            and left.energy <= right.energy
            and left.stability <= right.stability
        )
        strictly_better = (
            left.latency < right.latency
            or left.loss < right.loss
            or left.throughput > right.throughput
            or left.energy < right.energy
            or left.stability < right.stability
        )
        return no_worse and strictly_better

    def front(self, candidates: list[CandidateEvaluation]) -> list[str]:
        points = [self._point(candidate) for candidate in candidates if candidate.safe]
        return [point.policy for point in points if not any(self._dominates(other, point) for other in points if other != point)]
