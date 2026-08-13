"""Uncertainty, OOD, temporal safety, and multi-critic safeguards."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone

from agentic_ran.domain import CriticAssessment, NetworkObservation, SLAIntent, UncertaintyAssessment


_REFERENCE_BOUNDS: dict[str, tuple[float, float]] = {
    "prb_utilization": (0.05, 1.18),
    "active_users": (0.0, 1_400.0),
    "downlink_mbps": (0.0, 600.0),
    "uplink_mbps": (0.0, 180.0),
    "latency_ms": (0.5, 260.0),
    "jitter_ms": (0.0, 90.0),
    "packet_loss_pct": (0.0, 15.0),
    "throughput_demand_mbps": (0.0, 500.0),
    "energy_load": (0.05, 1.2),
    "handover_failure_pct": (0.0, 15.0),
    "rsrp_dbm": (-130.0, -55.0),
    "sinr_db": (-10.0, 40.0),
}


class OODDetector:
    """Deterministic feature-envelope OOD detector suitable for safety gating."""

    def __init__(self, bounds: dict[str, tuple[float, float]] | None = None):
        self.bounds = bounds or _REFERENCE_BOUNDS

    def score(self, observation: NetworkObservation) -> float:
        record = observation.feature_record()
        penalties: list[float] = []
        for name, (low, high) in self.bounds.items():
            value = float(record[name])
            span = max(high - low, 1e-9)
            if value < low:
                penalties.append(min(1.0, (low - value) / span * 5.0 + 0.35))
            elif value > high:
                penalties.append(min(1.0, (value - high) / span * 5.0 + 0.35))
            else:
                edge = min(value - low, high - value) / span
                penalties.append(max(0.0, 0.12 - edge) * 0.6)
        return round(min(1.0, max(penalties, default=0.0) * 0.7 + sum(penalties) / max(len(penalties), 1) * 0.3), 4)


class UncertaintyEstimator:
    @staticmethod
    def _normalized_entropy(probabilities: list[float]) -> float:
        values = [max(1e-12, float(value)) for value in probabilities if value >= 0]
        total = sum(values)
        if not values or total <= 0:
            return 1.0
        values = [value / total for value in values]
        entropy = -sum(value * math.log(value) for value in values)
        maximum = math.log(max(len(values), 2))
        return min(1.0, entropy / maximum)

    def assess(
        self,
        probabilities: list[float],
        observation: NetworkObservation,
        ood_score: float,
        twin_uncertainty: float,
    ) -> UncertaintyAssessment:
        entropy = self._normalized_entropy(probabilities)
        age_penalty = min(1.0, observation.telemetry_age_ms / 5_000.0)
        completeness_penalty = 1.0 - observation.telemetry_completeness
        telemetry = min(1.0, 0.6 * age_penalty + 0.4 * completeness_penalty)
        combined = min(1.0, 0.32 * entropy + 0.30 * ood_score + 0.20 * telemetry + 0.18 * twin_uncertainty)
        if combined < 0.30:
            level = "low"
        elif combined < 0.55:
            level = "moderate"
        elif combined < 0.78:
            level = "high"
        else:
            level = "critical"
        return UncertaintyAssessment(
            proposal_entropy=round(entropy, 4),
            ood_score=round(ood_score, 4),
            telemetry_penalty=round(telemetry, 4),
            twin_uncertainty=round(twin_uncertainty, 4),
            combined=round(combined, 4),
            level=level,
        )


@dataclass(slots=True)
class _PolicyEvent:
    when: datetime
    policy: str


class TemporalSafetyGuard:
    """Reject control churn and unsafe repeated transitions before execution."""

    def __init__(self, min_dwell_seconds: float = 5.0, max_switches_per_minute: int = 6):
        self.min_dwell_seconds = min_dwell_seconds
        self.max_switches_per_minute = max_switches_per_minute
        self._history: dict[str, deque[_PolicyEvent]] = defaultdict(lambda: deque(maxlen=64))

    def assess(self, cell_id: str, policy: str, now: datetime | None = None) -> list[str]:
        current = now or datetime.now(timezone.utc)
        events = self._history[cell_id]
        if not events:
            return []
        reasons: list[str] = []
        last = events[-1]
        dwell = (current - last.when).total_seconds()
        if policy != last.policy and dwell < self.min_dwell_seconds:
            reasons.append(f"policy dwell time is only {dwell:.1f}s; minimum is {self.min_dwell_seconds:.1f}s")
        recent = [event for event in events if (current - event.when).total_seconds() <= 60.0]
        switches = sum(1 for left, right in zip(recent, recent[1:]) if left.policy != right.policy)
        if policy != last.policy and switches >= self.max_switches_per_minute:
            reasons.append("policy churn limit reached for this cell")
        return reasons

    def record(self, cell_id: str, policy: str, now: datetime | None = None) -> None:
        self._history[cell_id].append(_PolicyEvent(now or datetime.now(timezone.utc), policy))


class CriticSuite:
    """Independent critics that explain why a candidate is acceptable or rejected."""

    @staticmethod
    def evaluate(
        observation: NetworkObservation,
        predicted: dict[str, float],
        policy_name: str,
        core_reasons: list[str],
        uncertainty: UncertaintyAssessment,
        intent: SLAIntent | None,
        temporal_reasons: list[str],
        interference_reason: str | None = None,
    ) -> CriticAssessment:
        reasons = list(core_reasons) + list(temporal_reasons)
        safety = "reject" if core_reasons else "pass"
        stability = "reject" if temporal_reasons else "pass"

        sla_reasons: list[str] = []
        if intent:
            for constraint in intent.constraints:
                if constraint.metric not in predicted:
                    continue
                value = predicted[constraint.metric]
                violated = (constraint.operator == "<=" and value > constraint.value) or (
                    constraint.operator == ">=" and value < constraint.value
                )
                if violated and constraint.hard:
                    sla_reasons.append(
                        f"intent constraint {constraint.metric} {constraint.operator} {constraint.value:g} is violated"
                    )
        reasons.extend(sla_reasons)
        sla = "reject" if sla_reasons else "pass"

        energy = "pass"
        if intent and intent.energy_budget is not None and predicted.get("energy_load", 0.0) > intent.energy_budget:
            energy = "reject"
            reasons.append("predicted energy load exceeds the intent energy budget")
        elif predicted.get("energy_load", 0.0) > observation.energy_load * 1.20 + 0.03:
            energy = "warning"

        if uncertainty.level == "critical":
            uncertainty_status = "reject"
            reasons.append("combined uncertainty is critical")
        elif uncertainty.level == "high":
            uncertainty_status = "warning"
        else:
            uncertainty_status = "pass"

        interference = "not_evaluated"
        if interference_reason:
            interference = "reject"
            reasons.append(interference_reason)

        return CriticAssessment(
            safety=safety,
            sla=sla,
            energy=energy,
            stability=stability,
            uncertainty=uncertainty_status,
            interference=interference,
            reasons=reasons,
        )

    @staticmethod
    def rejected(assessment: CriticAssessment) -> bool:
        return any(
            value == "reject"
            for value in (
                assessment.safety,
                assessment.sla,
                assessment.energy,
                assessment.stability,
                assessment.uncertainty,
                assessment.interference,
            )
        )
