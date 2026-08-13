"""Closed-loop orchestration: observe, forecast, decide, actuate, verify, audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from agentic_ran.actuation import Actuator, RollbackManager
from agentic_ran.audit import HashChainAuditLog
from agentic_ran.domain import ActionResult, NetworkObservation, PolicyDecision, SLAIntent
from agentic_ran.forecasting import Forecast, ForecastEnsemble, TelemetryHistory
from agentic_ran.telemetry import TelemetryProvider


@dataclass(frozen=True, slots=True)
class ControlStepResult:
    observation: NetworkObservation
    decision: PolicyDecision
    action_result: ActionResult
    forecast: Forecast | None
    audit_hash: str | None


class AgenticControlLoop:
    def __init__(self, service: Any, telemetry: TelemetryProvider, actuator: Actuator, audit: HashChainAuditLog | None = None, rollback: RollbackManager | None = None, forecaster: ForecastEnsemble | None = None):
        self.service = service
        self.telemetry = telemetry
        self.actuator = actuator
        self.audit = audit
        self.rollback = rollback or RollbackManager()
        self.history = TelemetryHistory()
        self.forecaster = forecaster or ForecastEnsemble()

    async def step(self, intent: SLAIntent | str | None = None) -> ControlStepResult:
        observation = await self.telemetry.observe()
        rollback_required, rollback_reasons = self.rollback.evaluate(observation)
        self.history.append(observation)
        forecast = self.forecaster.forecast(self.history, horizon=5) if len(self.history.values) >= 3 else None
        decision = self.service.decide(observation, intent=intent)
        if rollback_required and "balanced" in self.service.policies:
            decision = decision.model_copy(update={"selected_policy": "balanced", "action": self.service.policies["balanced"].action, "approved_for_execution": True, "safety_override": True, "explanation": decision.explanation + " Rollback guard forced the balanced fallback: " + "; ".join(rollback_reasons)})
        action_result = await self.actuator.execute(decision, observation)
        if action_result.executed:
            self.rollback.arm(observation, decision)
            self.service.engine.temporal_guard.record(observation.cell_id, decision.selected_policy)
        audit_hash = None
        if self.audit:
            audit_hash = self.audit.append("control_step", {"observation": observation.model_dump(mode="json"), "forecast": asdict(forecast) if forecast else None, "decision": decision.model_dump(mode="json"), "action_result": action_result.model_dump(mode="json")})
            decision = decision.model_copy(update={"audit_hash": audit_hash})
        return ControlStepResult(observation, decision, action_result, forecast, audit_hash)
