"""Execution modes, E2SM-RC bridge actuation, canary gating, and rollback."""

from __future__ import annotations

import asyncio
import json
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol

from agentic_ran.domain import ActionResult, ControlEnvelope, ExecutionMode, NetworkObservation, PolicyDecision
from agentic_ran.policies import PolicyDefinition
from agentic_ran.twin import RANWorldModel, SurrogateWorldModel


class Actuator(ABC):
    @abstractmethod
    async def execute(self, decision: PolicyDecision, observation: NetworkObservation) -> ActionResult:
        raise NotImplementedError


class RecommendationActuator(Actuator):
    async def execute(self, decision: PolicyDecision, observation: NetworkObservation) -> ActionResult:
        return ActionResult(decision_id=decision.decision_id, mode=ExecutionMode.RECOMMEND, attempted=False, executed=False, success=True, status="recommendation_only", message="Decision generated without network actuation.")


class ShadowActuator(Actuator):
    async def execute(self, decision: PolicyDecision, observation: NetworkObservation) -> ActionResult:
        return ActionResult(decision_id=decision.decision_id, mode=ExecutionMode.SHADOW, attempted=True, executed=False, success=True, status="shadow_recorded", message="Action passed the shadow pipeline but was not sent to the RAN.", transport_response={"would_execute": decision.action, "cell_id": observation.cell_id})


class SimulatedActuator(Actuator):
    def __init__(self, policies: dict[str, PolicyDefinition], world_model: RANWorldModel | None = None):
        self.policies = policies
        self.world_model = world_model or SurrogateWorldModel()

    async def execute(self, decision: PolicyDecision, observation: NetworkObservation) -> ActionResult:
        prediction = self.world_model.predict(observation, self.policies[decision.selected_policy], horizon=1)
        return ActionResult(decision_id=decision.decision_id, mode=ExecutionMode.SIMULATED, attempted=True, executed=True, success=True, status="simulated", message="Action applied to the configured RAN world model.", transport_response={"predicted_kpis": prediction.final, "world_model": prediction.model_name})


class E2ControlTransport(Protocol):
    def send_control(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class HttpE2BridgeTransport:
    """HTTP bridge to a standards-capable xApp/RIC E2 implementation."""

    def __init__(self, endpoint: str, timeout_seconds: float = 3.0):
        self.endpoint = endpoint.rstrip("/") + "/v1/e2/control"
        self.timeout_seconds = timeout_seconds

    def send_control(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode("utf-8"), headers={"content-type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))


class E2RCActuator(Actuator):
    """Maps bounded Agentic-RAN actions to a generic E2SM-RC Style-2 bridge payload."""

    def __init__(self, transport: E2ControlTransport, mode: ExecutionMode = ExecutionMode.ACTIVE):
        self.transport = transport
        self.mode = mode

    @staticmethod
    def to_control_payload(decision: PolicyDecision, observation: NetworkObservation) -> dict[str, Any]:
        return {"service_model": "E2SM-RC", "control_style_type": 2, "cell_id": observation.cell_id, "slice_type": observation.slice_type, "decision_id": decision.decision_id, "policy": decision.selected_policy, "parameters": decision.action}

    async def execute(self, decision: PolicyDecision, observation: NetworkObservation) -> ActionResult:
        payload = self.to_control_payload(decision, observation)
        try:
            response = await asyncio.to_thread(self.transport.send_control, payload)
        except Exception as exc:
            return ActionResult(decision_id=decision.decision_id, mode=self.mode, attempted=True, executed=False, success=False, status="transport_error", message=f"E2 bridge rejected or failed the control request: {exc}")
        accepted = bool(response.get("accepted", response.get("success", True)))
        return ActionResult(decision_id=decision.decision_id, mode=self.mode, attempted=True, executed=accepted, success=accepted, status="executed" if accepted else "rejected", message="Control request accepted by the E2 bridge." if accepted else "Control request rejected by the E2 bridge.", transport_response=response)


class GuardedActuator(Actuator):
    """Final authority gate separating decision generation from network actuation."""

    def __init__(self, delegate: Actuator, envelope: ControlEnvelope):
        self.delegate = delegate
        self.envelope = envelope

    async def execute(self, decision: PolicyDecision, observation: NetworkObservation) -> ActionResult:
        mode = self.envelope.mode
        if mode == ExecutionMode.RECOMMEND:
            return await RecommendationActuator().execute(decision, observation)
        if mode == ExecutionMode.SHADOW:
            return await ShadowActuator().execute(decision, observation)
        if self.envelope.require_operator_approval:
            return ActionResult(decision_id=decision.decision_id, mode=mode, attempted=False, executed=False, success=False, status="approval_required", message="Operator approval is required by the control envelope.")
        if not decision.approved_for_execution:
            return self._blocked(decision, mode, "decision is not approved for autonomous execution")
        if decision.confidence < self.envelope.min_confidence:
            return self._blocked(decision, mode, "decision confidence is below the execution threshold")
        if decision.uncertainty and decision.uncertainty.combined > self.envelope.max_uncertainty:
            return self._blocked(decision, mode, "combined uncertainty exceeds the execution threshold")
        if decision.ood_score > self.envelope.max_ood_score:
            return self._blocked(decision, mode, "observation is too far out of distribution")
        if self.envelope.allowed_policies is not None and decision.selected_policy not in self.envelope.allowed_policies:
            return self._blocked(decision, mode, "policy is not allowed by the execution envelope")
        if mode == ExecutionMode.CANARY and observation.cell_id not in self.envelope.canary_cells:
            return self._blocked(decision, mode, "cell is outside the canary allow-list")
        result = await self.delegate.execute(decision, observation)
        return result.model_copy(update={"mode": mode})

    @staticmethod
    def _blocked(decision: PolicyDecision, mode: ExecutionMode, reason: str) -> ActionResult:
        return ActionResult(decision_id=decision.decision_id, mode=mode, attempted=False, executed=False, success=False, status="blocked", message=reason)


@dataclass(slots=True)
class RollbackPolicy:
    throughput_drop_pct: float = 15.0
    packet_loss_increase_pct: float = 5.0
    latency_increase_pct: float = 20.0


class RollbackManager:
    def __init__(self, policy: RollbackPolicy | None = None):
        self.policy = policy or RollbackPolicy()
        self._before: dict[str, NetworkObservation] = {}
        self._decisions: dict[str, PolicyDecision] = {}

    def arm(self, observation: NetworkObservation, decision: PolicyDecision) -> None:
        self._before[observation.cell_id] = observation
        self._decisions[observation.cell_id] = decision

    def evaluate(self, after: NetworkObservation) -> tuple[bool, list[str]]:
        before = self._before.get(after.cell_id)
        if before is None:
            return False, []
        reasons: list[str] = []
        if before.downlink_mbps > 0:
            drop = (before.downlink_mbps - after.downlink_mbps) / before.downlink_mbps * 100.0
            if drop > self.policy.throughput_drop_pct:
                reasons.append(f"throughput dropped {drop:.1f}%")
        if after.packet_loss_pct - before.packet_loss_pct > self.policy.packet_loss_increase_pct:
            reasons.append("packet loss increased beyond the rollback threshold")
        if before.latency_ms > 0:
            increase = (after.latency_ms - before.latency_ms) / before.latency_ms * 100.0
            if increase > self.policy.latency_increase_pct:
                reasons.append(f"latency increased {increase:.1f}%")
        return bool(reasons), reasons
