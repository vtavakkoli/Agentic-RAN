"""Pluggable RAN world models for short-horizon counterfactual planning."""

from __future__ import annotations

import json
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from agentic_ran.domain import NetworkObservation
from agentic_ran.policies import PolicyDefinition


@dataclass(frozen=True, slots=True)
class TwinPrediction:
    trajectory: list[dict[str, float]]
    uncertainty: float
    model_name: str

    @property
    def final(self) -> dict[str, float]:
        return self.trajectory[-1] if self.trajectory else {}


class RANWorldModel(ABC):
    @abstractmethod
    def predict(self, observation: NetworkObservation, policy: PolicyDefinition, horizon: int = 3) -> TwinPrediction:
        raise NotImplementedError


class SurrogateWorldModel(RANWorldModel):
    """Transparent bounded model used as the offline-safe default."""

    def predict(self, observation: NetworkObservation, policy: PolicyDefinition, horizon: int = 3) -> TwinPrediction:
        state = {
            "latency_ms": observation.latency_ms,
            "packet_loss_pct": observation.packet_loss_pct,
            "downlink_mbps": observation.downlink_mbps,
            "energy_load": observation.energy_load,
            "handover_failure_pct": observation.handover_failure_pct,
            "sinr_db": observation.sinr_db,
            "prb_utilization": observation.prb_utilization,
        }
        impact = policy.impact
        trajectory: list[dict[str, float]] = []
        for step in range(max(1, horizon)):
            gain = 1.0 / (1.0 + 0.55 * step)
            state = {
                "latency_ms": max(0.2, state["latency_ms"] * (1.0 + (impact.get("latency_factor", 1.0) - 1.0) * gain)),
                "packet_loss_pct": max(
                    0.0, state["packet_loss_pct"] * (1.0 + (impact.get("loss_factor", 1.0) - 1.0) * gain)
                ),
                "downlink_mbps": max(
                    0.0, state["downlink_mbps"] * (1.0 + (impact.get("throughput_factor", 1.0) - 1.0) * gain)
                ),
                "energy_load": max(
                    0.0, min(1.5, state["energy_load"] * (1.0 + (impact.get("energy_factor", 1.0) - 1.0) * gain))
                ),
                "handover_failure_pct": max(
                    0.0,
                    state["handover_failure_pct"] * (1.0 + (impact.get("handover_factor", 1.0) - 1.0) * gain),
                ),
                "sinr_db": state["sinr_db"] + impact.get("sinr_delta", 0.0) * gain,
                "prb_utilization": max(
                    0.0,
                    min(
                        1.5,
                        state["prb_utilization"] + float(policy.action.get("prb_share_delta", 0.0)) * gain,
                    ),
                ),
            }
            trajectory.append({key: round(value, 6) for key, value in state.items()})
        uncertainty = min(0.55, 0.10 + 0.05 * max(0, horizon - 1))
        return TwinPrediction(trajectory=trajectory, uncertainty=uncertainty, model_name="bounded-surrogate-v2")


class TraceReplayWorldModel(RANWorldModel):
    """Counterfactual adapter that uses recorded per-policy deltas when available."""

    def __init__(self, policy_deltas: dict[str, dict[str, float]], fallback: RANWorldModel | None = None):
        self.policy_deltas = policy_deltas
        self.fallback = fallback or SurrogateWorldModel()

    def predict(self, observation: NetworkObservation, policy: PolicyDefinition, horizon: int = 3) -> TwinPrediction:
        deltas = self.policy_deltas.get(policy.name)
        if not deltas:
            return self.fallback.predict(observation, policy, horizon)
        base = {
            "latency_ms": observation.latency_ms,
            "packet_loss_pct": observation.packet_loss_pct,
            "downlink_mbps": observation.downlink_mbps,
            "energy_load": observation.energy_load,
            "handover_failure_pct": observation.handover_failure_pct,
            "sinr_db": observation.sinr_db,
            "prb_utilization": observation.prb_utilization,
        }
        trajectory: list[dict[str, float]] = []
        state = dict(base)
        for step in range(max(1, horizon)):
            decay = 1.0 / (step + 1)
            for key, delta in deltas.items():
                if key in state:
                    state[key] += float(delta) * decay
            state["prb_utilization"] = min(1.5, max(0.0, state["prb_utilization"]))
            state["energy_load"] = min(1.5, max(0.0, state["energy_load"]))
            trajectory.append({key: round(value, 6) for key, value in state.items()})
        return TwinPrediction(trajectory=trajectory, uncertainty=0.24, model_name="trace-replay-v1")


class ExternalWorldModelAdapter(RANWorldModel):
    """JSON bridge for ns-3, a calibrated simulator, or an external RAN twin.

    The remote service is expected to return ``{"trajectory": [...], "uncertainty": 0.2}``.
    """

    def __init__(self, endpoint: str, model_name: str, timeout_seconds: float = 2.0):
        self.endpoint = endpoint
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds

    def predict(self, observation: NetworkObservation, policy: PolicyDefinition, horizon: int = 3) -> TwinPrediction:
        payload = json.dumps(
            {
                "observation": observation.model_dump(mode="json"),
                "policy": {"name": policy.name, "action": policy.action, "impact": policy.impact},
                "horizon": horizon,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310 - configured endpoint
            data: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        trajectory = [{key: float(value) for key, value in row.items()} for row in data["trajectory"]]
        return TwinPrediction(
            trajectory=trajectory,
            uncertainty=float(data.get("uncertainty", 0.35)),
            model_name=str(data.get("model_name", self.model_name)),
        )


class NS3WorldModelAdapter(ExternalWorldModelAdapter):
    def __init__(self, endpoint: str = "http://127.0.0.1:8091/predict"):
        super().__init__(endpoint, "ns3-bridge")


class SrsRANDigitalTwinAdapter(ExternalWorldModelAdapter):
    def __init__(self, endpoint: str = "http://127.0.0.1:8092/predict"):
        super().__init__(endpoint, "srsran-digital-twin-bridge")
