"""Typed domain models for Agentic-RAN."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SliceType(str, Enum):
    """Supported 5G service classes."""

    EMBB = "eMBB"
    URLLC = "URLLC"
    MMTC = "mMTC"


class NetworkObservation(BaseModel):
    """A compact RAN KPI snapshot used by the policy-selection agent."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    cell_id: str = Field(default="cell-001", min_length=1, max_length=64)
    slice_type: SliceType = SliceType.EMBB
    prb_utilization: float = Field(ge=0.0, le=1.5, description="Physical resource-block utilization")
    active_users: int = Field(ge=0, le=100_000)
    downlink_mbps: float = Field(ge=0.0, le=100_000.0)
    uplink_mbps: float = Field(ge=0.0, le=100_000.0)
    latency_ms: float = Field(ge=0.0, le=10_000.0)
    jitter_ms: float = Field(ge=0.0, le=10_000.0)
    packet_loss_pct: float = Field(ge=0.0, le=100.0)
    throughput_demand_mbps: float = Field(ge=0.0, le=100_000.0)
    energy_load: float = Field(ge=0.0, le=1.5)
    handover_failure_pct: float = Field(ge=0.0, le=100.0)
    rsrp_dbm: float = Field(ge=-160.0, le=-20.0)
    sinr_db: float = Field(ge=-30.0, le=80.0)

    @field_validator("timestamp")
    @classmethod
    def ensure_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    def feature_record(self) -> dict[str, Any]:
        """Return the stable feature schema consumed by the trained proposer."""

        return {
            "slice_type": str(self.slice_type),
            "prb_utilization": self.prb_utilization,
            "active_users": self.active_users,
            "downlink_mbps": self.downlink_mbps,
            "uplink_mbps": self.uplink_mbps,
            "latency_ms": self.latency_ms,
            "jitter_ms": self.jitter_ms,
            "packet_loss_pct": self.packet_loss_pct,
            "throughput_demand_mbps": self.throughput_demand_mbps,
            "energy_load": self.energy_load,
            "handover_failure_pct": self.handover_failure_pct,
            "rsrp_dbm": self.rsrp_dbm,
            "sinr_db": self.sinr_db,
        }


class CandidateEvaluation(BaseModel):
    """Evaluation result for one candidate network policy."""

    policy: str
    proposal_probability: float = Field(ge=0.0, le=1.0)
    utility_score: float = Field(ge=0.0, le=100.0)
    safe: bool
    safety_reasons: list[str] = Field(default_factory=list)
    predicted_kpis: dict[str, float]
    rationale: str


class DecisionTrace(BaseModel):
    """Human-readable trace of the agentic decision cycle."""

    observation_summary: str
    proposed_policies: list[str]
    rejected_policies: list[str]
    selected_policy: str
    critique: str


class PolicyDecision(BaseModel):
    """Final policy-selection response."""

    decision_id: str
    created_at: datetime
    selected_policy: str
    confidence: float = Field(ge=0.0, le=1.0)
    safety_override: bool
    action: dict[str, Any]
    expected_kpis: dict[str, float]
    explanation: str
    candidates: list[CandidateEvaluation]
    trace: DecisionTrace
    model_version: str


class BatchDecisionRequest(BaseModel):
    observations: list[NetworkObservation] = Field(min_length=1, max_length=500)
