"""Typed domain models shared across the Agentic-RAN control plane."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SliceType(str, Enum):
    EMBB = "eMBB"
    URLLC = "URLLC"
    MMTC = "mMTC"


class ExecutionMode(str, Enum):
    """Control authority level. Live execution is opt-in and gated."""

    RECOMMEND = "recommend"
    SHADOW = "shadow"
    SIMULATED = "simulated"
    CANARY = "canary"
    ACTIVE = "active"


class NetworkObservation(BaseModel):
    """Normalized RAN KPI snapshot consumed by every Agentic-RAN backend."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    cell_id: str = Field(default="cell-001", min_length=1, max_length=128)
    slice_type: SliceType = SliceType.EMBB
    prb_utilization: float = Field(ge=0.0, le=1.5)
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
    source: str = Field(default="api", min_length=1, max_length=64)
    sequence: int | None = Field(default=None, ge=0)
    telemetry_age_ms: float = Field(default=0.0, ge=0.0)
    telemetry_completeness: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("timestamp")
    @classmethod
    def ensure_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    def feature_record(self) -> dict[str, Any]:
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


class SLAConstraint(BaseModel):
    metric: str
    operator: Literal["<=", ">="]
    value: float
    hard: bool = True


class SLAIntent(BaseModel):
    """Operator intent compiled into explicit constraints and optimization weights."""

    name: str = "default"
    slice_type: SliceType | None = None
    constraints: list[SLAConstraint] = Field(default_factory=list)
    weights: dict[str, float] = Field(default_factory=dict)
    risk_tolerance: float = Field(default=0.25, ge=0.0, le=1.0)
    energy_budget: float | None = Field(default=None, ge=0.0, le=1.5)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UncertaintyAssessment(BaseModel):
    proposal_entropy: float = Field(ge=0.0, le=1.0)
    ood_score: float = Field(ge=0.0, le=1.0)
    telemetry_penalty: float = Field(ge=0.0, le=1.0)
    twin_uncertainty: float = Field(ge=0.0, le=1.0)
    combined: float = Field(ge=0.0, le=1.0)
    level: Literal["low", "moderate", "high", "critical"]


class CriticAssessment(BaseModel):
    safety: str = "pass"
    sla: str = "pass"
    energy: str = "pass"
    stability: str = "pass"
    uncertainty: str = "pass"
    interference: str = "not_evaluated"
    reasons: list[str] = Field(default_factory=list)


class CandidateEvaluation(BaseModel):
    policy: str
    proposal_probability: float = Field(ge=0.0, le=1.0)
    utility_score: float = Field(ge=0.0, le=100.0)
    safe: bool
    safety_reasons: list[str] = Field(default_factory=list)
    predicted_kpis: dict[str, float]
    predicted_trajectory: list[dict[str, float]] = Field(default_factory=list)
    twin_uncertainty: float = Field(default=0.0, ge=0.0, le=1.0)
    critics: CriticAssessment = Field(default_factory=CriticAssessment)
    rationale: str


class DecisionTrace(BaseModel):
    observation_summary: str
    proposed_policies: list[str]
    rejected_policies: list[str]
    selected_policy: str
    critique: str
    stages: list[str] = Field(default_factory=list)
    pareto_policies: list[str] = Field(default_factory=list)
    intent_name: str = "default"


class PolicyDecision(BaseModel):
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
    execution_mode: ExecutionMode = ExecutionMode.RECOMMEND
    approved_for_execution: bool = False
    uncertainty: UncertaintyAssessment | None = None
    ood_score: float = Field(default=0.0, ge=0.0, le=1.0)
    intent: SLAIntent | None = None
    audit_hash: str | None = None


class ControlEnvelope(BaseModel):
    mode: ExecutionMode = ExecutionMode.RECOMMEND
    min_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    max_uncertainty: float = Field(default=0.55, ge=0.0, le=1.0)
    max_ood_score: float = Field(default=0.65, ge=0.0, le=1.0)
    allowed_policies: set[str] | None = None
    canary_cells: set[str] = Field(default_factory=set)
    require_operator_approval: bool = False


class ActionResult(BaseModel):
    decision_id: str
    mode: ExecutionMode
    attempted: bool
    executed: bool
    success: bool
    status: str
    message: str
    transport_response: dict[str, Any] = Field(default_factory=dict)
    rollback_required: bool = False


class BatchDecisionRequest(BaseModel):
    observations: list[NetworkObservation] = Field(min_length=1, max_length=500)


class CoordinationRequest(BaseModel):
    observations: list[NetworkObservation] = Field(min_length=2, max_length=256)
    topology: dict[str, list[str]] = Field(default_factory=dict)
    intent: SLAIntent | None = None
