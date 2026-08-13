"""Application service for training, decisions, coordination, control-loop composition, and governance."""

from __future__ import annotations

from pathlib import Path

from agentic_ran.actuation import (
    E2RCActuator,
    GuardedActuator,
    HttpE2BridgeTransport,
    RecommendationActuator,
    ShadowActuator,
    SimulatedActuator,
)
from agentic_ran.audit import HashChainAuditLog
from agentic_ran.config import Settings
from agentic_ran.control_loop import AgenticControlLoop
from agentic_ran.coordinator import MultiCellCoordinator
from agentic_ran.data import generate_dataset, load_dataset, write_dataset
from agentic_ran.domain import ControlEnvelope, ExecutionMode, NetworkObservation, PolicyDecision, SLAIntent
from agentic_ran.engine import AgenticPolicyEngine
from agentic_ran.model import PolicyProposer, train_policy_proposer, write_training_metrics
from agentic_ran.policies import PolicyDefinition, load_policy_catalog
from agentic_ran.telemetry import TelemetryProvider
from agentic_ran.twin import RANWorldModel, SurrogateWorldModel


class PolicyService:
    """Owns model, policies, planning, audit, coordination, and execution composition."""

    def __init__(
        self,
        settings: Settings,
        proposer: PolicyProposer,
        policies: dict[str, PolicyDefinition],
        world_model: RANWorldModel | None = None,
    ):
        self.settings = settings
        self.proposer = proposer
        self.policies = policies
        self.world_model = world_model or SurrogateWorldModel()
        self.engine = AgenticPolicyEngine(
            proposer,
            policies,
            top_k=settings.top_k,
            world_model=self.world_model,
            planning_horizon=settings.planning_horizon,
        )
        audit_path = settings.audit_path
        if audit_path == Path("results/audit/decisions.jsonl") and settings.results_dir != Path("results"):
            audit_path = settings.results_dir / "audit" / "decisions.jsonl"
        self.audit = HashChainAuditLog(audit_path)
        self.coordinator = MultiCellCoordinator()

    @classmethod
    def load(cls, settings: Settings | None = None, world_model: RANWorldModel | None = None) -> "PolicyService":
        resolved = settings or Settings.from_env()
        if not resolved.model_path.exists():
            if not resolved.auto_bootstrap:
                raise FileNotFoundError(f"Model not found: {resolved.model_path}")
            bootstrap(resolved)
        proposer = PolicyProposer.load(resolved.model_path)
        policies = load_policy_catalog(resolved.policy_config_path)
        missing = sorted(set(proposer.metadata.classes).difference(policies))
        if missing:
            raise ValueError(f"Model contains policies missing from catalog: {missing}")
        return cls(resolved, proposer, policies, world_model=world_model)

    def decide(self, observation: NetworkObservation, intent: SLAIntent | str | None = None) -> PolicyDecision:
        decision = self.engine.decide(observation, intent=intent)
        decision = decision.model_copy(update={"execution_mode": self.settings.execution_mode})
        audit_hash = self.audit.append(
            "decision",
            {"observation": observation.model_dump(mode="json"), "decision": decision.model_dump(mode="json")},
        )
        return decision.model_copy(update={"audit_hash": audit_hash})

    def coordinate(
        self,
        observations: list[NetworkObservation],
        topology: dict[str, list[str]] | None = None,
        intent: SLAIntent | str | None = None,
    ):
        by_cell = {item.cell_id: item for item in observations}
        decisions = {cell: self.decide(observation, intent=intent) for cell, observation in by_cell.items()}
        return self.coordinator.coordinate(by_cell, decisions, topology=topology)

    def actuator(self, mode: ExecutionMode | None = None) -> GuardedActuator:
        resolved_mode = mode or self.settings.execution_mode
        envelope = ControlEnvelope(mode=resolved_mode, canary_cells=set(self.settings.canary_cells))
        if resolved_mode == ExecutionMode.RECOMMEND:
            delegate = RecommendationActuator()
        elif resolved_mode == ExecutionMode.SHADOW:
            delegate = ShadowActuator()
        elif resolved_mode == ExecutionMode.SIMULATED:
            delegate = SimulatedActuator(self.policies, self.world_model)
        else:
            delegate = E2RCActuator(HttpE2BridgeTransport(self.settings.e2_bridge_url), mode=resolved_mode)
        return GuardedActuator(delegate, envelope)

    def control_loop(self, telemetry: TelemetryProvider, mode: ExecutionMode | None = None) -> AgenticControlLoop:
        return AgenticControlLoop(self, telemetry, self.actuator(mode), audit=self.audit)


def bootstrap(settings: Settings | None = None) -> tuple[Path, dict]:
    resolved = settings or Settings.from_env()
    if not resolved.dataset_path.exists():
        write_dataset(generate_dataset(seed=resolved.random_seed), resolved.dataset_path)
    frame = load_dataset(resolved.dataset_path)
    proposer, metrics = train_policy_proposer(frame, seed=resolved.random_seed)
    proposer.save(resolved.model_path)
    write_training_metrics(metrics, resolved.results_dir / "training_metrics.json")
    return resolved.model_path, metrics
