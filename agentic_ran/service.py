"""Application service for bootstrapping, training, and decisions."""

from __future__ import annotations

from pathlib import Path

from agentic_ran.config import Settings
from agentic_ran.data import generate_dataset, load_dataset, write_dataset
from agentic_ran.domain import NetworkObservation, PolicyDecision
from agentic_ran.engine import AgenticPolicyEngine
from agentic_ran.model import PolicyProposer, train_policy_proposer, write_training_metrics
from agentic_ran.policies import PolicyDefinition, load_policy_catalog


class PolicyService:
    """Own model, policy catalog, and decision engine lifecycle."""

    def __init__(self, settings: Settings, proposer: PolicyProposer, policies: dict[str, PolicyDefinition]):
        self.settings = settings
        self.proposer = proposer
        self.policies = policies
        self.engine = AgenticPolicyEngine(proposer, policies, top_k=settings.top_k)

    @classmethod
    def load(cls, settings: Settings | None = None) -> "PolicyService":
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
        return cls(resolved, proposer, policies)

    def decide(self, observation: NetworkObservation) -> PolicyDecision:
        return self.engine.decide(observation)


def bootstrap(settings: Settings | None = None) -> tuple[Path, dict]:
    resolved = settings or Settings.from_env()
    if not resolved.dataset_path.exists():
        write_dataset(generate_dataset(seed=resolved.random_seed), resolved.dataset_path)
    frame = load_dataset(resolved.dataset_path)
    proposer, metrics = train_policy_proposer(frame, seed=resolved.random_seed)
    proposer.save(resolved.model_path)
    write_training_metrics(metrics, resolved.results_dir / "training_metrics.json")
    return resolved.model_path, metrics
