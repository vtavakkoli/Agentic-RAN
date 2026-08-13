"""Multi-cell coordination and interference-aware conflict resolution."""

from __future__ import annotations

from dataclasses import dataclass

from agentic_ran.domain import NetworkObservation, PolicyDecision


@dataclass(frozen=True, slots=True)
class CoordinatedAction:
    cell_id: str
    policy: str
    approved: bool
    network_score: float
    reasons: tuple[str, ...]


class MultiCellCoordinator:
    """Resolve locally-good actions that are globally harmful to neighboring cells."""

    def coordinate(self, observations: dict[str, NetworkObservation], decisions: dict[str, PolicyDecision], topology: dict[str, list[str]] | None = None) -> list[CoordinatedAction]:
        graph = topology or {cell: [other for other in observations if other != cell] for cell in observations}
        output: list[CoordinatedAction] = []
        for cell_id, decision in decisions.items():
            observation = observations[cell_id]
            tx_delta = float(decision.action.get("tx_power_db_delta", 0.0))
            reasons: list[str] = []
            interference_cost = 0.0
            for neighbor_id in graph.get(cell_id, []):
                neighbor = observations.get(neighbor_id)
                if neighbor is None:
                    continue
                if tx_delta > 0.5:
                    sensitivity = max(0.0, (8.0 - neighbor.sinr_db) / 18.0) + max(0.0, (neighbor.prb_utilization - 0.78) * 1.8)
                    interference_cost += tx_delta * sensitivity * 8.0
                if decision.selected_policy == "throughput_boost" and neighbor.prb_utilization > 0.95:
                    reasons.append(f"neighbor {neighbor_id} is critically loaded")
            local_score = max((candidate.utility_score for candidate in decision.candidates if candidate.policy == decision.selected_policy), default=0.0)
            network_score = max(0.0, local_score - interference_cost)
            if interference_cost > 12.0:
                reasons.append("predicted inter-cell interference cost is excessive")
            if observation.handover_failure_pct > 6.0 and tx_delta < 0:
                reasons.append("power reduction conflicts with local mobility recovery")
            output.append(CoordinatedAction(cell_id=cell_id, policy=decision.selected_policy, approved=not reasons, network_score=round(network_score, 4), reasons=tuple(reasons)))
        return output
