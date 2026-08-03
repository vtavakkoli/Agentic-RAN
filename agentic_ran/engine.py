"""Agentic propose-evaluate-guard-select loop for RAN policy control."""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone

from agentic_ran.domain import CandidateEvaluation, DecisionTrace, NetworkObservation, PolicyDecision
from agentic_ran.model import PolicyProposer
from agentic_ran.policies import PolicyDefinition


_SLA = {
    "URLLC": {"latency": 10.0, "loss": 0.10, "weights": (0.43, 0.27, 0.12, 0.08, 0.10)},
    "eMBB": {"latency": 40.0, "loss": 0.80, "weights": (0.18, 0.14, 0.40, 0.10, 0.18)},
    "mMTC": {"latency": 110.0, "loss": 1.20, "weights": (0.14, 0.30, 0.14, 0.22, 0.20)},
}


class AgenticPolicyEngine:
    """Select policies with a learned proposer and deterministic safety critic."""

    def __init__(self, proposer: PolicyProposer, policies: dict[str, PolicyDefinition], top_k: int = 4):
        self.proposer = proposer
        self.policies = policies
        self.top_k = max(1, min(top_k, len(policies)))

    def decide(self, observation: NetworkObservation) -> PolicyDecision:
        record = observation.feature_record()
        proposals = self.proposer.top_k(record, self.top_k)
        proposal_map = dict(proposals)
        candidate_names = [name for name, _ in proposals if name in self.policies]
        for mandatory in ("balanced", self._heuristic_candidate(observation)):
            if mandatory in self.policies and mandatory not in candidate_names:
                candidate_names.append(mandatory)

        evaluations = [
            self._evaluate(observation, self.policies[name], proposal_map.get(name, 0.01))
            for name in candidate_names
        ]
        safe_candidates = [candidate for candidate in evaluations if candidate.safe]
        if not safe_candidates:
            fallback = self._evaluate(observation, self.policies["balanced"], proposal_map.get("balanced", 0.01), force_safe=True)
            evaluations.append(fallback)
            safe_candidates = [fallback]

        selected = max(
            safe_candidates,
            key=lambda candidate: candidate.utility_score + 8.0 * candidate.proposal_probability,
        )
        top_proposal = proposals[0][0] if proposals else "balanced"
        safety_override = top_proposal != selected.policy and any(
            candidate.policy == top_proposal and not candidate.safe for candidate in evaluations
        )
        confidence = self._confidence(selected, safe_candidates)
        rejected = [candidate.policy for candidate in evaluations if not candidate.safe]
        critique = self._critique(selected, rejected, safety_override)
        explanation = (
            f"Selected '{selected.policy}' with utility {selected.utility_score:.1f}/100. "
            f"{selected.rationale} {critique}"
        )
        trace = DecisionTrace(
            observation_summary=self._observation_summary(observation),
            proposed_policies=[name for name, _ in proposals],
            rejected_policies=rejected,
            selected_policy=selected.policy,
            critique=critique,
        )
        return PolicyDecision(
            decision_id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc),
            selected_policy=selected.policy,
            confidence=confidence,
            safety_override=safety_override,
            action=self.policies[selected.policy].action,
            expected_kpis=selected.predicted_kpis,
            explanation=explanation,
            candidates=sorted(evaluations, key=lambda item: item.utility_score, reverse=True),
            trace=trace,
            model_version=self.proposer.metadata.version,
        )

    def _evaluate(
        self,
        observation: NetworkObservation,
        policy: PolicyDefinition,
        proposal_probability: float,
        force_safe: bool = False,
    ) -> CandidateEvaluation:
        impact = policy.impact
        predicted = {
            "latency_ms": max(0.2, observation.latency_ms * impact.get("latency_factor", 1.0)),
            "packet_loss_pct": max(0.0, observation.packet_loss_pct * impact.get("loss_factor", 1.0)),
            "downlink_mbps": max(0.0, observation.downlink_mbps * impact.get("throughput_factor", 1.0)),
            "energy_load": max(0.0, observation.energy_load * impact.get("energy_factor", 1.0)),
            "handover_failure_pct": max(0.0, observation.handover_failure_pct * impact.get("handover_factor", 1.0)),
            "sinr_db": observation.sinr_db + impact.get("sinr_delta", 0.0),
            "prb_utilization": max(0.0, min(1.5, observation.prb_utilization + float(policy.action.get("prb_share_delta", 0.0)))),
        }
        reasons = [] if force_safe else self._safety_reasons(observation, policy, predicted)
        score = self._utility_score(observation, predicted)
        if reasons:
            score = max(0.0, score - min(55.0, 18.0 * len(reasons)))
        rationale = self._candidate_rationale(observation, policy, predicted)
        return CandidateEvaluation(
            policy=policy.name,
            proposal_probability=max(0.0, min(1.0, proposal_probability)),
            utility_score=score,
            safe=not reasons,
            safety_reasons=reasons,
            predicted_kpis={key: round(value, 4) for key, value in predicted.items()},
            rationale=rationale,
        )

    def _utility_score(self, observation: NetworkObservation, predicted: dict[str, float]) -> float:
        sla = _SLA[str(observation.slice_type)]
        latency_target = float(sla["latency"])
        loss_target = float(sla["loss"])
        latency_score = math.exp(-max(0.0, predicted["latency_ms"] - latency_target) / max(latency_target, 1.0))
        loss_score = math.exp(-max(0.0, predicted["packet_loss_pct"] - loss_target) / max(loss_target, 0.1))
        throughput_ratio = predicted["downlink_mbps"] / max(observation.throughput_demand_mbps, 1.0)
        throughput_score = min(1.0, throughput_ratio)
        stability_score = math.exp(-predicted["handover_failure_pct"] / 4.0)
        energy_score = max(0.0, min(1.0, 1.15 - predicted["energy_load"]))
        weights = sla["weights"]
        raw = sum(
            weight * score
            for weight, score in zip(
                weights,
                (latency_score, loss_score, throughput_score, stability_score, energy_score),
                strict=True,
            )
        )
        overload_penalty = max(0.0, predicted["prb_utilization"] - 1.0) * 22.0
        return round(max(0.0, min(100.0, raw * 100.0 - overload_penalty)), 4)

    def _safety_reasons(
        self,
        observation: NetworkObservation,
        policy: PolicyDefinition,
        predicted: dict[str, float],
    ) -> list[str]:
        reasons: list[str] = []
        tx_delta = float(policy.action.get("tx_power_db_delta", 0.0))
        if not -3.0 <= tx_delta <= 3.0:
            reasons.append("transmit-power adjustment exceeds the configured ±3 dB guardrail")
        if policy.name == "energy_saver" and str(observation.slice_type) == "URLLC" and (
            observation.latency_ms > 8.0 or observation.packet_loss_pct > 0.1
        ):
            reasons.append("energy saving is blocked while URLLC is near or outside its SLA")
        if predicted["prb_utilization"] > 1.18:
            reasons.append("predicted PRB pressure exceeds the safe operating envelope")
        if predicted["packet_loss_pct"] > max(8.0, observation.packet_loss_pct * 1.15 + 0.2):
            reasons.append("predicted packet loss degrades beyond the allowed margin")
        if predicted["sinr_db"] < -6.0:
            reasons.append("predicted SINR is below the minimum service threshold")
        if policy.name == "throughput_boost" and observation.prb_utilization > 0.96:
            reasons.append("throughput boost is unsafe during critical congestion")
        if policy.name == "energy_saver" and observation.rsrp_dbm < -110.0:
            reasons.append("power reduction is unsafe for weak-coverage users")
        return reasons

    @staticmethod
    def _heuristic_candidate(observation: NetworkObservation) -> str:
        if observation.rsrp_dbm < -112.0 or observation.sinr_db < 1.5 or observation.handover_failure_pct > 5.0:
            return "coverage_recovery"
        if str(observation.slice_type) == "URLLC" and (observation.latency_ms > 12.0 or observation.packet_loss_pct > 0.35):
            return "latency_guard"
        if observation.prb_utilization > 0.93 or observation.packet_loss_pct > 2.2:
            return "congestion_relief"
        if str(observation.slice_type) == "mMTC" and observation.active_users > 700:
            return "massive_iot_access"
        if str(observation.slice_type) == "eMBB" and observation.throughput_demand_mbps > observation.downlink_mbps * 1.15:
            return "throughput_boost"
        if observation.prb_utilization < 0.38 and observation.energy_load < 0.48:
            return "energy_saver"
        return "balanced"

    @staticmethod
    def _confidence(selected: CandidateEvaluation, safe_candidates: list[CandidateEvaluation]) -> float:
        ordered = sorted(safe_candidates, key=lambda item: item.utility_score, reverse=True)
        margin = selected.utility_score - (ordered[1].utility_score if len(ordered) > 1 else 0.0)
        confidence = 0.48 + 0.34 * selected.proposal_probability + min(0.18, max(0.0, margin / 100.0))
        return round(max(0.0, min(0.99, confidence)), 4)

    @staticmethod
    def _candidate_rationale(
        observation: NetworkObservation,
        policy: PolicyDefinition,
        predicted: dict[str, float],
    ) -> str:
        latency_delta = observation.latency_ms - predicted["latency_ms"]
        throughput_delta = predicted["downlink_mbps"] - observation.downlink_mbps
        loss_delta = observation.packet_loss_pct - predicted["packet_loss_pct"]
        return (
            f"{policy.description} Estimated changes: latency {latency_delta:+.1f} ms, "
            f"downlink {throughput_delta:+.1f} Mbps, packet loss {loss_delta:+.2f} percentage points."
        )

    @staticmethod
    def _observation_summary(observation: NetworkObservation) -> str:
        return (
            f"{observation.cell_id}/{observation.slice_type}: PRB={observation.prb_utilization:.2f}, "
            f"latency={observation.latency_ms:.1f} ms, loss={observation.packet_loss_pct:.2f}%, "
            f"demand={observation.throughput_demand_mbps:.1f} Mbps, SINR={observation.sinr_db:.1f} dB"
        )

    @staticmethod
    def _critique(selected: CandidateEvaluation, rejected: list[str], safety_override: bool) -> str:
        if safety_override:
            return f"The highest-probability proposal was rejected by safety checks; '{selected.policy}' is the best safe alternative."
        if rejected:
            return f"The critic rejected {', '.join(rejected)} and retained the highest-utility safe candidate."
        return "All evaluated candidates passed guardrails; the highest combined utility and proposal confidence won."
