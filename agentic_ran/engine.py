"""Agentic plan-simulate-critic-guard-select loop for RAN policy control."""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone

from agentic_ran.domain import CandidateEvaluation, DecisionTrace, NetworkObservation, PolicyDecision, SLAIntent
from agentic_ran.intents import IntentCompiler, ParetoOptimizer
from agentic_ran.model import PolicyProposer
from agentic_ran.policies import PolicyDefinition
from agentic_ran.safety import CriticSuite, OODDetector, TemporalSafetyGuard, UncertaintyEstimator
from agentic_ran.twin import RANWorldModel, SurrogateWorldModel


_SLA = {
    "URLLC": {"latency": 10.0, "loss": 0.10, "weights": (0.43, 0.27, 0.12, 0.08, 0.10)},
    "eMBB": {"latency": 40.0, "loss": 0.80, "weights": (0.18, 0.14, 0.40, 0.10, 0.18)},
    "mMTC": {"latency": 110.0, "loss": 1.20, "weights": (0.14, 0.30, 0.14, 0.22, 0.20)},
}


class AgenticPolicyEngine:
    """Select bounded policies through planning, multiple critics, and deterministic guards."""

    def __init__(
        self,
        proposer: PolicyProposer,
        policies: dict[str, PolicyDefinition],
        top_k: int = 4,
        world_model: RANWorldModel | None = None,
        planning_horizon: int = 3,
        ood_detector: OODDetector | None = None,
        uncertainty_estimator: UncertaintyEstimator | None = None,
        temporal_guard: TemporalSafetyGuard | None = None,
        intent_compiler: IntentCompiler | None = None,
    ):
        self.proposer = proposer
        self.policies = policies
        self.top_k = max(1, min(top_k, len(policies)))
        self.world_model = world_model or SurrogateWorldModel()
        self.planning_horizon = max(1, planning_horizon)
        self.ood_detector = ood_detector or OODDetector()
        self.uncertainty_estimator = uncertainty_estimator or UncertaintyEstimator()
        self.temporal_guard = temporal_guard or TemporalSafetyGuard()
        self.intent_compiler = intent_compiler or IntentCompiler()
        self.pareto = ParetoOptimizer()

    def decide(self, observation: NetworkObservation, intent: SLAIntent | str | None = None) -> PolicyDecision:
        record = observation.feature_record()
        proposals = self.proposer.top_k(record, self.top_k)
        proposal_map = dict(proposals)
        candidate_names = [name for name, _ in proposals if name in self.policies]
        for mandatory in ("balanced", self._heuristic_candidate(observation)):
            if mandatory in self.policies and mandatory not in candidate_names:
                candidate_names.append(mandatory)

        compiled_intent = self.intent_compiler.compile(intent, str(observation.slice_type))
        ood_score = self.ood_detector.score(observation)
        proposal_probs = [probability for _, probability in proposals]
        evaluations = [
            self._evaluate(
                observation,
                self.policies[name],
                proposal_map.get(name, 0.01),
                proposal_probs,
                ood_score,
                compiled_intent,
            )
            for name in candidate_names
        ]
        safe_candidates = [candidate for candidate in evaluations if candidate.safe]
        if not safe_candidates:
            fallback = self._evaluate(
                observation,
                self.policies["balanced"],
                proposal_map.get("balanced", 0.01),
                proposal_probs,
                ood_score,
                compiled_intent,
                force_safe=True,
            )
            evaluations.append(fallback)
            safe_candidates = [fallback]

        selected = max(safe_candidates, key=lambda candidate: candidate.utility_score + 8.0 * candidate.proposal_probability)
        top_proposal = proposals[0][0] if proposals else "balanced"
        safety_override = top_proposal != selected.policy and any(
            candidate.policy == top_proposal and not candidate.safe for candidate in evaluations
        )
        uncertainty = self.uncertainty_estimator.assess(proposal_probs, observation, ood_score, selected.twin_uncertainty)
        confidence = self._confidence(selected, safe_candidates, uncertainty.combined)
        rejected = [candidate.policy for candidate in evaluations if not candidate.safe]
        critique = self._critique(selected, rejected, safety_override, uncertainty.level, ood_score)
        explanation = f"Selected '{selected.policy}' with utility {selected.utility_score:.1f}/100. {selected.rationale} {critique}"
        trace = DecisionTrace(
            observation_summary=self._observation_summary(observation),
            proposed_policies=[name for name, _ in proposals],
            rejected_policies=rejected,
            selected_policy=selected.policy,
            critique=critique,
            stages=["observe", "propose", "plan", "simulate", "critic", "guard", "pareto-rank", "select", "explain"],
            pareto_policies=self.pareto.front(evaluations),
            intent_name=compiled_intent.name,
        )
        approved = (
            selected.safe
            and uncertainty.level not in {"high", "critical"}
            and ood_score <= max(0.50, 0.78 - compiled_intent.risk_tolerance * 0.3)
            and confidence >= 0.55
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
            approved_for_execution=approved,
            uncertainty=uncertainty,
            ood_score=ood_score,
            intent=compiled_intent,
        )

    def _evaluate(self, observation, policy, proposal_probability, proposal_probabilities, ood_score, intent, force_safe=False):
        twin = self.world_model.predict(observation, policy, horizon=self.planning_horizon)
        predicted = twin.final
        base_reasons = [] if force_safe else self._safety_reasons(observation, policy, predicted)
        temporal_reasons = [] if force_safe else self.temporal_guard.assess(observation.cell_id, policy.name)
        provisional_uncertainty = self.uncertainty_estimator.assess(proposal_probabilities, observation, ood_score, twin.uncertainty)
        if not force_safe and ood_score > 0.72 and policy.name not in {"balanced", "coverage_recovery"}:
            base_reasons.append("aggressive policy is blocked for a strongly out-of-distribution observation")
        critics = CriticSuite.evaluate(observation, predicted, policy.name, base_reasons, provisional_uncertainty, intent, temporal_reasons)
        score = self._utility_score(observation, predicted, intent)
        if CriticSuite.rejected(critics) and not force_safe:
            score = max(0.0, score - min(65.0, 14.0 * max(1, len(critics.reasons))))
        return CandidateEvaluation(
            policy=policy.name,
            proposal_probability=max(0.0, min(1.0, proposal_probability)),
            utility_score=score,
            safe=True if force_safe else not CriticSuite.rejected(critics),
            safety_reasons=[] if force_safe else list(critics.reasons),
            predicted_kpis={key: round(value, 4) for key, value in predicted.items()},
            predicted_trajectory=[{key: round(value, 4) for key, value in row.items()} for row in twin.trajectory],
            twin_uncertainty=twin.uncertainty,
            critics=critics,
            rationale=self._candidate_rationale(observation, policy, predicted, twin.model_name),
        )

    def _utility_score(self, observation, predicted, intent):
        sla = _SLA[str(observation.slice_type)]
        latency_target = float(sla["latency"])
        loss_target = float(sla["loss"])
        latency_score = math.exp(-max(0.0, predicted["latency_ms"] - latency_target) / max(latency_target, 1.0))
        loss_score = math.exp(-max(0.0, predicted["packet_loss_pct"] - loss_target) / max(loss_target, 0.1))
        throughput_score = min(1.0, predicted["downlink_mbps"] / max(observation.throughput_demand_mbps, 1.0))
        stability_score = math.exp(-predicted["handover_failure_pct"] / 4.0)
        energy_score = max(0.0, min(1.0, 1.15 - predicted["energy_load"]))
        weights = intent.weights or {
            "latency": float(sla["weights"][0]),
            "loss": float(sla["weights"][1]),
            "throughput": float(sla["weights"][2]),
            "stability": float(sla["weights"][3]),
            "energy": float(sla["weights"][4]),
        }
        total_weight = max(sum(weights.values()), 1e-9)
        raw = (
            weights.get("latency", 0.0) * latency_score
            + weights.get("loss", 0.0) * loss_score
            + weights.get("throughput", 0.0) * throughput_score
            + weights.get("stability", 0.0) * stability_score
            + weights.get("energy", 0.0) * energy_score
        ) / total_weight
        overload_penalty = max(0.0, predicted["prb_utilization"] - 1.0) * 22.0
        return round(max(0.0, min(100.0, raw * 100.0 - overload_penalty)), 4)

    def _safety_reasons(self, observation, policy, predicted):
        reasons = []
        tx_delta = float(policy.action.get("tx_power_db_delta", 0.0))
        if not -3.0 <= tx_delta <= 3.0:
            reasons.append("transmit-power adjustment exceeds the configured ±3 dB guardrail")
        if policy.name == "energy_saver" and str(observation.slice_type) == "URLLC" and (observation.latency_ms > 8.0 or observation.packet_loss_pct > 0.1):
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
        if abs(float(policy.action.get("prb_share_delta", 0.0))) > 0.20:
            reasons.append("PRB share adjustment exceeds the bounded action envelope")
        return reasons

    @staticmethod
    def _heuristic_candidate(observation):
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
    def _confidence(selected, safe_candidates, uncertainty=0.0):
        ordered = sorted(safe_candidates, key=lambda item: item.utility_score, reverse=True)
        margin = selected.utility_score - (ordered[1].utility_score if len(ordered) > 1 else 0.0)
        confidence = 0.48 + 0.34 * selected.proposal_probability + min(0.18, max(0.0, margin / 100.0))
        confidence *= 1.0 - min(0.45, uncertainty * 0.45)
        return round(max(0.0, min(0.99, confidence)), 4)

    @staticmethod
    def _candidate_rationale(observation, policy, predicted, world_model):
        latency_delta = observation.latency_ms - predicted["latency_ms"]
        throughput_delta = predicted["downlink_mbps"] - observation.downlink_mbps
        loss_delta = observation.packet_loss_pct - predicted["packet_loss_pct"]
        return f"{policy.description} {world_model} predicts: latency {latency_delta:+.1f} ms, downlink {throughput_delta:+.1f} Mbps, packet loss {loss_delta:+.2f} percentage points."

    @staticmethod
    def _observation_summary(observation):
        return f"{observation.cell_id}/{observation.slice_type}: PRB={observation.prb_utilization:.2f}, latency={observation.latency_ms:.1f} ms, loss={observation.packet_loss_pct:.2f}%, demand={observation.throughput_demand_mbps:.1f} Mbps, SINR={observation.sinr_db:.1f} dB, source={observation.source}"

    @staticmethod
    def _critique(selected, rejected, safety_override, uncertainty_level, ood_score):
        suffix = f" Uncertainty is {uncertainty_level}; OOD score is {ood_score:.2f}."
        if safety_override:
            return f"The highest-probability proposal was rejected by independent critics; '{selected.policy}' is the best safe alternative." + suffix
        if rejected:
            return f"The critics rejected {', '.join(rejected)} and retained the highest-utility safe candidate." + suffix
        return "All evaluated candidates passed the configured critics; the highest combined utility and proposal confidence won." + suffix
