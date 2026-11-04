"""Research-only incumbent-relative constrained policies for offline RAN.

The policy-side critic's 'uncertainty' is NOT a calibrated action-advantage
confidence interval. Thus the LCB names below denote *sensitivity-score
heuristics*, never a safe-policy-improvement theorem. No action is issued
unless the original slice/OOD, conformal, predicted SLA and uncertainty gates
are all satisfied. The independently trained evaluator is NEVER an input.

The 'jev_guided' candidate accepts only externally supplied, schema-checked
numeric hints; this module does not run or train a Jev model. No external
advice can change the eligibility gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

MODES = (
    "advantage_lcb",
    "constraint_conditioned",
    "behavior_anchored_lcb",
    "verified_hold",
    "jev_guided",
)


@dataclass(frozen=True)
class SafeImprovementSelection:
    actions: np.ndarray  # -1 is no actuation
    abstain: np.ndarray
    eligible_counts: np.ndarray
    reasons: np.ndarray
    switched: np.ndarray
    incumbent_eligible: np.ndarray


def validated_jev_hints(
    hints: Mapping[str, float], action_names: tuple[str, ...],
    *, max_abs_score: float = 0.25,
) -> np.ndarray:
    """Convert typed, allow-listed Jev decision hints into a bounded score.

    No arbitrary strings, candidate names, or unsupported actions accepted.
    Jev integration is an OFFLINE adapter; scores are a *hypothesis* until
    external model outputs, source provenance, and latency are recorded.
    """
    if not isinstance(hints, Mapping):
        raise TypeError("Jev hints must be a mapping")
    names = set(action_names)
    if set(hints) - names:
        raise ValueError("Jev advice refers to unknown RAN action")
    if not 0 <= max_abs_score <= 1:
        raise ValueError("invalid Jev hint limit")
    out = np.array([float(hints.get(name, 0.0)) for name in action_names])
    if not np.isfinite(out).all() or (np.abs(out) > max_abs_score).any():
        raise ValueError("Jev hint score must be finite and bounded")
    return out


def select_safe_improvement(
    scores: np.ndarray,
    sla: np.ndarray,
    uncertainty: np.ndarray,
    allowed: np.ndarray,
    support: np.ndarray,
    incumbents: np.ndarray,
    *,
    mode: str,
    max_sla: float,
    max_uncertainty: float,
    advantage_penalty: float = 1.0,
    switch_cost: float = 0.01,
    risk_weight: float = 0.2,
    behavior_prior: np.ndarray | None = None,
    behavior_weight: float = 0.05,
    min_behavior_prob: float = 0.005,
    advice_scores: np.ndarray | None = None,
    advice_weight: float = 0.0,
) -> SafeImprovementSelection:
    """Generate conservative proposals under the SAME external gates.

    advantage_lcb: switch only if (predicted delta) exceeds uncertainty proxy
        and switching cost relative to a *verified eligible* incumbent.
    constraint_conditioned: utility - risk-weighted predicted normalized
        SLA/uncertainty burden within the hard safety envelope.
    behavior_anchored_lcb: switch only when behavior-regularized advantage
        clears the uncertainty/switch cost (policy-role prior required).
    verified_hold: prefer the safe incumbent; switch only on larger gain.
    jev_guided: additive bounded typed advice, then same hard safety mask.
    """
    if mode not in MODES:
        raise ValueError("unknown research candidate mode")
    score = np.asarray(scores, dtype=float)
    risk = np.asarray(sla, dtype=float)
    unc = np.asarray(uncertainty, dtype=float)
    if score.ndim != 2 or score.shape[1] < 1:
        raise ValueError("scores must be a nonempty action matrix")
    shape = score.shape
    if any(np.shape(x) != shape for x in (risk, unc, allowed, support)):
        raise ValueError("policy input dimensions differ")
    incumbent = np.asarray(incumbents, dtype=int)
    if incumbent.shape != (shape[0],):
        raise ValueError("incumbent size differs")
    if any(not np.isfinite(z) or z < 0 for z in (
        advantage_penalty, switch_cost, risk_weight, behavior_weight, advice_weight,
    )):
        raise ValueError("penalty parameters must be finite and nonnegative")
    if max_sla <= 0 or max_uncertainty <= 0 or not 0 <= min_behavior_prob < 1:
        raise ValueError("invalid SLA/uncertainty or behavior floor")
    prior = np.zeros(shape, dtype=float)
    if behavior_prior is not None:
        prior = np.asarray(behavior_prior, dtype=float)
        if prior.shape != shape or not np.isfinite(prior).all() or (prior < 0).any():
            raise ValueError("invalid policy-role behavior prior")
    if mode == "behavior_anchored_lcb" and behavior_prior is None:
        raise ValueError("behavior-anchored mode requires fitted prior")
    advice = np.zeros(shape, dtype=float)
    if advice_scores is not None:
        advice = np.asarray(advice_scores, dtype=float)
        if advice.shape != shape or not np.isfinite(advice).all():
            raise ValueError("invalid structured Jev advice")
        if np.any(np.abs(advice) > 0.25 + 1e-12):
            raise ValueError("Jev hints must be bounded to [-0.25,0.25]")
    if mode == "jev_guided" and advice_scores is None:
        raise ValueError("Jev candidate requires externally recorded advice; do not fake inference")

    valid = (
        np.asarray(allowed, dtype=bool)
        & np.asarray(support, dtype=bool)
        & np.isfinite(score) & np.isfinite(risk) & np.isfinite(unc)
        & (risk <= max_sla) & (unc <= max_uncertainty)
    )
    row = np.arange(shape[0])
    col = np.maximum(incumbent, 0)
    verified = (incumbent >= 0) & (incumbent < shape[1])
    verified &= valid[row, np.minimum(col, shape[1] - 1)]
    counts = valid.sum(axis=1)
    adjusted = score.copy()
    if mode == "constraint_conditioned":
        adjusted -= risk_weight * (
            0.7 * np.clip(risk / max_sla, 0, 1) ** 2
            + 0.3 * np.clip(unc / max_uncertainty, 0, 1) ** 2
        )
    if mode == "behavior_anchored_lcb":
        # The log prior is a regularizer, not a calibrated propensity.
        valid &= prior >= min_behavior_prob
        counts = valid.sum(axis=1)
        adjusted += behavior_weight * np.log(np.maximum(prior, 1e-12))
    if mode == "jev_guided":
        adjusted += advice_weight * advice
    adjusted -= switch_cost * (
        np.arange(shape[1])[None, :] != incumbent[:, None]
    )
    ranked = np.where(valid, adjusted, -np.inf)
    candidate = np.argmax(ranked, axis=1)
    no_candidate = counts == 0
    reasons = np.full(shape[0], "selected", dtype=object)
    if mode in ("advantage_lcb", "behavior_anchored_lcb", "verified_hold"):
        # Never compare improvement with unsupported / non-feasible incumbent.
        # A separate, externally verified hold controller is required then.
        missing = ~verified
        no_candidate |= missing
        reasons[missing] = "unverified_incumbent"
        best = ranked[row, candidate]
        inc_score = adjusted[row, np.minimum(col, shape[1] - 1)]
        penalty = advantage_penalty * np.sqrt(
            np.maximum(unc[row, candidate], 0) ** 2
            + np.maximum(unc[row, np.minimum(col, shape[1] - 1)], 0) ** 2
        )
        if mode == "verified_hold":
            penalty = 1.5 * penalty
        switch = (candidate != incumbent) & (best > inc_score + penalty)
        candidate = np.where(switch, candidate, np.minimum(col, shape[1] - 1))
        reasons[(~switch) & verified] = "verified_hold"
    no_candidate |= counts == 0
    reasons[(counts == 0)] = "no_eligible_action"
    result = np.where(no_candidate, -1, candidate)
    switched = (result >= 0) & (result != incumbent)
    return SafeImprovementSelection(
        result.astype(int), no_candidate,
        counts.astype(int), reasons, switched, verified,
    )
