"""Adaptive, common-gated RAN candidate selection (offline research only).

Combines the strongest observed aspects of conformal support, per-slice risk
ranking, a policy-fit behavior prior, and incumbent hysteresis. No candidate
can bypass the existing SLA, uncertainty, OOD or two-role support checks.
The 'hold' output means that the current action label was selected under the
same eligibility mask; it is NOT a measured no-op or verified intervention.

This controller does not claim closed-loop safety or causal improvement.
Selection is deterministic, vectorized, and independent of test outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HybridConfig:
    name: str
    risk_weight: float = 0.0
    behavior_weight: float = 0.0
    switch_extra: float = 0.0
    hold_margin: float = 0.0
    ood_risk_weight: float = 0.0


# Small preregistered search. The exact conformal baseline is included,
# preventing an apparently forced hybrid selection when none improves.
HYBRID_PROFILES: tuple[HybridConfig, ...] = (
    HybridConfig("conformal_anchor"),
    HybridConfig("slice_risk_light", risk_weight=0.025),
    HybridConfig("slice_risk", risk_weight=0.075, ood_risk_weight=0.5),
    HybridConfig("slice_risk_strong", risk_weight=0.15, ood_risk_weight=1.0),
    HybridConfig("behavior_anchor", behavior_weight=0.01),
    HybridConfig("behavior_risk", risk_weight=0.025, behavior_weight=0.02),
    HybridConfig("incumbent_hysteresis", hold_margin=0.01),
    HybridConfig("adaptive_balanced", risk_weight=0.05, behavior_weight=0.01,
                 switch_extra=0.01, hold_margin=0.01, ood_risk_weight=0.5),
    HybridConfig("adaptive_conservative", risk_weight=0.10, behavior_weight=0.02,
                 switch_extra=0.02, hold_margin=0.02, ood_risk_weight=1.0),
)


@dataclass(frozen=True)
class HybridSelection:
    actions: np.ndarray
    abstain: np.ndarray
    eligible_counts: np.ndarray
    switched: np.ndarray
    verified_hold: np.ndarray
    eligible_incumbent: np.ndarray


def select_hybrid(
    scores: np.ndarray,
    sla: np.ndarray,
    uncertainty: np.ndarray,
    allowed: np.ndarray,
    support: np.ndarray,
    incumbents: np.ndarray,
    slices: np.ndarray,
    ood_scores: np.ndarray,
    *,
    profile: HybridConfig,
    max_sla: float,
    max_uncertainty: float,
    max_ood: float,
    behavior_prior: np.ndarray | None = None,
    base_switch_cost: float = 0.01,
) -> HybridSelection:
    """Choose a ranked eligible action; never fabricate fallback outcomes.

    The same mask as the legacy conformal selector is enforced again.
    An eligible incumbent may be retained by a switch margin; an ineligible
    incumbent never blocks another eligible action (the key coverage fix).
    """
    score = np.asarray(scores, dtype=float)
    if score.ndim != 2 or score.shape[1] < 1:
        raise ValueError("scores must be a nonempty 2-D action matrix")
    shape = score.shape
    if any(np.shape(a) != shape for a in (sla, uncertainty, allowed, support)):
        raise ValueError("action arrays must share shape")
    sla = np.asarray(sla, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    incumbent = np.asarray(incumbents, dtype=int)
    slice_names = np.asarray(slices, dtype=str)
    ood = np.asarray(ood_scores, dtype=float)
    if any(x.shape != (shape[0],) for x in (incumbent, slice_names, ood)):
        raise ValueError("incumbent, slice and OOD arrays must match row count")
    if not set(slice_names).issubset({"eMBB", "mMTC", "URLLC"}):
        raise ValueError("unknown network slice")
    params = (
        profile.risk_weight, profile.behavior_weight, profile.switch_extra,
        profile.hold_margin, profile.ood_risk_weight, base_switch_cost,
    )
    if any(not np.isfinite(x) or x < 0 for x in params):
        raise ValueError("hybrid weights and costs must be finite and nonnegative")
    if not np.isfinite(max_sla) or not np.isfinite(max_uncertainty) or not np.isfinite(max_ood):
        raise ValueError("invalid risk/OOD limits")
    if min(max_sla, max_uncertainty, max_ood) <= 0:
        raise ValueError("risk/OOD limits must be strictly positive")
    if profile.behavior_weight:
        if behavior_prior is None:
            raise ValueError("behavior prior required for this hybrid profile")
        prior = np.asarray(behavior_prior, dtype=float)
        if prior.shape != shape or not np.isfinite(prior).all() or (prior < 0).any():
            raise ValueError("behavior prior must be a finite nonnegative action matrix")
    else:
        prior = None

    valid = (
        np.asarray(allowed, dtype=bool) & np.asarray(support, dtype=bool)
        & np.isfinite(score) & np.isfinite(sla) & np.isfinite(uncertainty)
        & np.isfinite(ood[:, None]) & (ood[:, None] <= max_ood)
        & (sla <= max_sla) & (uncertainty <= max_uncertainty)
    )
    counts = valid.sum(axis=1)
    n, a = shape
    row = np.arange(n)
    col = np.clip(incumbent, 0, a - 1)
    eligible_incumbent = (incumbent >= 0) & (incumbent < a) & valid[row, col]

    adjusted = score.copy()
    if profile.risk_weight:
        # The per-slice weights are fixed before seeing final-test outcomes.
        # Stronger emphasis on URLLC and less on eMBB allows a risk/utility
        # balance without ever relaxing the hard eligibility gate.
        slice_factor = np.select(
            [slice_names == "URLLC", slice_names == "mMTC"],
            [1.5, 1.0], default=0.65,
        )
        pressure = (
            0.7 * np.clip(sla / max_sla, 0, 2) ** 2
            + 0.3 * np.clip(uncertainty / max_uncertainty, 0, 2) ** 2
        )
        ood_pressure = 1 + profile.ood_risk_weight * np.clip(ood / max_ood, 0, 1)
        adjusted -= (
            profile.risk_weight * pressure * slice_factor[:, None]
            * ood_pressure[:, None]
        )
    if profile.behavior_weight:
        # Smoothing avoids turning a weak/zero behavior probability into a
        # second hidden eligibility gate.
        adjusted += profile.behavior_weight * np.log(np.clip(prior, 0.02, 1.0))

    is_switch = np.arange(a)[None, :] != incumbent[:, None]
    if profile.switch_extra:
        adaptive_cost = (
            base_switch_cost
            + profile.switch_extra * np.clip(uncertainty / max_uncertainty, 0, 1)
        )
    else:
        adaptive_cost = base_switch_cost
    adjusted -= is_switch * adaptive_cost

    ranked = np.where(valid, adjusted, -np.inf)
    chosen = np.argmax(ranked, axis=1)
    if profile.hold_margin:
        best = ranked[row, chosen]
        incumbent_score = ranked[row, col]
        hold = (
            eligible_incumbent
            & (best <= incumbent_score + profile.hold_margin)
        )
        chosen = np.where(hold, col, chosen)
    abstain = counts == 0
    chosen = np.where(abstain, -1, chosen)
    switched = (~abstain) & (chosen != incumbent)
    verified_hold = (~abstain) & eligible_incumbent & (chosen == incumbent)
    return HybridSelection(
        chosen.astype(int), abstain, counts.astype(int), switched,
        verified_hold, eligible_incumbent,
    )
