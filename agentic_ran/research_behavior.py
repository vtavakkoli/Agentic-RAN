"""Behavior-aware offline RAN action proposals with a shared fail-closed mask.

These are research policies, NOT safety-certified actions. In all four modes,
policy action proposals are restricted to the SAME state-conditional support,
OOD, predicted SLA and uncertainty eligibility mask. A behavior model uses
only policy-role training data and decision-time context features. OPE
predictions are never used to choose actions.

Reference concepts: maximum-entropy Boltzmann selection, behavior cloning,
KL-regularized offline policy improvement, and behavior-tree control.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from agentic_ran.publication_support import CONTEXT_COLUMNS

MODES = ("boltzmann", "behavior_clone", "behavior_kl", "behavior_tree")


def context_features(frame: pd.DataFrame) -> np.ndarray:
    """Avoid direct scheduler/PRB fields, and prohibit future outcome features."""
    numeric = frame[list(CONTEXT_COLUMNS)].to_numpy(dtype=float)
    slices = frame["slice_type"].astype(str).to_numpy()
    one_hot = np.column_stack([
        (slices == label).astype(float) for label in ("eMBB", "mMTC", "URLLC")
    ])
    if not np.isfinite(numeric).all():
        raise ValueError("behavior-model context contains nonfinite inputs")
    return np.column_stack([numeric, one_hot])


@dataclass
class BehaviorPrior:
    model: HistGradientBoostingClassifier
    actions: tuple[str, ...]
    policy_rows: int

    @classmethod
    def fit(cls, frame: pd.DataFrame, actions, *, seed=2026, per_group_limit=2500):
        """Only policy-fit rows; stratify before capped training for efficiency."""
        if per_group_limit < 30:
            raise ValueError("per-group cap too small")
        allowed = set(map(str, actions))
        fit = frame.loc[frame["action"].astype(str).isin(allowed)]
        if fit.empty or fit["action"].nunique() < 2:
            raise ValueError("need >=2 supported logged actions to fit behavior prior")
        rng = np.random.default_rng(seed)
        picks = []
        for _, group in fit.groupby(["slice_type", "action"], sort=True):
            ix = group.index.to_numpy()
            if len(ix) > per_group_limit:
                ix = rng.choice(ix, size=per_group_limit, replace=False)
            picks.append(ix)
        sample = fit.loc[np.concatenate(picks)]
        model = HistGradientBoostingClassifier(
            max_iter=70, max_leaf_nodes=15, max_depth=5,
            min_samples_leaf=20, l2_regularization=0.2,
            learning_rate=0.06, random_state=seed,
            early_stopping=len(sample) >= 1000,
        )
        model.fit(context_features(sample), sample["action"].astype(str))
        return cls(model, tuple(map(str, actions)), len(sample))

    def probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        pred = self.model.predict_proba(context_features(frame))
        output = np.zeros((len(frame), len(self.actions)), dtype=float)
        action_idx = {action: j for j, action in enumerate(self.actions)}
        for j, label in enumerate(self.model.classes_):
            ix = action_idx.get(str(label))
            if ix is not None:
                output[:, ix] = pred[:, j]
        return output  # deliberately zero for actions never observed in fit


@dataclass(frozen=True)
class BehaviorSelection:
    actions: np.ndarray
    abstain: np.ndarray
    eligible_counts: np.ndarray
    probability_entropy: np.ndarray
    reasons: np.ndarray


def stable_uniforms(frame: pd.DataFrame, *, seed: int) -> np.ndarray:
    """Stable across batches and row ordering, no held-out outcomes used."""
    columns = [c for c in ("episode_id", "timestamp_s", "base_station") if c in frame]
    if not columns:
        raise ValueError("need stable pre-decision row identity for Boltzmann sampling")
    hashed = pd.util.hash_pandas_object(
        frame[columns].astype(str), index=False,
    ).to_numpy(dtype=np.uint64).copy()
    hashed ^= np.uint64(seed)
    # SplitMix64 integer scrambling; uint64 overflow is intentional.
    # An explicit writable copy is necessary with recent pandas versions.
    with np.errstate(over="ignore"):
        hashed = hashed + np.uint64(0x9E3779B97F4A7C15)
        hashed = (hashed ^ (hashed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        hashed = (hashed ^ (hashed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        hashed ^= hashed >> np.uint64(31)
    return ((hashed >> np.uint64(11)).astype(float) + 0.5) / 2**53


def select_behavior(
    scores: np.ndarray, risk: np.ndarray, uncertainty: np.ndarray,
    allowed: np.ndarray, support: np.ndarray, prior: np.ndarray,
    incumbents: np.ndarray,
    *, mode: str, temperature: float = 0.05, behavior_weight: float = 0.5,
    min_behavior_prob: float = 0.01, max_sla: float = 0.5,
    max_uncertainty: float = 0.6, switch_cost: float = 0.01,
    hold_margin: float = 0.02, uniforms: np.ndarray | None = None,
) -> BehaviorSelection:
    """Choose by distinct policy, never bypass common eligibility gates.

    Boltzmann draws from softmax(scores/temperature) using external seeded
    uniforms; behavior_clone greedifies the logged behavior distribution;
    behavior_kl uses Gibbs MAP log(pi_b)+score/(temperature*weight); the
    behavior-tree supervisor holds a VERIFIED incumbent near the best,
    otherwise chooses a behavior-supported safe candidate, else abstains.
    """
    if mode not in MODES:
        raise ValueError("unknown behavior control mode")
    if temperature <= 0 or behavior_weight < 0 or not 0 <= min_behavior_prob < 1:
        raise ValueError("invalid temperature, behavior weight or floor")
    score = np.asarray(scores, float)
    if score.ndim != 2:
        raise ValueError("expected 2-D action scores")
    shape = score.shape
    if any(np.shape(a) != shape for a in (risk, uncertainty, allowed, support, prior)):
        raise ValueError("all action arrays must have identical shape")
    incumbent = np.asarray(incumbents, dtype=int)
    if incumbent.shape != (shape[0],):
        raise ValueError("incumbent length mismatch")
    behavior = np.asarray(prior, float)
    if not np.isfinite(behavior).all() or (behavior < 0).any():
        raise ValueError("invalid behavior probabilities")
    feasible = (
        np.asarray(allowed, bool) & np.asarray(support, bool)
        & np.isfinite(score) & np.isfinite(risk) & np.isfinite(uncertainty)
        & (np.asarray(risk) <= max_sla)
        & (np.asarray(uncertainty) <= max_uncertainty)
    )
    counts = feasible.sum(axis=1)
    adjusted = score - switch_cost * (
        np.arange(shape[1])[None, :] != incumbent[:, None]
    )
    row, col = np.arange(shape[0]), np.maximum(incumbent, 0)
    reasons = np.full(shape[0], "selected", dtype=object)
    entropy = np.zeros(shape[0], dtype=float)
    if mode == "boltzmann":
        if uniforms is None:
            raise ValueError("Boltzmann sampling needs reproducible external uniforms")
        u = np.asarray(uniforms, dtype=float)
        if u.shape != (shape[0],) or not np.isfinite(u).all() or ((u < 0) | (u >= 1)).any():
            raise ValueError("invalid random uniforms")
        logits = np.where(feasible, adjusted / temperature, -np.inf)
        max_logit = np.max(logits, axis=1, keepdims=True)
        exponents = np.exp(
            np.where(feasible, logits - np.where(np.isfinite(max_logit), max_logit, 0), -np.inf)
        )
        weights = exponents / np.maximum(exponents.sum(axis=1, keepdims=True), 1e-300)
        entropy = -(weights * np.log(np.clip(weights, 1e-300, 1))).sum(axis=1)
        chosen = np.minimum(
            (np.cumsum(weights, axis=1) < u[:, None]).sum(axis=1),
            shape[1] - 1,
        )
    elif mode == "behavior_clone":
        logits = np.where(feasible & (behavior > 0), behavior, -np.inf)
        chosen = np.argmax(logits, axis=1)
        reasons[(counts > 0) & (~np.isfinite(logits).any(axis=1))] = "no_behavior_prior"
        counts = np.isfinite(logits).sum(axis=1)
    elif mode == "behavior_kl":
        # Argmax of pi_b(a|s) exp(utility/(weight*temperature)):
        # a deterministic Gibbs MAP proposal, distinct from stochastic softmax.
        logits = np.where(
            feasible & (behavior > 0),
            adjusted / (temperature * max(behavior_weight, 1e-8))
            + np.log(np.clip(behavior, 1e-300, 1.0)), -np.inf,
        )
        chosen = np.argmax(logits, axis=1)
        counts = np.isfinite(logits).sum(axis=1)
    else:
        masked = feasible & (behavior >= min_behavior_prob)
        ranked = np.where(masked, adjusted, -np.inf)
        chosen = np.argmax(ranked, axis=1)
        best = np.max(ranked, axis=1)
        verified_incumbent = (
            (incumbent >= 0) & masked[row, col]
            & (adjusted[row, col] >= best - hold_margin)
        )
        chosen = np.where(verified_incumbent, col, chosen)
        reasons[verified_incumbent] = "verified_hold"
        counts = masked.sum(axis=1)
    abstain = counts == 0
    reasons[abstain] = "no_eligible_action"
    return BehaviorSelection(
        np.where(abstain, -1, chosen).astype(int), abstain, counts,
        entropy, reasons,
    )
