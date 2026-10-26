"""Optional real-COMMAG conformal support re-ranking (research-only).

Designed to be called inside the existing experiment-role-swap pipeline.
Hyperparameters are selected on *seen validation* OPE estimates only. Final
test observations never calibrate models or thresholds. No action is issued
when the constraint/support intersection is empty (abstention).

Important: SLA critic values are predictions, NOT independently certified
one-sided upper bounds. This component cannot claim safe RAN execution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from agentic_ran.offline_policy import StudyConfig, _derive_targets
from agentic_ran.publication_science import _policy_precompute
from agentic_ran.publication_support import CONTEXT_COLUMNS
from agentic_ran.research_support import ConformalActionSupport, rerank


def _eligible_inputs(bundle, frame: pd.DataFrame, config: StudyConfig):
    pre = _policy_precompute(bundle, frame)
    predicted = pre["predictions"]
    denom = (
        config.sla_weight + config.energy_weight + config.stability_weight
        + config.uncertainty_weight
    )
    utility = 1.0 - (
        config.sla_weight * predicted["sla"]
        + config.energy_weight * predicted["energy"]
        + config.stability_weight * predicted["stability"]
        + config.uncertainty_weight * predicted["uncertainty"]
    ) / max(denom, 1e-9)
    utility += config.planning_weight * pre["q_norm"]
    allowed = pre["allowed"] & (
        pre["ood"][:, None] <= config.ood_threshold
    )
    return utility, predicted["sla"], predicted["uncertainty"], allowed


def _actions_and_metrics(
    support: ConformalActionSupport,
    bundle, evaluator, frame: pd.DataFrame,
    config: StudyConfig, method: str, alpha: float,
    *, sla_residual_margin: float = 0.0, batch_size: int = 16384,
):
    # Bound peak memory and kNN query sizes for full-size COMMAG splits.
    n = len(frame)
    accepted = 0
    reranked = 0
    counted = 0
    sum_utility = 0.0
    sum_sla = 0.0
    sum_energy = 0.0
    sum_eligible = 0.0
    for start in range(0, n, batch_size):
        chunk = frame.iloc[start:start + batch_size]
        scores, sla, uncertainty, allowed = _eligible_inputs(bundle, chunk, config)
        # One-sided *logged-action* validation residual margin. This is not
        # an identified counterfactual SLA upper confidence bound.
        sla = np.clip(sla + sla_residual_margin, 0, 1)
        p = support.action_support(
            chunk[list(CONTEXT_COLUMNS)].to_numpy(float),
            chunk["slice_type"].astype(str).to_numpy(),
            bundle.actions, method, alpha,
        )
        # Logged incumbent is for the switching penalty only, NOT a bypass.
        from agentic_ran.offline_policy import _current_actions
        incumbents = _current_actions(chunk)
        action_index = {a: i for i, a in enumerate(bundle.actions)}
        incumbent_idx = [action_index.get(str(a), -1) for a in incumbents]
        decisions = rerank(
            scores, sla, uncertainty, allowed, p, incumbent_idx,
            method=method, alpha=alpha,
            max_sla=config.sla_threshold,
            max_uncertainty=config.uncertainty_threshold,
            switch_cost=0.01,
        )
        ok = np.flatnonzero(~decisions.abstain)
        counted += len(chunk)
        accepted += len(ok)
        reranked += int(decisions.reranked.sum())
        sum_eligible += float(decisions.eligible_counts.sum())
        if not len(ok):
            continue
        # OPE predictions never influence the selection decision.
        ope = evaluator.predict_all(chunk)
        ix = decisions.actions[ok]
        risk = ope["sla"][ok, ix]
        energy = ope["energy"][ok, ix]
        stability = ope["stability"][ok, ix]
        weight = config.sla_weight + config.energy_weight + config.stability_weight
        value = 1.0 - (
            config.sla_weight * risk
            + config.energy_weight * energy
            + config.stability_weight * stability
        ) / max(weight, 1e-9)
        sum_utility += float(np.sum(value))
        sum_sla += float(np.sum(risk > config.sla_threshold))
        sum_energy += float(np.sum(energy))
    return {
        "rows": counted,
        "accepted": accepted,
        "coverage": accepted / max(counted, 1),
        "abstention_rate": 1 - accepted / max(counted, 1),
        "rerank_rate": reranked / max(counted, 1),
        "mean_eligible_actions": sum_eligible / max(counted, 1),
        "conditional_dm_utility": sum_utility / accepted if accepted else None,
        "conditional_dm_sla_violation": sum_sla / accepted if accepted else None,
        "conditional_dm_energy_proxy": sum_energy / accepted if accepted else None,
        # An explicitly chosen opportunity cost makes unconditional results
        # comparable at differing abstention rates; this is NOT observed RAN utility.
        "assumed_abstention_utility": -0.12,
        "unconditional_decision_score": (
            sum_utility - 0.12 * (counted - accepted)
        ) / max(counted, 1),
        "evidence": "independent_direct_method_estimate_not_observed",
    }


def run_optional_conformal_fold(
    bundle, evaluator, policy_fit: pd.DataFrame,
    ope_fit: pd.DataFrame, validation: pd.DataFrame,
    seen: pd.DataFrame, unseen: pd.DataFrame,
    config: StudyConfig, *, seed: int, fold: str,
):
    roles = {
        role: (
            frame[list(CONTEXT_COLUMNS)].to_numpy(float),
            frame["slice_type"].astype(str).to_numpy(),
            frame["action"].astype(str).to_numpy(),
        )
        for role, frame in (("policy", policy_fit), ("evaluator", ope_fit))
    }
    support = ConformalActionSupport(
        seed=seed, max_reference=800,
    ).fit(roles)
    # Separate episodes for critic residual calibration and selection.
    episode_values = validation["episode_id"].astype(str).unique()
    if len(episode_values) < 2:
        raise ValueError("need >=2 validation episodes for disjoint calibration/selection")
    rng = np.random.default_rng(seed + 177)
    shuffled = rng.permutation(episode_values)
    pivot = max(1, len(shuffled) // 2)
    residual_frame = validation[
        validation["episode_id"].astype(str).isin(shuffled[:pivot])
    ]
    selection_frame = validation[
        validation["episode_id"].astype(str).isin(shuffled[pivot:])
    ]
    critic = _policy_precompute(bundle, residual_frame)["predictions"]["sla"]
    targets = _derive_targets(residual_frame, bundle.energy_scale, bundle.config)
    action_index = {a: i for i, a in enumerate(bundle.actions)}
    logged = targets["action"].astype(str).map(action_index).fillna(-1).to_numpy(int)
    observed = targets["_sla_cost"].to_numpy(float)
    available = logged >= 0
    residuals = observed[available] - critic[
        np.flatnonzero(available), logged[available]
    ]
    residuals = residuals[np.isfinite(residuals)]
    if not len(residuals):
        # Nothing can be certified even under observational calibration.
        margin = float("inf")
    else:
        quantile = min(1.0, np.ceil((len(residuals) + 1) * 0.95) / len(residuals))
        margin = max(0.0, float(np.quantile(residuals, quantile, method="higher")))
    grid = []
    for method in ("knn", "gaussian", "gmm"):
        for alpha in (0.01, 0.05, 0.10, 0.20):
            metrics = _actions_and_metrics(
                support, bundle, evaluator, selection_frame, config, method, alpha,
                sla_residual_margin=margin,
            )
            grid.append({
                "fold": fold, "split": "validation",
                "method": method, "alpha": alpha, **metrics,
            })
    winner = max(
        grid, key=lambda item: (
            item["unconditional_decision_score"]
            - 0.01 * item["abstention_rate"]
        ),
    )
    results = []
    for name, frame in (("seen", seen), ("unseen", unseen)):
        metrics = _actions_and_metrics(
            support, bundle, evaluator, frame, config,
            winner["method"], winner["alpha"],
            sla_residual_margin=margin,
        )
        results.append({
            "fold": fold, "split": name,
            "method": winner["method"], "alpha": winner["alpha"],
            **metrics,
        })
    return {
        "selection": {
            "fold": fold, "method": winner["method"], "alpha": winner["alpha"],
            "selection_on": "seen_validation_only_disjoint_episodes",
            "calibration_episodes": len(shuffled[:pivot]),
            "selection_episodes": len(shuffled[pivot:]),
            "logged_action_sla_residual_margin": margin,
            "logged_residual_count": len(residuals),
            "support_fit_on": "training_experiment_roles_only",
        },
        "validation_grid": grid,
        "final_tests": results,
    }
