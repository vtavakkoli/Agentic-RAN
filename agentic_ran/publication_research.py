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

from agentic_ran.offline_policy import StudyConfig, _current_actions, _derive_targets
from agentic_ran.publication_science import _choose_from_precomputed, _policy_precompute
from agentic_ran.selective_evidence import cluster_accounting, combine_chunks, accounting_summary
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
    return utility, predicted["sla"], predicted["uncertainty"], allowed, pre


def _actions_and_metrics(
    support: ConformalActionSupport,
    bundle, evaluator, frame: pd.DataFrame,
    config: StudyConfig, method: str, alpha: float,
    *, sla_residual_margin: float = 0.0, batch_size: int = 16384,
    collect_clusters: bool = False, switch_margin: float = 0.0,
):
    """Paired evaluation with three explicitly distinct abstention estimands.

    All policy candidates use policy-side critics; independent evaluator
    predictions enter only *after* selection. If no candidate is safe and
    supported, actuation is withheld. Holding the current action is an
    *external counterfactual evaluation scenario*, never silent execution.
    """
    n = len(frame)
    accepted = reranked = counted = 0
    sum_utility = sum_sla = sum_energy = sum_eligible = 0.0
    sum_hold_score = 0.0
    hold_available = hold_abstentions = 0
    evidence_rows = []
    breakdown = []
    for start in range(0, n, batch_size):
        chunk = frame.iloc[start:start + batch_size]
        scores, sla, uncertainty, allowed, pre = _eligible_inputs(bundle, chunk, config)
        sla = np.clip(sla + sla_residual_margin, 0, 1)
        p = support.action_support(
            chunk[list(CONTEXT_COLUMNS)].to_numpy(float),
            chunk["slice_type"].astype(str).to_numpy(),
            bundle.actions, method, alpha,
        )
        incumbents = _current_actions(chunk)
        action_index = {a: i for i, a in enumerate(bundle.actions)}
        incumbent_idx = np.array(
            [action_index.get(str(a), -1) for a in incumbents], dtype=int,
        )
        decisions = rerank(
            scores, sla, uncertainty, allowed, p, incumbent_idx,
            method=method, alpha=alpha, max_sla=config.sla_threshold,
            max_uncertainty=config.uncertainty_threshold,
            switch_cost=0.01,
        )
        keep = ~decisions.abstain
        ok = np.flatnonzero(keep)
        counted += len(chunk)
        accepted += len(ok)
        reranked += int(decisions.reranked.sum())
        sum_eligible += float(decisions.eligible_counts.sum())
        ope = evaluator.predict_all(chunk)
        weight = config.sla_weight + config.energy_weight + config.stability_weight
        dm = 1.0 - (
            config.sla_weight * ope["sla"]
            + config.energy_weight * ope["energy"]
            + config.stability_weight * ope["stability"]
        ) / max(weight, 1e-9)
        mapping = {action: i for i, action in enumerate(evaluator.actions)}
        action_names = np.asarray(bundle.actions, dtype=object)
        selected_dm = np.full(len(chunk), np.nan)
        if len(ok):
            chosen = action_names[decisions.actions[ok]]
            chosen_idx = np.array([mapping[str(action)] for action in chosen], dtype=int)
            selected_dm[ok] = dm[ok, chosen_idx]
            sum_utility += float(selected_dm[ok].sum())
            sum_sla += float(np.sum(ope["sla"][ok, chosen_idx] > config.sla_threshold))
            sum_energy += float(np.sum(ope["energy"][ok, chosen_idx]))
        local = np.flatnonzero(incumbent_idx >= 0)
        incumbent_dm = np.full(len(chunk), np.nan)
        if len(local):
            eval_idx = np.array([mapping.get(str(incumbents[i]), -1) for i in local])
            valid = eval_idx >= 0
            incumbent_dm[local[valid]] = dm[local[valid], eval_idx[valid]]
        incumbent_ok = (
            np.isfinite(incumbent_dm)
            & evaluator.supported(chunk, incumbents)
            & pre["allowed"][
                np.arange(len(chunk)), np.maximum(incumbent_idx, 0)
            ]
        )
        hold = ~keep
        supported_hold = hold & incumbent_ok
        hold_available += int(supported_hold.sum())
        hold_abstentions += int(hold.sum())
        # For unresolved hold decisions, retain the explicit penalty rather
        # than pretending the fallback was validated or has zero cost.
        sum_hold_score += float(np.nansum(selected_dm[ok])) + float(
            np.sum(incumbent_dm[supported_hold])
        ) - 0.12 * int((hold & ~incumbent_ok).sum())
        if collect_clusters:
            reference_decisions = _choose_from_precomputed(
                bundle, pre, config, switch_margin,
            )
            references = reference_decisions["selected_action"].astype(str).to_numpy()
            reference_idx = np.array([mapping[str(a)] for a in references], dtype=int)
            reference_dm = dm[np.arange(len(chunk)), reference_idx]
            observed = _derive_targets(
                chunk, evaluator.energy_scale, config,
            )
            logged_is_incumbent = (
                observed["action"].astype(str).to_numpy()
                == incumbents.astype(str)
            )
            evidence_rows.extend(cluster_accounting(
                chunk, keep, selected_dm, reference_dm,
                incumbent_dm, incumbent_ok, logged_is_incumbent,
                observed["_utility_reward"].to_numpy(float),
                abstention_penalty=-0.12,
            ))
            if {"scenario", "slice_type"} <= set(chunk.columns):
                for (scenario, sl), pos in chunk.groupby(
                    ["scenario", "slice_type"], sort=True,
                ).indices.items():
                    pos = np.asarray(pos, dtype=int)
                    breakdown.append({
                        "scenario": str(scenario), "slice_type": str(sl),
                        "rows": len(pos), "accepted": int(keep[pos].sum()),
                    })
    metrics = {
        "rows": counted, "accepted": accepted,
        "coverage": accepted / max(counted, 1),
        "abstention_rate": 1 - accepted / max(counted, 1),
        "rerank_rate": reranked / max(counted, 1),
        "mean_eligible_actions": sum_eligible / max(counted, 1),
        "conditional_dm_utility": sum_utility / accepted if accepted else None,
        "conditional_dm_sla_violation": sum_sla / accepted if accepted else None,
        "conditional_dm_energy_proxy": sum_energy / accepted if accepted else None,
        "assumed_abstention_utility": -0.12,
        "unconditional_decision_score": (
            sum_utility - 0.12 * (counted - accepted)
        ) / max(counted, 1),
        "hold_incumbent_dm_sensitivity_score": sum_hold_score / max(counted, 1),
        "hold_incumbent_support_on_abstentions": hold_available / max(hold_abstentions, 1),
        "evidence": "independent_direct_method_and_observational_proxy_not_causal",
    }
    if collect_clusters:
        cluster_table = combine_chunks(evidence_rows)
        metrics["cluster_rows"] = cluster_table.to_dict(orient="records")
        metrics["paired_accounting"] = accounting_summary(cluster_table)
        metrics["slice_scenario"] = (
            pd.DataFrame(breakdown)
            .groupby(["scenario", "slice_type"], as_index=False)[["rows", "accepted"]]
            .sum().to_dict(orient="records") if breakdown else []
        )
    return metrics

def run_optional_conformal_fold(
    bundle, evaluator, policy_fit: pd.DataFrame,
    ope_fit: pd.DataFrame, validation: pd.DataFrame,
    seen: pd.DataFrame, unseen: pd.DataFrame,
    config: StudyConfig, *, seed: int, fold: str,
    switch_margin: float = 0.0,
):
    roles = {
        role: (
            frame[list(CONTEXT_COLUMNS)].to_numpy(float),
            frame["slice_type"].astype(str).to_numpy(),
            frame["action"].astype(str).to_numpy(),
            frame["episode_id"].astype(str).to_numpy(),
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
            item["hold_incumbent_dm_sensitivity_score"]
            - 0.01 * item["abstention_rate"]
        ),
    )
    results = []
    clusters = []
    slices = []
    for name, frame in (("seen", seen), ("unseen", unseen)):
        metrics = _actions_and_metrics(
            support, bundle, evaluator, frame, config,
            winner["method"], winner["alpha"],
            sla_residual_margin=margin, collect_clusters=True,
            switch_margin=switch_margin,
        )
        clusters.extend([
            {"fold": fold, "split": name, **row}
            for row in metrics.pop("cluster_rows")
        ])
        slices.extend([
            {"fold": fold, "split": name, **row}
            for row in metrics.pop("slice_scenario")
        ])
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
            "support_calibration_protocol": support.calibration_protocol,
            "support_calibration_units": {
                "|".join(key): int(count)
                for key, count in support.calibration_units.items()
            },
            "score_selected_on": "validation_hold_incumbent_dm_with_unresolved_penalty",
        },
        "validation_grid": grid,
        "final_tests": results,
        "cluster_evidence": clusters,
        "slice_scenario": slices,
    }
