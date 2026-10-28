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
from agentic_ran.research_calibration import calibrate_episode_sla
from agentic_ran.publication_science import _choose_from_precomputed, _policy_precompute
from agentic_ran.selective_evidence import cluster_accounting, combine_chunks, accounting_summary
from agentic_ran.publication_support import CONTEXT_COLUMNS
from agentic_ran.research_support import ConformalActionSupport, rerank
from agentic_ran.research_behavior import (
    MODES, BehaviorPrior, select_behavior, stable_uniforms,
)


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
    behavior_mode: str | None = None, behavior_model: BehaviorPrior | None = None,
    temperature: float = 0.06, behavior_weight: float = 0.5,
    behavior_floor: float = 0.02, hold_margin: float = 0.02,
    random_seed: int = 2026,
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
    gate_totals = {
        "slice_and_ood_rows": 0,
        "sla_and_uncertainty_rows": 0,
        "conditional_support_rows": 0,
        "joint_eligible_rows": 0,
    }
    evidence_rows = []
    breakdown = []
    total_entropy = 0.0
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
        # Report *where* rejection occurs. Overlap and predicted risk are
        # distinct gates: a zero-coverage policy must never pass unnoticed.
        finite = np.isfinite(scores) & np.isfinite(sla) & np.isfinite(uncertainty)
        risk_ok = (
            finite & (sla <= config.sla_threshold)
            & (uncertainty <= config.uncertainty_threshold)
        )
        eligible = allowed & risk_ok & p
        gate_totals["slice_and_ood_rows"] += int(np.any(allowed, axis=1).sum())
        gate_totals["sla_and_uncertainty_rows"] += int(np.any(allowed & risk_ok, axis=1).sum())
        gate_totals["conditional_support_rows"] += int(np.any(allowed & p, axis=1).sum())
        gate_totals["joint_eligible_rows"] += int(np.any(eligible, axis=1).sum())
        if behavior_mode is None:
            decisions = rerank(
                scores, sla, uncertainty, allowed, p, incumbent_idx,
                method=method, alpha=alpha, max_sla=config.sla_threshold,
                max_uncertainty=config.uncertainty_threshold,
                switch_cost=0.01,
            )
            changed = decisions.reranked
        else:
            if behavior_model is None:
                raise ValueError("behavior model required for experimental controller")
            decisions = select_behavior(
                scores, sla, uncertainty, allowed, p,
                behavior_model.probabilities(chunk), incumbent_idx,
                mode=behavior_mode, temperature=temperature,
                behavior_weight=behavior_weight, min_behavior_prob=behavior_floor,
                max_sla=config.sla_threshold,
                max_uncertainty=config.uncertainty_threshold,
                switch_cost=0.01, hold_margin=hold_margin,
                uniforms=(
                    stable_uniforms(chunk, seed=random_seed)
                    if behavior_mode == "boltzmann" else None
                ),
            )
            original = np.argmax(np.where(allowed, scores, -np.inf), axis=1)
            changed = (~decisions.abstain) & (decisions.actions != original)
            total_entropy += float(np.sum(decisions.probability_entropy))
        keep = ~decisions.abstain
        ok = np.flatnonzero(keep)
        counted += len(chunk)
        accepted += len(ok)
        reranked += int(changed.sum())
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
        incumbent_positions = np.maximum(incumbent_idx, 0)
        row_positions = np.arange(len(chunk))
        # A prospective hold action is evaluated only if it passes the SAME
        # policy-side SLA/uncertainty/OOD and two-role support mask as an
        # issued proposal. These are model-based gates, not real safety proof.
        incumbent_ok = (
            np.isfinite(incumbent_dm)
            & evaluator.supported(chunk, incumbents)
            & allowed[row_positions, incumbent_positions]
            & p[row_positions, incumbent_positions]
            & (sla[row_positions, incumbent_positions] <= config.sla_threshold)
            & (
                uncertainty[row_positions, incumbent_positions]
                <= config.uncertainty_threshold
            )
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
        "policy_variant": behavior_mode or "conformal_rerank",
        "mean_action_entropy": total_entropy / max(counted, 1),
        "gate_slice_ood_coverage": gate_totals["slice_and_ood_rows"] / max(counted, 1),
        "gate_predicted_risk_coverage": gate_totals["sla_and_uncertainty_rows"] / max(counted, 1),
        "gate_support_coverage": gate_totals["conditional_support_rows"] / max(counted, 1),
        "gate_joint_coverage": gate_totals["joint_eligible_rows"] / max(counted, 1),
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
    behavior_variants_enabled: bool = False,
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
    episode_for_residual = targets.loc[
        available, "episode_id"
    ].astype(str).to_numpy()
    finite_residual = np.isfinite(residuals)
    residuals = residuals[finite_residual]
    episode_for_residual = episode_for_residual[finite_residual]
    # This is a logged-action episode-MEAN bias diagnostic. The previous
    # pointwise episode maximum exceeded the SLA cutoff in both COMMAG folds
    # and rendered every candidate impossible. The worst-step diagnostic
    # remains reported, never substituted for the mean-risk estimand.
    risk_calibration = calibrate_episode_sla(
        residuals, episode_for_residual, confidence=0.95, min_episodes=19,
    )
    margin = risk_calibration.mean_loss_bias_margin
    blocked_residuals = np.unique(episode_for_residual)
    grid = []
    for method in ("knn", "gaussian", "gmm"):
        # Action-conditional episode counts are often 6-16 on COMMAG.
        # At alpha=.01 every group fails resolution by construction.
        # The method still fails closed where n_cal is too small for alpha.
        for alpha in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35):
            metrics = _actions_and_metrics(
                support, bundle, evaluator, selection_frame, config, method, alpha,
                sla_residual_margin=margin,
            )
            grid.append({
                "fold": fold, "split": "validation",
                "method": method, "alpha": alpha, **metrics,
            })
    # Never pronounce an all-abstain validation grid as a successful tune.
    min_validation_coverage = 0.05
    feasible_grid = [
        item for item in grid
        if item["coverage"] >= min_validation_coverage
        and np.isfinite(item["hold_incumbent_dm_sensitivity_score"])
    ]
    validation_feasible = bool(feasible_grid)
    winner = max(
        feasible_grid if feasible_grid else grid,
        key=lambda item: (
            item["hold_incumbent_dm_sensitivity_score"]
            - 0.01 * item["abstention_rate"],
            item["gate_joint_coverage"],
        ),
    )
    results = []
    clusters = []
    slices = []
    behavior_validation = []
    behavior_tests = []
    behavior_cluster_rows = []
    behavior_slice_rows = []
    behavior_selections = []
    behavior_model = None
    if behavior_variants_enabled:
        behavior_model = BehaviorPrior.fit(
            policy_fit, bundle.actions, seed=seed + 9001,
        )
        variants = {
            "boltzmann": [
                {"temperature": t} for t in (0.025, 0.06, 0.12)
            ],
            "behavior_clone": [{}],
            "behavior_kl": [
                {"temperature": t, "behavior_weight": w}
                for t in (0.035, 0.09) for w in (0.25, 0.75)
            ],
            "behavior_tree": [
                {"behavior_floor": floor, "hold_margin": margin}
                for floor in (0.01, 0.05) for margin in (0.01, 0.04)
            ],
        }
        for mode, grid_options in variants.items():
            candidates = []
            for hyper in grid_options:
                metrics = _actions_and_metrics(
                    support, bundle, evaluator, selection_frame, config,
                    winner["method"], winner["alpha"],
                    sla_residual_margin=margin,
                    behavior_mode=mode, behavior_model=behavior_model,
                    random_seed=seed + 9002, **hyper,
                )
                item = {
                    "fold": fold, "split": "validation",
                    "mode": mode, "support_method": winner["method"],
                    "support_alpha": winner["alpha"], **hyper, **metrics,
                }
                candidates.append(item)
                behavior_validation.append(item)
            feasible = [item for item in candidates if item["coverage"] >= 0.05]
            pool = feasible or candidates
            best = max(
                pool, key=lambda item: (
                    item["hold_incumbent_dm_sensitivity_score"]
                    - 0.01 * item["abstention_rate"],
                    item["coverage"],
                ),
            )
            chosen_hyper = {name: best[name] for name in (
                "temperature", "behavior_weight", "behavior_floor", "hold_margin"
            ) if name in best}
            behavior_selections.append({
                "fold": fold, "mode": mode,
                "validated": bool(feasible), "validation_coverage": best["coverage"],
                "support_method": winner["method"],
                "support_alpha": winner["alpha"],
                "training_prior_rows": behavior_model.policy_rows,
                "hyperparameters": chosen_hyper,
                "selection_scope": "seen_validation_only; no final-test tuning",
            })
            for split, final_frame in (("seen", seen), ("unseen", unseen)):
                metrics = _actions_and_metrics(
                    support, bundle, evaluator, final_frame, config,
                    winner["method"], winner["alpha"],
                    sla_residual_margin=margin, collect_clusters=True,
                    switch_margin=switch_margin,
                    behavior_mode=mode, behavior_model=behavior_model,
                    random_seed=seed + 9002, **chosen_hyper,
                )
                behavior_cluster_rows.extend([
                    {"fold": fold, "split": split, "mode": mode, **row}
                    for row in metrics.pop("cluster_rows")
                ])
                behavior_slice_rows.extend([
                    {"fold": fold, "split": split, "mode": mode, **row}
                    for row in metrics.pop("slice_scenario")
                ])
                behavior_tests.append({
                    "fold": fold, "split": split, "mode": mode,
                    "support_method": winner["method"],
                    "support_alpha": winner["alpha"],
                    **chosen_hyper, **metrics,
                })
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
            "residual_calibration_episode_count": len(blocked_residuals),
            "residual_calibration_protocol": "episode_mean_quantile_95pct",
            "residual_worst_step_diagnostic_margin": (
                risk_calibration.worst_step_residual_margin
            ),
            "sla_calibration": risk_calibration.as_dict(),
            "validation_feasible": validation_feasible,
            "minimum_validation_coverage": min_validation_coverage,
            "feasibility_warning": (
                None if validation_feasible
                else "NO_FEASIBLE_VALIDATION_CANDIDATE; method is diagnostic only"
            ),
            "support_fit_on": "training_experiment_roles_only",
            "support_calibration_protocol": support.calibration_protocol,
            "support_calibration_units": {
                "|".join(key): int(count)
                for key, count in support.calibration_units.items()
            },
            "score_selected_on": (
                "validation_feasible_hold_incumbent_dm_with_unresolved_penalty"
            ),
        },
        "validation_grid": grid,
        "final_tests": results,
        "cluster_evidence": clusters,
        "slice_scenario": slices,
        "behavior_variants": {
            "enabled": behavior_variants_enabled,
            "validation_grid": behavior_validation,
            "final_tests": behavior_tests,
            "cluster_evidence": behavior_cluster_rows,
            "slice_scenario": behavior_slice_rows,
            "selections": behavior_selections,
        },
    }
