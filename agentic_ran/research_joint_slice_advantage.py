"""Joint-Slice Advantage-Gated Control (JAGC), offline COMMAG research.

The controller builds *matched* eMBB/mMTC/URLLC decision contexts at the
same base station and second, optimizes one supported scheduler/PRB candidate
per slice with a common training-derived PRB-budget PROXY, and promotes the
joint alternative only if its policy-side uncertainty-penalized advantage
exceeds a prespecified validation-chosen margin.

Independent outcome/OPE estimates are used ONLY for validation selection and
held-out reporting, NEVER to rank actions on held-out states. The input is
recorded Colosseum COMMAG telemetry; no proposed joint action was executed.
Consequently, these are offline DIRECT-METHOD estimates, NOT causal or
realized network-control gains. No action-dependent dynamics are identified.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
import re
from typing import Any

import numpy as np
import pandas as pd

from agentic_ran.offline_policy import STATE_COLUMNS
from agentic_ran.publication_science import (
    _choose_from_precomputed, _policy_precompute,
)

SLICES = ("eMBB", "mMTC", "URLLC")
GROUP_KEYS = (
    "publication_split", "scenario", "training_config", "experiment",
    "base_station", "timestamp_s",
)
PRB_PATTERN = re.compile(r":prb=(\d+)$")


@dataclass(frozen=True)
class JointProfile:
    name: str
    minimum_advantage: float
    uncertainty_penalty: float
    switching_penalty: float


PROFILES = (
    JointProfile("original_agentic_ran", 0.0, 0.0, 0.0),
    JointProfile("joint_utility", 0.0, 0.0, 0.0),
    JointProfile("joint_margin_005", 0.005, 0.0, 0.0),
    JointProfile("joint_margin_010", 0.010, 0.0, 0.0),
    JointProfile("joint_uncertainty_025", 0.005, 0.025, 0.0),
    JointProfile("joint_uncertainty_050", 0.005, 0.050, 0.0),
    JointProfile("joint_switch_010", 0.005, 0.025, 0.010),
)


def parse_prb_costs(action_names: list[str]) -> np.ndarray:
    """Reject any action without an explicit nonnegative PRB resource count."""
    costs = []
    for action in action_names:
        match = PRB_PATTERN.search(str(action))
        if match is None:
            raise ValueError(f"uninterpretable COMMAG PRB action: {action!r}")
        costs.append(int(match.group(1)))
    if not costs or min(costs) < 0:
        raise ValueError("invalid PRB action set")
    return np.asarray(costs, dtype=int)


def aligned_joint_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Average observed per-UE features into synchronized per-slice BS-seconds.

    Does not fill missing slices; downstream scores always have three rows
    per accepted joint group and use exactly the same representative states.
    """
    cols = list(GROUP_KEYS) + ["slice_type", "action", "done"]
    numeric = [*STATE_COLUMNS, *(f"next_{name}" for name in STATE_COLUMNS)]
    missing = sorted(set(cols + numeric) - set(frame.columns))
    if missing:
        raise ValueError(f"COMMAG joint-slice alignment requires columns: {missing}")
    if frame.empty:
        raise ValueError("cannot align an empty COMMAG frame")
    if not set(frame["slice_type"].astype(str)).issubset(SLICES):
        raise ValueError("unknown COMMAG slice")
    data = frame[list(dict.fromkeys(cols + numeric))].copy()
    # Recorded simultaneous UE traces may have different KPI measurements
    # but must refer to a common base station/slice/time state.
    grouping = list(GROUP_KEYS) + ["slice_type"]
    grouped = data.groupby(grouping, sort=True, observed=True, dropna=False)
    means = grouped[numeric].mean().reset_index()
    info = grouped.agg(
        action=("action", "first"),
        action_variants=("action", "nunique"),
        done=("done", "max"),
        source_ue_rows=("action", "size"),
    ).reset_index()
    rep = means.merge(
        info, on=grouping, validate="one_to_one", how="inner",
    )
    counts = rep.groupby(list(GROUP_KEYS), observed=True, dropna=False)["slice_type"].nunique()
    complete = counts[counts.eq(3)].index
    if len(complete) == 0:
        raise ValueError(
            "no matched three-slice base-station seconds; refuse to invent "
            "eMBB/mMTC/URLLC joint states"
        )
    idx = pd.MultiIndex.from_frame(rep[list(GROUP_KEYS)])
    rep = rep.loc[idx.isin(complete)].copy()
    rep["slice_rank"] = pd.Categorical(
        rep["slice_type"], categories=list(SLICES), ordered=True,
    )
    rep.sort_values([*GROUP_KEYS, "slice_rank"], kind="mergesort", inplace=True)
    rep.reset_index(drop=True, inplace=True)
    rep.drop(columns="slice_rank", inplace=True)
    if len(rep) != 3 * len(complete):
        raise AssertionError("joint-slice alignment not exactly three rows per group")
    rep["joint_group"] = np.repeat(np.arange(len(complete)), 3)
    # Group all matched BS-seconds from one experiment into an open-loop
    # replay sequence per slice. Does not claim recorded actuation changes.
    rep["episode_id"] = (
        rep["scenario"].astype(str) + "/" + rep["training_config"].astype(str)
        + "/" + rep["experiment"].astype(str) + "/"
        + rep["base_station"].astype(str) + "/" + rep["slice_type"].astype(str)
    )
    rep["split"] = np.where(
        rep["publication_split"].astype(str) == "train", "train", "test",
    )
    return rep, {
        "original_rows": int(len(frame)),
        "all_bs_seconds": int(len(counts)),
        "complete_bs_seconds": int(len(complete)),
        "missing_slice_bs_seconds": int(len(counts) - len(complete)),
        "alignment_rate": float(len(complete) / max(len(counts), 1)),
        "mixed_logged_action_slice_seconds": int(
            np.sum(rep["action_variants"].to_numpy() > 1)
        ),
        "representative_rows": int(len(rep)),
        "aggregation": "mean_real_UE_KPI_per_slice_BS_second; no missing-slice imputation",
    }


def training_prb_budgets(
    policy_fit: pd.DataFrame, *, quantile: float = .90,
) -> dict[str, Any]:
    """Derive a PRB-budget *proxy* exclusively from the policy-fit role."""
    if not 0.5 <= quantile <= 1.0:
        raise ValueError("budget quantile must be between 0.5 and 1")
    keys = [
        name for name in GROUP_KEYS
        if name in policy_fit.columns and name != "publication_split"
    ]
    needed = [*keys, "slice_type", "slice_prb"]
    missing = sorted(set(needed) - set(policy_fit))
    if missing:
        raise ValueError(f"missing training budget inputs: {missing}")
    per_slice = policy_fit.groupby(
        [*keys, "slice_type"], observed=True, dropna=False,
    )["slice_prb"].mean().unstack("slice_type")
    complete = per_slice.reindex(columns=list(SLICES)).dropna()
    if complete.empty:
        raise ValueError("no complete training-only budget observations")
    totals = complete.sum(axis=1)
    if not np.isfinite(totals).all() or (totals <= 0).any():
        raise ValueError("nonpositive or nonfinite training allocation budget")
    base_stations = complete.index.get_level_values("base_station").astype(str)
    budget_by_bs = {}
    for bs in sorted(set(base_stations)):
        local = totals[base_stations == bs]
        budget_by_bs[bs] = int(max(1, math.ceil(float(local.quantile(quantile)))))
    return {
        "budget_by_base_station": budget_by_bs,
        "global_budget": int(max(1, math.ceil(float(totals.quantile(quantile))))),
        "quantile": float(quantile),
        "training_groups": int(len(complete)),
        "budget_units": "recorded_slice_prb_sum_proxy_not_verified_physical_gNB_capacity",
    }


def exact_multiple_choice_knapsack(
    scores: np.ndarray,
    eligible: np.ndarray,
    prb_cost: np.ndarray,
    budget: int,
    *, scheduler_names: tuple[str, ...] | None = None,
) -> tuple[np.ndarray | None, float]:
    """Select exactly one eligible action per slice with an integer PRB budget.

    Dominated options of the same action cost are compressed first. The
    remaining 3-slice dynamic programming is exact; no arbitrary top-K
    shortlist can remove the only low-cost feasible solution.
    """
    score = np.asarray(scores, dtype=float)
    allowed = np.asarray(eligible, dtype=bool)
    costs = np.asarray(prb_cost, dtype=int)
    if score.ndim != 2 or score.shape[0] != 3 or score.shape != allowed.shape:
        raise ValueError("expected three slice rows with matched action matrices")
    if costs.shape != (score.shape[1],) or np.any(costs < 0) or budget < 0:
        raise ValueError("invalid PRB costs or budget")
    if scheduler_names is not None:
        if len(scheduler_names) != score.shape[1]:
            raise ValueError("scheduler identities must match the action set")
        best_choice: np.ndarray | None = None
        best_score = float("-inf")
        for scheduler in sorted(set(scheduler_names)):
            group_mask = np.asarray([name == scheduler for name in scheduler_names], dtype=bool)
            choice, value = exact_multiple_choice_knapsack(
                score, allowed & group_mask[None, :], costs, budget,
            )
            if choice is not None and value > best_score:
                best_choice, best_score = choice, value
        return best_choice, best_score
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for slice_index in range(3):
        best_per_cost: dict[int, tuple[float, int]] = {}
        for action in np.flatnonzero(allowed[slice_index]):
            cost = int(costs[action])
            value = float(score[slice_index, action])
            if cost > budget or not np.isfinite(value):
                continue
            old = best_per_cost.get(cost)
            if old is None or value > old[0] + 1e-12:
                best_per_cost[cost] = (value, int(action))
        if not best_per_cost:
            return None, float("-inf")
        next_states: dict[int, tuple[float, tuple[int, ...]]] = {}
        for used, (utility, selected) in states.items():
            for cost, (value, action) in best_per_cost.items():
                total = used + cost
                if total > budget:
                    continue
                candidate = (utility + value, selected + (action,))
                old = next_states.get(total)
                if old is None or candidate[0] > old[0] + 1e-12:
                    next_states[total] = candidate
        states = next_states
        if not states:
            return None, float("-inf")
    _, best = max(
        states.items(), key=lambda item: (
            item[1][0], -item[0],
        ),
    )
    return np.asarray(best[1], dtype=int), float(best[0])


@dataclass
class JointInputs:
    representative: pd.DataFrame
    alignment: dict[str, Any]
    policy_score: np.ndarray
    policy_uncertainty: np.ndarray
    eligible: np.ndarray
    reference_idx: np.ndarray
    incumbent_idx: np.ndarray
    evaluator_utility: np.ndarray
    evaluator_sla: np.ndarray
    evaluator_supported: np.ndarray
    actions: list[str]
    prb_costs: np.ndarray
    budgets: np.ndarray
    shared_scheduler_required: bool = False


def prepare_joint_inputs(
    bundle: Any,
    evaluator: Any,
    frame: pd.DataFrame,
    calibrated: Any,
    switch_margin: float,
    budget_model: dict[str, Any],
    *,
    batch_groups: int = 2048,
) -> JointInputs:
    """Compute policy-side and independent-evaluator predictions separately."""
    if batch_groups < 1:
        raise ValueError("batch_groups must be positive")
    rep, alignment = aligned_joint_frame(frame)
    n = len(rep)
    na = len(bundle.actions)
    score_parts: list[np.ndarray] = []
    uncertainty_parts: list[np.ndarray] = []
    eligible_parts: list[np.ndarray] = []
    ref_parts: list[np.ndarray] = []
    incumbent_parts: list[np.ndarray] = []
    dm_parts: list[np.ndarray] = []
    dm_sla_parts: list[np.ndarray] = []
    eval_support_parts: list[np.ndarray] = []
    action_index = {str(v): i for i, v in enumerate(bundle.actions)}
    costs = parse_prb_costs(list(bundle.actions))
    for start in range(0, n, batch_groups * 3):
        chunk = rep.iloc[start:start + batch_groups * 3].copy()
        pre = _policy_precompute(bundle, chunk)
        predicted = pre["predictions"]
        denominator = (
            calibrated.sla_weight + calibrated.energy_weight
            + calibrated.stability_weight + calibrated.uncertainty_weight
        )
        score = 1.0 - (
            calibrated.sla_weight * predicted["sla"]
            + calibrated.energy_weight * predicted["energy"]
            + calibrated.stability_weight * predicted["stability"]
            + calibrated.uncertainty_weight * predicted["uncertainty"]
        ) / max(denominator, 1e-9)
        score += calibrated.planning_weight * pre["q_norm"]
        allowed = (
            pre["allowed"]
            & np.isfinite(score)
            & np.isfinite(predicted["sla"])
            & np.isfinite(predicted["uncertainty"])
            & (predicted["sla"] <= calibrated.sla_threshold)
            & (predicted["uncertainty"] <= calibrated.uncertainty_threshold)
            & (pre["ood"][:, None] <= calibrated.ood_threshold)
        )
        original = _choose_from_precomputed(
            bundle, pre, calibrated, switch_margin,
        )["selected_action"].astype(str).to_numpy()
        reference_idx = np.asarray(
            [action_index.get(x, -1) for x in original], dtype=int,
        )
        if np.any(reference_idx < 0):
            raise ValueError("original Agentic-RAN action is outside the action set")
        current = np.asarray(
            [action_index.get(str(x), -1) for x in pre["current"]], dtype=int,
        )
        ope = evaluator.predict_all(chunk)
        dm_denominator = (
            calibrated.sla_weight + calibrated.energy_weight
            + calibrated.stability_weight
        )
        dm = 1.0 - (
            calibrated.sla_weight * ope["sla"]
            + calibrated.energy_weight * ope["energy"]
            + calibrated.stability_weight * ope["stability"]
        ) / max(dm_denominator, 1e-9)
        # Safety-critical: if action order differs, remap columns BEFORE
        # evaluating or comparing any predicted utility/safety statistic.
        eval_lookup = {str(a): i for i, a in enumerate(evaluator.actions)}
        eval_indices = np.asarray(
            [eval_lookup.get(str(a), -1) for a in bundle.actions], dtype=int,
        )
        if np.any(eval_indices < 0):
            raise ValueError("independent evaluator lacks a policy action")
        dm_parts.append(dm[:, eval_indices])
        dm_sla_parts.append(ope["sla"][:, eval_indices])
        eval_support_parts.append(np.asarray([
            np.asarray([
                str(bundle.actions[a]) in evaluator.support_by_slice.get(str(sl), set())
                for a in range(na)
            ], dtype=bool)
            for sl in chunk["slice_type"].astype(str)
        ], dtype=bool))
        score_parts.append(score)
        uncertainty_parts.append(predicted["uncertainty"])
        eligible_parts.append(allowed)
        ref_parts.append(reference_idx)
        incumbent_parts.append(current)
    mapped = rep.iloc[::3]["base_station"].astype(str).map(
        budget_model["budget_by_base_station"]
    )
    if budget_model.get("physical_capacity_confirmed"):
        if mapped.isna().any():
            raise ValueError(
                "unmapped base station: cannot infer a physical NRB capacity"
            )
        budgets = mapped.to_numpy(dtype=int)
    else:
        budgets = mapped.fillna(budget_model["global_budget"]).to_numpy(dtype=int)
    return JointInputs(
        rep, alignment,
        np.concatenate(score_parts, axis=0).reshape(-1, 3, na),
        np.concatenate(uncertainty_parts, axis=0).reshape(-1, 3, na),
        np.concatenate(eligible_parts, axis=0).reshape(-1, 3, na),
        np.concatenate(ref_parts).reshape(-1, 3),
        np.concatenate(incumbent_parts).reshape(-1, 3),
        np.concatenate(dm_parts, axis=0).reshape(-1, 3, na),
        np.concatenate(dm_sla_parts, axis=0).reshape(-1, 3, na),
        np.concatenate(eval_support_parts, axis=0).reshape(-1, 3, na),
        list(bundle.actions), costs, budgets,
        bool(budget_model.get("shared_scheduler_required", False)),
    )


def evaluate_joint_profile(
    prepared: JointInputs,
    profile: JointProfile,
    *,
    reference_sla_threshold: float,
    include_decisions: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    n = len(prepared.budgets)
    if n < 1:
        raise ValueError("empty aligned dataset")
    row = np.arange(n)[:, None]
    slice_idx = np.arange(3)[None, :]
    baseline = prepared.reference_idx.copy()
    selected = baseline.copy()
    kept = 0
    proposed = 0
    no_feasible = 0
    gain_rejected = 0
    before_budget = prepared.prb_costs[baseline].sum(axis=1)
    baseline_budget_ok = before_budget <= prepared.budgets
    # Independent evaluator must have support on all selected actions;
    # otherwise do NOT force a novel policy based on imagined OPE values.
    for g in range(n):
        if profile.name == "original_agentic_ran":
            continue
        slice_eligible = prepared.eligible[g] & prepared.evaluator_supported[g]
        per_action = prepared.policy_score[g].copy()
        reference_indices = baseline[g]
        # The penalty is only a conservative *proxy*, NOT a statistically
        # calibrated action-advantage LCB. The same term is applied to the
        # selected and original actions before comparing them.
        per_action -= profile.uncertainty_penalty * prepared.policy_uncertainty[g]
        per_action -= profile.switching_penalty * (
            np.arange(len(prepared.actions))[None, :]
            != prepared.incumbent_idx[g, :, None]
        )
        schedulers = (
            tuple(str(action).split(":prb=", 1)[0] for action in prepared.actions)
            if prepared.shared_scheduler_required else None
        )
        chosen, best_obj = exact_multiple_choice_knapsack(
            per_action / 3.0, slice_eligible,
            prepared.prb_costs, int(prepared.budgets[g]),
            scheduler_names=schedulers,
        )
        if chosen is None:
            no_feasible += 1
            continue
        proposed += 1
        reference_obj = float(np.mean(
            per_action[np.arange(3), reference_indices]
        ))
        advantage_proxy = best_obj - reference_obj
        if not np.isfinite(advantage_proxy) or (
            advantage_proxy < profile.minimum_advantage - 1e-12
        ):
            gain_rejected += 1
            continue
        if np.array_equal(chosen, reference_indices):
            continue
        selected[g] = chosen
        kept += 1
    target_dm = prepared.evaluator_utility[row, slice_idx, selected]
    ref_dm = prepared.evaluator_utility[row, slice_idx, baseline]
    target_sla = prepared.evaluator_sla[row, slice_idx, selected]
    ref_sla = prepared.evaluator_sla[row, slice_idx, baseline]
    changed = np.any(selected != baseline, axis=1)
    target_joint = target_dm.mean(axis=1)
    ref_joint = ref_dm.mean(axis=1)
    ref_violation = float(np.mean(ref_sla > reference_sla_threshold))
    current_violation = float(np.mean(target_sla > reference_sla_threshold))
    selected_budget_ok = prepared.prb_costs[selected].sum(axis=1) <= prepared.budgets
    metrics = {
        "profile": profile.name,
        "aligned_joint_groups": int(n),
        "represented_slice_rows": int(n * 3),
        "source_commag_rows": int(prepared.alignment["original_rows"]),
        "aligned_fraction_of_base_station_seconds": prepared.alignment["alignment_rate"],
        "joint_selection_rate": float(np.mean(changed)),
        "proposed_feasible_group_rate": float(proposed / n),
        "no_feasible_joint_action_rate": float(no_feasible / n),
        "advantage_gate_rejection_rate": float(gain_rejected / n),
        "training_proxy_budget_compliance": float(np.mean(selected_budget_ok)),
        "original_budget_proxy_compliance": float(np.mean(baseline_budget_ok)),
        "independent_dm_utility": float(target_joint.mean()),
        "original_independent_dm_utility": float(ref_joint.mean()),
        "independent_dm_delta": float((target_joint - ref_joint).mean()),
        "estimated_sla_threshold_exceedance": current_violation,
        "original_estimated_sla_threshold_exceedance": ref_violation,
        "estimated_sla_delta": float(current_violation - ref_violation),
        "estimand": "independent_model_based_direct_method_on_aligned_BS_seconds_only",
        "realized_control_effect": False,
    }
    if not include_decisions:
        return metrics, None
    group_meta = prepared.representative.iloc[::3][
        list(GROUP_KEYS) + ["joint_group"]
    ].copy().reset_index(drop=True)
    group_meta["cross_slice_dm_delta"] = target_joint - ref_joint
    group_meta["joint_selected"] = changed
    group_meta["budget_proxy"] = prepared.budgets
    group_meta["selected_budget_proxy_ok"] = selected_budget_ok
    for sl, i in (("eMBB", 0), ("mMTC", 1), ("URLLC", 2)):
        names = np.asarray(prepared.actions, dtype=object)
        group_meta[f"{sl}_original_action"] = names[baseline[:, i]]
        group_meta[f"{sl}_joint_action"] = names[selected[:, i]]
    return metrics, group_meta


def paired_joint_clusters(
    group_evidence: pd.DataFrame,
    *,
    seed: int, bootstrap_samples: int = 1000,
    permutation_samples: int = 2000,
) -> dict[str, Any]:
    grouped = group_evidence.groupby(
        ["scenario", "training_config", "experiment"], sort=True, observed=True,
    )["cross_slice_dm_delta"].mean()
    data = grouped.to_numpy(dtype=float)
    if not len(data):
        raise ValueError("no independent experiment clusters")
    result = {
        "clusters": int(len(data)),
        "unweighted_mean_cluster_delta": float(data.mean()),
        "source": "independent_DM_not_observed_causal_outcome",
    }
    if len(data) < 2:
        result.update({"ci95": None, "paired_sign_randomization_p": None})
        return result
    rng = np.random.default_rng(seed)
    boot = np.empty(bootstrap_samples)
    for i in range(bootstrap_samples):
        boot[i] = data[rng.integers(0, len(data), len(data))].mean()
    pcount = 0
    observed = abs(data.mean())
    for _ in range(permutation_samples):
        signs = rng.choice((-1.0, 1.0), size=len(data))
        if abs(np.mean(signs * data)) >= observed - 1e-12:
            pcount += 1
    result["ci95"] = [float(x) for x in np.quantile(boot, [.025, .975])]
    result["paired_sign_randomization_p"] = float(
        (1 + pcount) / (1 + permutation_samples)
    )
    return result


def validation_model_headroom(
    prepared: JointInputs, *, relative_target: float = .10,
) -> dict[str, Any]:
    """Optimistic validation-only independent-DM diagnostic.

    Deliberately chooses the best action using evaluator outputs, so the
    result is NOT an executable policy, a statistical/causal upper bound
    or a permitted selection rule. It diagnoses whether the desired
    relative improvement is even visible in this biased scoring model.
    """
    if not 0 < relative_target < 1:
        raise ValueError("relative target must be between zero and one")
    reference = prepared.evaluator_utility[
        np.arange(len(prepared.budgets))[:, None],
        np.arange(3)[None, :],
        prepared.reference_idx,
    ].mean(axis=1)
    optimistic = reference.copy()
    feasible = 0
    for g in range(len(reference)):
        eligible = prepared.eligible[g] & prepared.evaluator_supported[g]
        schedulers = (
            tuple(str(action).split(":prb=", 1)[0] for action in prepared.actions)
            if prepared.shared_scheduler_required else None
        )
        candidate, predicted = exact_multiple_choice_knapsack(
            prepared.evaluator_utility[g] / 3.0, eligible,
            prepared.prb_costs, int(prepared.budgets[g]),
            scheduler_names=schedulers,
        )
        if candidate is None:
            continue
        feasible += 1
        # Best model-predicted joint assignment is also allowed to retain
        # the original assignment; never diagnose negative headroom.
        optimistic[g] = max(float(reference[g]), predicted)
    baseline = float(reference.mean())
    ceiling = float(optimistic.mean())
    return {
        "validation_only": True,
        "independent_DM_reference_mean": baseline,
        "optimistic_evaluator_selected_mean": ceiling,
        "optimistic_model_headroom_absolute": ceiling - baseline,
        "optimistic_model_headroom_relative": (
            (ceiling - baseline) / abs(baseline)
            if abs(baseline) > 1e-9 else None
        ),
        "target_relative_gain": relative_target,
        "target_comparison_score": baseline * (1.0 + relative_target),
        "model_headroom_reaches_target": (
            ceiling >= baseline * (1.0 + relative_target)
            if baseline >= 0 else False
        ),
        "feasible_joint_group_rate": feasible / max(len(reference), 1),
        "interpretation": (
            "optimistic validation model sensitivity using evaluator to pick "
            "actions: NOT deployable policy, causal upper bound or held-out "
            "success prediction. Do not optimize policy on these oracle choices."
        ),
    }


def run_joint_slice_fold(
    bundle: Any, evaluator: Any,
    policy_fit: pd.DataFrame, validation: pd.DataFrame,
    seen: pd.DataFrame, unseen: pd.DataFrame,
    calibrated: Any, switch_margin: float,
    *, fold: str, seed: int,
    maximum_sla_regression: float = .005,
    minimum_validation_gain: float = .002,
) -> dict[str, Any]:
    """Select from predefined profiles on validation; freeze before final test."""
    config_path = os.environ.get("AGENTIC_RAN_NR_CELL_CONFIG", "")
    if config_path:
        from agentic_ran.research_physical_prb import commag_physical_budget_config

        budgets = commag_physical_budget_config(config_path)
    else:
        # Historical offline result is retained for reproducibility ONLY.
        # It is explicitly NOT evidence of physically feasible control.
        budgets = training_prb_budgets(policy_fit)
        budgets["physical_capacity_confirmed"] = False
        budgets["shared_scheduler_required"] = False
    val = prepare_joint_inputs(
        bundle, evaluator, validation, calibrated, switch_margin, budgets,
    )
    model_headroom = validation_model_headroom(val)
    validation_results = []
    for profile in PROFILES:
        result, _ = evaluate_joint_profile(
            val, profile, reference_sla_threshold=calibrated.sla_threshold,
        )
        validation_results.append({
            "fold": fold, "split": "validation", **result,
        })
    anchor = validation_results[0]
    eligible = [
        item for item in validation_results[1:]
        if (
            item["independent_dm_delta"] >= minimum_validation_gain
            and item["estimated_sla_delta"] <= maximum_sla_regression
            and item["training_proxy_budget_compliance"] >=
            anchor["training_proxy_budget_compliance"] - 1e-12
        )
    ]
    if eligible:
        selected = max(
            eligible, key=lambda d: (
                d["independent_dm_delta"],
                -d["estimated_sla_threshold_exceedance"],
                -d["joint_selection_rate"],
            ),
        )
        selection_reason = "validation_estimated_gain_and_sla_budget_constraints"
    else:
        selected = anchor
        selection_reason = "no_validated_joint_gain_baseline_retained"
    profile = next(p for p in PROFILES if p.name == selected["profile"])
    tests: list[dict[str, Any]] = []
    evidence: list[pd.DataFrame] = []
    inference = {}
    alignment = {"validation": val.alignment}
    for i, (split_name, data) in enumerate((("seen", seen), ("unseen", unseen))):
        prepared = prepare_joint_inputs(
            bundle, evaluator, data, calibrated, switch_margin, budgets,
        )
        result, groups = evaluate_joint_profile(
            prepared, profile, reference_sla_threshold=calibrated.sla_threshold,
            include_decisions=True,
        )
        assert groups is not None
        tests.append({
            "fold": fold, "split": split_name, **result,
        })
        alignment[split_name] = prepared.alignment
        groups["fold"] = fold
        groups["split"] = split_name
        evidence.append(groups)
        inference[split_name] = paired_joint_clusters(
            groups, seed=seed + i * 31,
        )
    return {
        "fold": fold,
        "budget": budgets,
        "validation_headroom": model_headroom,
        "validation": validation_results,
        "final_tests": tests,
        "decisions": pd.concat(evidence, ignore_index=True),
        "alignment": alignment,
        "paired_clusters": inference,
        "selected_profile": profile.name,
        "selection_reason": selection_reason,
        "selection_uses_final_test": False,
        "minimum_validation_gain": minimum_validation_gain,
        "maximum_validation_sla_regression": maximum_sla_regression,
        "methodology": (
            "offline joint base-station slice ranking subject to a train-only "
            "PRB-budget proxy and policy-side advantage gate; independent DM "
            "for validation and held-out comparison; no executed actions"
        ),
    }
