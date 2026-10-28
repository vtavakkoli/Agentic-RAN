"""Experiment-level cross-fitted publication evaluation.

The configuration-level policy/OPE split used by the previous benchmark could
leave the independent evaluator without support for most selected actions.
This module fixes that positivity problem by role-swapping the two COMMAG
experiments and constraining every counterfactual method to common action
support before scoring.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from agentic_ran.offline_policy import (
    StudyConfig,
    _derive_targets,
    train_bundle,
)
from agentic_ran.publication_crossfit_report import render_crossfit_report
from agentic_ran.publication_science import (
    FinalScienceConfig,
    _augment_shortcut,
    _evaluate_split,
    _ood_generalization,
    _paired_group_statistics,
    fit_independent_ope,
    latency_profile,
    ood_detection_metrics,
    transition_audit,
    tune_cql_alpha,
    calibrate_controller,
)
from agentic_ran.publication_v2 import PubConfig, fit_cql, fit_fqi, shortcut
from agentic_ran.publication_support import StateConditionalSupportAudit
from agentic_ran.selective_evidence import (
    accounting_summary, combine_chunks, paired_cluster_inference,
)

COMMON_SUPPORT_GATE = 0.80
MIN_COMMON_ACTIONS_PER_SLICE = 2


def _action_cells(frame: pd.DataFrame) -> set[str]:
    return set(frame["slice_type"].astype(str) + "|" + frame["action"].astype(str))


def _support_by_slice(frame: pd.DataFrame) -> dict[str, set[str]]:
    return {
        str(slice_type): set(group["action"].astype(str))
        for slice_type, group in frame.groupby("slice_type", sort=True)
    }


def experiment_crossfit_plan(train: pd.DataFrame) -> dict[str, Any]:
    experiments = sorted(train["experiment"].astype(str).unique())
    if len(experiments) < 2:
        raise ValueError("experiment cross-fitting requires at least two experiments")
    all_cells = _action_cells(train)
    folds: list[dict[str, Any]] = []
    for policy_experiment in experiments:
        ope_experiments = [item for item in experiments if item != policy_experiment]
        policy = train[train["experiment"].astype(str).eq(policy_experiment)]
        ope = train[train["experiment"].astype(str).isin(ope_experiments)]
        if policy.empty or ope.empty:
            raise ValueError("every cross-fit fold requires non-empty policy and OPE roles")
        policy_support = _support_by_slice(policy)
        ope_support = _support_by_slice(ope)
        common_support: dict[str, list[str]] = {}
        for slice_type in sorted(set(policy_support) | set(ope_support)):
            common = sorted(
                policy_support.get(slice_type, set()) & ope_support.get(slice_type, set())
            )
            if not common:
                raise ValueError(
                    f"no common policy/OPE support for slice {slice_type} "
                    f"in {policy_experiment}"
                )
            common_support[slice_type] = common
        policy_cells = _action_cells(policy)
        ope_cells = _action_cells(ope)
        common_cells = {
            f"{slice_type}|{action}"
            for slice_type, actions in common_support.items()
            for action in actions
        }
        folds.append(
            {
                "fold": f"policy_{policy_experiment}",
                "policy_fit_experiment": policy_experiment,
                "ope_fit_experiments": ope_experiments,
                "policy_fit_rows": int(len(policy)),
                "ope_fit_rows": int(len(ope)),
                "all_action_cells": int(len(all_cells)),
                "policy_action_cells": int(len(policy_cells)),
                "ope_action_cells": int(len(ope_cells)),
                "common_action_cells": int(len(common_cells)),
                "policy_action_coverage": len(policy_cells & all_cells)
                / max(len(all_cells), 1),
                "ope_action_coverage": len(ope_cells & all_cells)
                / max(len(all_cells), 1),
                "common_policy_action_coverage": len(common_cells & policy_cells)
                / max(len(policy_cells), 1),
                "common_ope_action_coverage": len(common_cells & ope_cells)
                / max(len(ope_cells), 1),
                "minimum_common_actions_per_slice": min(
                    len(actions) for actions in common_support.values()
                ),
                "common_support_by_slice": common_support,
            }
        )
    return {
        "strategy": "experiment_role_swap_crossfit",
        "routing_rule": (
            "each final-test experiment is scored once by a policy fitted on that "
            "experiment and an independent OPE evaluator fitted on the other experiment(s)"
        ),
        "experiments": experiments,
        "all_action_cells": len(all_cells),
        "common_support_gate": COMMON_SUPPORT_GATE,
        "minimum_common_actions_per_slice_gate": MIN_COMMON_ACTIONS_PER_SLICE,
        "folds": folds,
    }


def _common_support(
    policy_fit: pd.DataFrame,
    ope_fit: pd.DataFrame,
) -> dict[str, set[str]]:
    policy = _support_by_slice(policy_fit)
    ope = _support_by_slice(ope_fit)
    result: dict[str, set[str]] = {}
    for slice_type in sorted(set(policy) | set(ope)):
        common = policy.get(slice_type, set()) & ope.get(slice_type, set())
        if not common:
            raise ValueError(f"empty common support for slice {slice_type}")
        result[slice_type] = common
    return result


def _fallback_by_slice(
    policy_fit: pd.DataFrame,
    common_support: dict[str, set[str]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for slice_type, supported in common_support.items():
        local = policy_fit[
            policy_fit["slice_type"].astype(str).eq(slice_type)
            & policy_fit["action"].astype(str).isin(supported)
        ]
        if local.empty:
            result[slice_type] = sorted(supported)[0]
            continue
        counts = local["action"].astype(str).value_counts()
        top = counts[counts.eq(counts.max())].index.astype(str)
        result[slice_type] = sorted(top)[0]
    return result


def _constrain_bundle_to_common_support(
    bundle: Any,
    policy_fit: pd.DataFrame,
    ope_fit: pd.DataFrame,
) -> tuple[dict[str, set[str]], dict[str, str], dict[str, Any]]:
    common = _common_support(policy_fit, ope_fit)
    fallback = _fallback_by_slice(policy_fit, common)
    policy_cells = _action_cells(policy_fit)
    common_cells = {
        f"{slice_type}|{action}"
        for slice_type, actions in common.items()
        for action in actions
    }
    audit = {
        "common_policy_action_coverage": len(common_cells & policy_cells)
        / max(len(policy_cells), 1),
        "minimum_common_actions_per_slice": min(len(values) for values in common.values()),
        "common_support_by_slice": {key: sorted(value) for key, value in common.items()},
        "fallback_by_slice": fallback,
    }
    # OfflinePolicyBundle is mutable. Restricting actions_by_slice makes the
    # existing proposed/FQI/CQL evaluation path obey the same positivity set.
    bundle.actions_by_slice = {key: set(value) for key, value in common.items()}
    bundle.fallback_by_slice = dict(fallback)
    return common, fallback, audit


def _aggregate_baselines(fold_rows: list[pd.DataFrame]) -> pd.DataFrame:
    table = pd.concat(fold_rows, ignore_index=True)
    metrics = [
        "selected_ope_mean_utility",
        "selected_ope_sla_violation_rate",
        "selected_ope_mean_energy_proxy",
        "selected_ope_mean_stability_cost",
        "policy_churn_rate",
        "logged_action_disagreement_rate",
        "replay_transition_count",
        "evaluator_support_rate",
        "rollback_rate",
        "ood_fallback_rate",
        "support_deferral_rate",
        "unsupported_incumbent_fallback_rate",
    ]
    rows: list[dict[str, Any]] = []
    for (split, model), group in table.groupby(["split", "model"], sort=False):
        weights = group["fold_rows"].to_numpy(dtype=float)
        row: dict[str, Any] = {
            "split": split,
            "model": model,
            "estimate_type": str(group["estimate_type"].iloc[0]),
            "rows": int(weights.sum()),
        }
        for metric in metrics:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(values)
            metric_weights = (
                pd.to_numeric(
                    group["replay_transition_count"], errors="coerce"
                ).to_numpy(dtype=float)
                if metric == "policy_churn_rate"
                else weights
            )
            eligible = finite & np.isfinite(metric_weights) & (metric_weights > 0)
            row[metric] = (
                float(np.average(values[eligible], weights=metric_weights[eligible]))
                if eligible.any()
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_latency(items: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    if not items:
        return {}
    result: dict[str, Any] = {}
    for key in (
        "critic_prediction_batch_ms_per_row",
        "ood_batch_ms_per_row",
        "selection_logic_batch_ms_per_row",
        "end_to_end_p50_ms",
        "end_to_end_p95_ms",
        "end_to_end_p99_ms",
    ):
        values = [float(item[key]) for _, item in items if key in item]
        if values:
            result[key] = (
                max(values)
                if key in {"end_to_end_p95_ms", "end_to_end_p99_ms"}
                else float(np.mean(values))
            )
    first = items[0][1]
    for key in ("python", "scikit_learn", "cpu_count", "platform"):
        if key in first:
            result[key] = first[key]
    result["note"] = (
        "cross-fit aggregate; p95/p99 are conservative maxima across experiment folds. "
        "These are host/container inference measurements, not RIC-to-gNB latency."
    )
    return result


def _tradeoff_summary(
    baselines: pd.DataFrame,
    clustered: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("seen", "unseen"):
        local = baselines[baselines["split"].eq(split)].set_index("model")
        if "proposed" not in local.index:
            continue
        proposed = local.loc["proposed"]
        result[split] = {}
        for baseline in (
            "fqi", "cql_linear", "behavior_hgb",
            "fqi_matched", "cql_linear_matched", "behavior_hgb_matched",
        ):
            if baseline not in local.index:
                continue
            other = local.loc[baseline]
            result[split][baseline] = {
                "utility_delta": float(
                    proposed["selected_ope_mean_utility"]
                    - other["selected_ope_mean_utility"]
                ),
                # Keep the legacy key for downstream consumers, but add
                # explicit semantic names so replay switching cannot be
                # confused with disagreement with the original log.
                "churn_reduction_absolute": float(
                    other["policy_churn_rate"] - proposed["policy_churn_rate"]
                ),
                "open_loop_replay_switch_reduction_absolute": float(
                    other["policy_churn_rate"] - proposed["policy_churn_rate"]
                ),
                "logged_action_disagreement_reduction_absolute": float(
                    other["logged_action_disagreement_rate"]
                    - proposed["logged_action_disagreement_rate"]
                ),
                "sla_violation_delta": float(
                    proposed["selected_ope_sla_violation_rate"]
                    - other["selected_ope_sla_violation_rate"]
                ),
                "clustered_effect": clustered.get(
                    f"{split}:proposed_vs_{baseline}", {}
                ),
            }
    return result


def run_crossfit_benchmark(
    data_path: str | Path,
    output: str | Path,
    pub_cfg: PubConfig,
    science: FinalScienceConfig,
    *,
    seed: int | None = None,
    literature_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    seed = science.random_seed if seed is None else int(seed)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(data_path, low_memory=False)
    train = frame[frame["publication_split"].eq("train")].copy()
    validation = frame[frame["publication_split"].eq("validation")].copy()
    seen = frame[frame["publication_split"].eq("test_seen")].copy()
    unseen = frame[frame["publication_split"].eq("test_unseen")].copy()
    if min(map(len, (train, validation, seen, unseen))) == 0:
        raise ValueError("all train/validation/seen/unseen splits must be non-empty")

    plan = experiment_crossfit_plan(train)
    # Opt in explicitly; the original publication comparison stays unchanged.
    experimental_guard_enabled = (
        os.environ.get("AGENTIC_RAN_EXPERIMENTAL_SUPPORT_GUARD", "0") == "1"
    )
    matched_wrappers_enabled = (
        os.environ.get("AGENTIC_RAN_EVAL_MATCHED_WRAPPERS", "0") == "1"
    )
    conformal_rerank_enabled = (
        os.environ.get("AGENTIC_RAN_EVAL_CONFORMAL_RERANK", "0") == "1"
    )
    conformal_selection = []
    conformal_validation = []
    conformal_tests = []
    conformal_cluster_rows = []
    conformal_slice_rows = []
    fold_baselines: list[pd.DataFrame] = []
    decision_tables: list[pd.DataFrame] = []
    comparison_models = ["proposed", "fqi", "cql_linear", "behavior_hgb"]
    if experimental_guard_enabled:
        comparison_models.append("support_guarded")
    if matched_wrappers_enabled:
        comparison_models.extend(
            ["fqi_matched", "cql_linear_matched", "behavior_hgb_matched"]
        )
    details_acc: dict[str, dict[str, list[pd.DataFrame]]] = {
        split: {model: [] for model in comparison_models}
        for split in ("seen", "unseen")
    }
    calibration_tables: list[pd.DataFrame] = []
    cql_tables: list[pd.DataFrame] = []
    calibration_selection: dict[str, Any] = {}
    evaluator_fit: dict[str, Any] = {}
    model_folds: dict[str, Any] = {}
    latency_items: list[tuple[str, dict[str, Any]]] = []
    fold_metadata: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    state_overlap_rows: list[dict[str, Any]] = []

    for fold_index, base_fold in enumerate(plan["folds"]):
        fold = dict(base_fold)
        fold_name = str(fold["fold"])
        policy_experiment = str(fold["policy_fit_experiment"])
        ope_experiments = [str(value) for value in fold["ope_fit_experiments"]]
        policy_fit = train[train["experiment"].astype(str).eq(policy_experiment)].copy()
        ope_fit = train[train["experiment"].astype(str).isin(ope_experiments)].copy()
        validation_fold = validation[
            validation["experiment"].astype(str).eq(policy_experiment)
        ].copy()
        seen_fold = seen[seen["experiment"].astype(str).eq(policy_experiment)].copy()
        unseen_fold = unseen[unseen["experiment"].astype(str).eq(policy_experiment)].copy()
        if min(map(len, (policy_fit, ope_fit, validation_fold, seen_fold, unseen_fold))) == 0:
            raise ValueError(f"empty evidence role in {fold_name}")
        if set(policy_fit["episode_id"].astype(str)) & set(ope_fit["episode_id"].astype(str)):
            raise ValueError(f"policy/OPE episode overlap in {fold_name}")

        bundle = train_bundle(
            policy_fit.assign(split="train"),
            StudyConfig(),
            seed + fold_index * 10000,
        )
        _, _, support_audit = _constrain_bundle_to_common_support(
            bundle, policy_fit, ope_fit
        )
        fold.update(support_audit)
        # This local kNN audit is deliberately separate from the action-only
        # common support mask. It diagnoses, but cannot prove, conditional
        # counterfactual overlap or causal identifiability.
        local_overlap = StateConditionalSupportAudit(policy_fit, ope_fit)

        evaluator = fit_independent_ope(
            ope_fit,
            bundle.actions,
            bundle.config,
            science,
            seed + 1000 + fold_index * 10000,
        )
        calibrated, switch_margin, calibration_table, selection = calibrate_controller(
            bundle, validation_fold, evaluator, science
        )
        calibration_table.insert(0, "crossfit_fold", fold_name)
        calibration_tables.append(calibration_table)
        calibration_selection[fold_name] = selection

        policy_targets = _derive_targets(policy_fit, bundle.energy_scale, bundle.config)
        selected_alpha, cql_table = tune_cql_alpha(
            policy_targets,
            validation_fold,
            bundle,
            evaluator,
            pub_cfg,
            science,
            seed + fold_index * 10000,
        )
        cql_table.insert(0, "crossfit_fold", fold_name)
        cql_tables.append(cql_table)
        fqi = fit_fqi(
            policy_targets,
            bundle,
            pub_cfg,
            seed + 2000 + fold_index * 10000,
        )
        cql = fit_cql(
            policy_targets,
            bundle,
            replace(pub_cfg, cql_alpha=selected_alpha),
            seed + 3000 + fold_index * 10000,
        )

        if conformal_rerank_enabled:
            from agentic_ran.publication_research import run_optional_conformal_fold

            trial = run_optional_conformal_fold(
                bundle, evaluator, policy_fit, ope_fit, validation_fold,
                seen_fold, unseen_fold, calibrated,
                seed=seed + 12000 + fold_index, fold=fold_name,
                switch_margin=switch_margin,
            )
            conformal_selection.append(trial["selection"])
            conformal_validation.extend(trial["validation_grid"])
            conformal_tests.extend(trial["final_tests"])
            conformal_cluster_rows.extend(trial["cluster_evidence"])
            conformal_slice_rows.extend(trial["slice_scenario"])

        for split_name, test in (("seen", seen_fold), ("unseen", unseen_fold)):
            fold_ablations: list[dict[str, Any]] = []
            summary_table, details, combined = _evaluate_split(
                split_name,
                test,
                bundle,
                evaluator,
                calibrated,
                switch_margin,
                fqi,
                cql,
                ablation_sink=fold_ablations,
                state_support_guard=(
                    local_overlap if experimental_guard_enabled else None
                ),
                evaluate_matched_wrappers=matched_wrappers_enabled,
            )
            for row in fold_ablations:
                row["crossfit_fold"] = fold_name
            ablation_rows.extend(fold_ablations)
            state_overlap_rows.extend(local_overlap.audit(
                test,
                {method: item["selected_action"].to_numpy()
                 for method, item in details.items()},
                split=split_name,
                fold=fold_name,
            ))
            summary_table["fold_rows"] = len(test)
            summary_table["crossfit_fold"] = fold_name
            summary_table["policy_fit_experiment"] = policy_experiment
            summary_table["ope_fit_experiments"] = ",".join(ope_experiments)
            fold_baselines.append(summary_table)
            combined["crossfit_fold"] = fold_name
            decision_tables.append(combined)
            for model, detail in details.items():
                detail["crossfit_fold"] = fold_name
                details_acc[split_name][model].append(detail)

        latency_items.append(
            (
                fold_name,
                latency_profile(bundle, seen_fold, calibrated, switch_margin, science),
            )
        )
        evaluator_fit[fold_name] = evaluator.fit_summary
        fold["calibrated_controller"] = {
            "sla_threshold": calibrated.sla_threshold,
            "uncertainty_threshold": calibrated.uncertainty_threshold,
            "ood_threshold": calibrated.ood_threshold,
            "planning_weight": calibrated.planning_weight,
            "switch_margin": switch_margin,
            "selected_cql_alpha": selected_alpha,
        }
        fold_metadata.append(fold)
        model_folds[fold_name] = {
            "policy_bundle": bundle,
            "independent_ope": evaluator,
            "fqi": fqi,
            "cql": cql,
            "calibrated_controller": fold["calibrated_controller"],
        }

    baselines = _aggregate_baselines(fold_baselines)
    decisions = pd.concat(decision_tables, ignore_index=True)
    conformal_clustered = {}
    conformal_accounting = {}
    if conformal_rerank_enabled:
        if not conformal_cluster_rows:
            raise RuntimeError("conformal evidence was enabled but no clusters were exported")
        for split_index, (split, source) in enumerate(
            (("seen", seen), ("unseen", unseen))
        ):
            records = [
                row for row in conformal_cluster_rows if row["split"] == split
            ]
            # Additive accounting across fold-specific, bounded-size batches.
            grouped = combine_chunks(records)
            if int(grouped["rows"].sum()) != len(source):
                raise RuntimeError(
                    f"conformal {split} accounting does not cover final rows once"
                )
            conformal_accounting[split] = accounting_summary(grouped)
            conformal_clustered[split] = {
                estimand: paired_cluster_inference(
                    grouped, estimand,
                    seed=seed + 20000 + split_index * 20 + j,
                    bootstrap_samples=science.bootstrap_samples,
                    permutation_samples=science.permutation_samples,
                )
                for j, estimand in enumerate((
                    "fixed_penalty", "hold_incumbent_dm",
                    "observed_logged_hold_proxy",
                ))
            }
    ablations = pd.DataFrame(ablation_rows)
    state_overlap = pd.DataFrame(state_overlap_rows)
    overlap_summary = {}
    for (split, method), group in state_overlap.groupby(["split", "method"]):
        weights = group["sampled_rows"].to_numpy(float)
        overlap_summary[f"{split}:{method}"] = {
            "sampled_rows": int(weights.sum()),
            "joint_local_overlap": float(np.average(
                group["joint_local_overlap"].to_numpy(float), weights=weights
            )),
            "limitation": "sampled kNN state overlap proxy, not causal identification",
        }
    details = {
        split: {
            model: pd.concat(parts, ignore_index=True)
            for model, parts in models.items()
        }
        for split, models in details_acc.items()
    }
    if len(details["seen"]["proposed"]) != len(seen):
        raise RuntimeError("seen final-test rows were not routed exactly once")
    if len(details["unseen"]["proposed"]) != len(unseen):
        raise RuntimeError("unseen final-test rows were not routed exactly once")

    clustered: dict[str, Any] = {}
    episode_secondary: dict[str, Any] = {}
    for split_index, split in enumerate(("seen", "unseen")):
        proposed = details[split]["proposed"]
        for offset, baseline in enumerate(("fqi", "cql_linear", "behavior_hgb")):
            key = f"{split}:proposed_vs_{baseline}"
            clustered[key] = _paired_group_statistics(
                proposed,
                details[split][baseline],
                science,
                seed + 5000 + split_index * 100 + offset,
                "cluster_id",
            )
            episode_secondary[key] = _paired_group_statistics(
                proposed,
                details[split][baseline],
                science,
                seed + 6000 + split_index * 100 + offset,
                "episode_id",
            )
        if matched_wrappers_enabled:
            for offset, model in enumerate(
                ("fqi_matched", "cql_linear_matched", "behavior_hgb_matched")
            ):
                key = f"{split}:proposed_vs_{model}"
                clustered[key] = _paired_group_statistics(
                    proposed, details[split][model], science,
                    seed + 9000 + split_index * 100 + offset, "cluster_id",
                )
                episode_secondary[key] = _paired_group_statistics(
                    proposed, details[split][model], science,
                    seed + 9200 + split_index * 100 + offset, "episode_id",
                )
        if experimental_guard_enabled:
            guard_key = f"{split}:support_guarded_vs_proposed"
            clustered[guard_key] = _paired_group_statistics(
                details[split]["support_guarded"], proposed, science,
                seed + 8000 + split_index, "cluster_id",
            )
            episode_secondary[guard_key] = _paired_group_statistics(
                details[split]["support_guarded"], proposed, science,
                seed + 8100 + split_index, "episode_id",
            )

    proposed_seen = details["seen"]["proposed"]
    proposed_unseen = details["unseen"]["proposed"]
    threshold = float(
        np.median(
            [
                item["calibrated_controller"]["ood_threshold"]
                for item in fold_metadata
            ]
        )
    )
    ood_detection, ood_sensitivity = ood_detection_metrics(
        proposed_seen, proposed_unseen, threshold
    )
    ood_generalization = _ood_generalization(
        proposed_seen, proposed_unseen, threshold
    )

    audit_table, audit_summary = transition_audit(
        pd.concat([seen, unseen], ignore_index=True)
    )
    shortcut_tables = []
    for split, test in (("seen", seen), ("unseen", unseen)):
        _, split_audit = transition_audit(test)
        diagnostic = shortcut(train, test, seed)
        diagnostic.insert(0, "split", split)
        shortcut_tables.append(_augment_shortcut(diagnostic, split_audit))
    shortcut_table = pd.concat(shortcut_tables, ignore_index=True)

    latency = _aggregate_latency(latency_items)
    tradeoffs = _tradeoff_summary(baselines, clustered)
    support_seen = float(
        baselines[
            baselines["split"].eq("seen") & baselines["model"].eq("proposed")
        ]["evaluator_support_rate"].iloc[0]
    )
    support_unseen = float(
        baselines[
            baselines["split"].eq("unseen") & baselines["model"].eq("proposed")
        ]["evaluator_support_rate"].iloc[0]
    )
    min_common_coverage = min(
        float(item["common_policy_action_coverage"]) for item in fold_metadata
    )
    min_common_actions = min(
        int(item["minimum_common_actions_per_slice"]) for item in fold_metadata
    )
    fit_finite = all(
        math.isfinite(float(metric[value]))
        for summary in evaluator_fit.values()
        for metric in summary["fit_metrics"].values()
        for value in ("oof_mae", "oof_r2")
    )
    roles_disjoint = all(
        str(item["policy_fit_experiment"])
        not in set(map(str, item["ope_fit_experiments"]))
        for item in fold_metadata
    )
    stat_clusters_ok = all(int(item["groups"]) >= 8 for item in clustered.values())

    readiness_gates = {
        "policy/OPE fitting roles are experiment-disjoint in every fold": roles_disjoint,
        "every final-test row is routed exactly once by experiment": (
            len(proposed_seen) == len(seen) and len(proposed_unseen) == len(unseen)
        ),
        "validation calibration completed without test data": all(
            not item["test_data_used_for_calibration"]
            for item in calibration_selection.values()
        ),
        "validation policy satisfies feasibility constraints": all(
            item["constraint_feasible"] for item in calibration_selection.values()
        ),
        "independent OPE cross-fit metrics are finite": fit_finite,
        f"common support retains >= {COMMON_SUPPORT_GATE:.0%} of policy action cells": (
            min_common_coverage >= COMMON_SUPPORT_GATE
        ),
        f"common support has >= {MIN_COMMON_ACTIONS_PER_SLICE} actions per slice": (
            min_common_actions >= MIN_COMMON_ACTIONS_PER_SLICE
        ),
        f"proposed evaluator support >= {science.evaluator_support_gate:.0%} on seen test": (
            support_seen >= science.evaluator_support_gate
        ),
        f"proposed evaluator support >= {science.evaluator_support_gate:.0%} on unseen test": (
            support_unseen >= science.evaluator_support_gate
        ),
        "primary inference uses >=8 experimental clusters per comparison": stat_clusters_ok,
        "OOD discrimination metrics computed on held-out seen vs unseen": math.isfinite(
            float(ood_detection["overall"]["auroc"])
        ),
        "transition/persistence audit completed": not audit_table.empty,
        "counterfactual outcomes use independent evaluator on common support": True,
    }

    warnings: list[str] = []
    claim_scope = {
        "primary": "offline support-constrained RAN configuration/policy selection",
        "dynamic_control": "not established by the current COMMAG transition evidence",
        "ood": "secondary diagnostic",
        "latency": "host/container inference only; not RIC-to-gNB latency",
        "churn": "open-loop selected-action changes across consecutive logged states, not realized actuation",
        "support": "slice-level action overlap plus a sampled state-conditional diagnostic; no causal guarantee",
    }
    if conformal_rerank_enabled:
        if any(not x.get("validation_feasible", False) for x in conformal_selection):
            warnings.append(
                "Conformal model selection found NO validation candidate with "
                "at least 5% action coverage in one or more folds. Its selected "
                "diagnostic method must not be called a functioning controller."
            )
        warnings.append(
            "The selective SLA gate now calibrates an episode-MEAN logged-action "
            "residual, not a worst-step bound. It supports offline average-risk "
            "screening only, and cannot certify per-action or per-step safety."
        )
        warnings.append(
            "Conformal comparison now uses episode-blocked calibrations and "
            "paired cluster summaries. Its abstention cases are assessed via "
            "explicit model-based holding and observational sensitivity "
            "analyses. Neither identifies the effect of doing nothing in a "
            "live RAN."
        )
        warnings.append(
            "Research conformal re-ranking is secondary evidence and may "
            "abstain. Selected-action utility and SLA are conditional direct-method "
            "estimates. The critic SLA prediction is NOT a calibrated upper "
            "confidence bound, and kNN/GMM/normal scores are NOT causal action support."
        )
    if matched_wrappers_enabled:
        warnings.append(
            "Matched-wrapper FQI, CQL and HGB baselines are experimental. "
            "They use the same learned critic and safety/hysteresis envelope "
            "as the proposed method; do not describe them as independent "
            "unmodified RL baselines."
        )
    if experimental_guard_enabled:
        warnings.append(
            "Experimental dual-role kNN support guard enabled. Abstention is "
            "not proof of causal identification or safe network actuation; "
            "compare its independent OPE utility and deferral rate against "
            "the unguarded proposed method before promoting it."
        )
    if any(
        summary["joint_local_overlap"] < 0.5
        for key, summary in overlap_summary.items()
        if key.endswith(":proposed")
    ):
        warnings.append(
            "Sampled state-conditional overlap is below 50% in at least one "
            "test split. Action-set support alone cannot validate direct-method "
            "counterfactual policy comparisons."
        )
    if not audit_summary["within_episode_change_evidence_available"]:
        warnings.append(
            "No within-episode joint scheduler+PRB action changes were observed; "
            "do not claim demonstrated dynamic closed-loop control. Frame results "
            "as offline RAN configuration/policy selection."
        )
    if float(ood_detection["overall"]["auroc"]) < 0.70:
        warnings.append(
            "OOD AUROC is below 0.70; treat OOD as a secondary conservative signal "
            "rather than a headline detection contribution."
        )
        claim_scope["ood"] = "secondary limited-discrimination diagnostic"
    if float(latency.get("end_to_end_p95_ms", 0.0)) > 500:
        warnings.append(
            "Host/container p95 decision latency exceeds 500 ms; avoid real-time "
            "RIC claims without deployment measurements."
        )
    if any(float(item.get("bootstrap_ci95_high", 0.0)) < 0 for item in clustered.values()):
        warnings.append(
            "Some primary comparisons have statistically lower scalar utility for "
            "the proposed method. Do not claim utility superiority; report the "
            "utility-versus-churn/safety trade-off."
        )
        claim_scope["utility"] = "trade-off result; superiority claim restricted"
    else:
        claim_scope["utility"] = "report clustered effects and confidence intervals"

    plan["folds"] = fold_metadata
    summary: dict[str, Any] = {
        "run_status": "EXPERIMENT-COMPLETE",
        "evidence_status": (
            "READY-FOR-MANUSCRIPT"
            if all(readiness_gates.values())
            else "REVIEW-REQUIRED"
        ),
        "policy_fit_rows": int(sum(item["policy_fit_rows"] for item in fold_metadata)),
        "ope_fit_rows": int(sum(item["ope_fit_rows"] for item in fold_metadata)),
        "validation_rows": len(validation),
        "seen_test_rows": len(seen),
        "unseen_test_rows": len(unseen),
        "partition": plan,
        "calibrated_controller": {
            "mode": "per-experiment cross-fit calibration",
            "folds": {
                str(item["fold"]): item["calibrated_controller"]
                for item in fold_metadata
            },
            "combined_ood_threshold_for_diagnostics": threshold,
        },
        "readiness_gates": readiness_gates,
        "warnings": warnings,
        "claim_scope": claim_scope,
        "tradeoff_summary": tradeoffs,
        "state_conditional_overlap": overlap_summary,
        "research_conformal_rerank": {
            "enabled": conformal_rerank_enabled,
            "research_evidence_status": (
                "OFFLINE-ANALYSIS-ONLY" if conformal_rerank_enabled
                else "NOT-RUN"
            ),
            "abstention_outcomes": {
                "fixed_penalty": "assumed -0.12",
                "hold_incumbent_dm": "OPE only where common-action support exists",
                "observed_logged_hold_proxy": "observational only when logged action equals incumbent",
            },
            "paired_cluster_statistics": conformal_clustered,
            "paired_accounting": conformal_accounting,
            "conformal_calibration_unit": "held_out_episode_maximum",
            "sla_calibration_estimand": "logged_action_episode_mean_residual_95pct",
            "all_validation_folds_feasible": (
                all(x.get("validation_feasible", False) for x in conformal_selection)
                if conformal_rerank_enabled else False
            ),
            "validation_only_selection": True,
            "fail_closed_no_action": True,
            "claim_scope": "simulated/offline candidate-ranking diagnostic, not network safety",
            "selections": conformal_selection,
            "publication_ready": False,
            "reason": (
                "no closed-loop RAN validation; abstention outcomes are "
                "sensitivity estimates, not identified causal interventions"
            ),
        },
        "experimental_matched_wrappers": {
            "enabled": matched_wrappers_enabled,
            "same_safety_hysteresis_ood_envelope": True,
            "not_unmodified_baselines": True,
        },
        "experimental_support_guard": {
            "enabled": experimental_guard_enabled,
            "evaluation_only": True,
            "training_roles_only": True,
            "local_support_proxy_not_causal": True,
            "results": {
                split: {
                    "deferral_rate": float(
                        baselines.loc[
                            baselines["split"].eq(split)
                            & baselines["model"].eq("support_guarded"),
                            "support_deferral_rate",
                        ].iloc[0]
                    ),
                    "utility": float(
                        baselines.loc[
                            baselines["split"].eq(split)
                            & baselines["model"].eq("support_guarded"),
                            "selected_ope_mean_utility",
                        ].iloc[0]
                    ),
                }
                for split in ("seen", "unseen")
            } if experimental_guard_enabled else {},
        },
        "ablation_variants": sorted(set(ablations["variant"])),
        "clustered_statistics": clustered,
        "ood_detection": ood_detection,
        "transition_audit": audit_summary,
        "limitations": [
            "alternative selected-action outcomes are independent direct-method/OPE estimates on common observed support, not causal online intervention effects",
            "energy is a normalized proxy rather than measured joules",
            "the original COMMAG PPO findings are literature-reference values and are not reproduced or used in paired tests",
            "primary inferential units are scenario/training-configuration/experiment clusters; UE-level statistics are secondary only",
            "counterfactual actions are restricted to policy/OPE common support; support coverage is reported explicitly",
        ],
    }
    if literature_reference is not None:
        summary["literature_reference"] = literature_reference

    calibration = pd.concat(calibration_tables, ignore_index=True)
    cql_validation = pd.concat(cql_tables, ignore_index=True)
    if conformal_rerank_enabled:
        pd.DataFrame(conformal_cluster_rows).to_csv(
            destination / "research_conformal_cluster_evidence.csv",
            index=False,
        )
        pd.DataFrame(conformal_slice_rows).to_csv(
            destination / "research_conformal_slice_scenario.csv",
            index=False,
        )
        (destination / "research_conformal_paired_statistics.json").write_text(
            json.dumps(conformal_clustered, indent=2), encoding="utf-8",
        )
        pd.DataFrame(conformal_validation).to_csv(
            destination / "research_conformal_validation.csv", index=False
        )
        pd.DataFrame(conformal_tests).to_csv(
            destination / "research_conformal_final_tests.csv", index=False
        )
    baselines.to_csv(destination / "publication_baselines.csv", index=False)
    ablations.to_csv(destination / "publication_ablations.csv", index=False)
    state_overlap.to_csv(destination / "conditional_support.csv", index=False)
    decisions.to_csv(
        destination / "publication_decisions.csv.gz",
        index=False,
        compression="gzip",
    )
    calibration.to_csv(destination / "validation_calibration.csv", index=False)
    cql_validation.to_csv(destination / "cql_validation.csv", index=False)
    audit_table.to_csv(destination / "transition_audit.csv", index=False)
    shortcut_table.to_csv(destination / "policy_shortcut_test.csv", index=False)
    ood_sensitivity.to_csv(destination / "ood_threshold_sensitivity.csv", index=False)
    for name, payload in (
        ("clustered_statistics.json", clustered),
        ("paired_statistics.json", clustered),
        ("episode_statistics_secondary.json", episode_secondary),
        ("ood_detection.json", ood_detection),
        ("ood_generalization.json", ood_generalization),
        ("validation_selection.json", calibration_selection),
        ("independent_ope_fit.json", evaluator_fit),
        ("latency_profile.json", latency),
        ("partition.json", plan),
        ("tradeoff_summary.json", tradeoffs),
    ):
        (destination / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if literature_reference is not None:
        (destination / "literature_reference.json").write_text(
            json.dumps(literature_reference, indent=2), encoding="utf-8"
        )
    joblib.dump(
        {"crossfit_strategy": plan["strategy"], "folds": model_folds},
        destination / "publication_models.joblib",
    )
    (destination / "publication_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    report = render_crossfit_report(
        destination,
        summary=summary,
        baselines=baselines,
        clustered_statistics=clustered,
        ood=ood_generalization,
        ood_detection=ood_detection,
        transition_audit=audit_table,
        shortcut=shortcut_table,
        calibration=calibration,
        latency=latency,
        literature_reference=literature_reference or {},
        crossfit_plan=plan,
    )
    summary["report_html"] = report.name
    summary["readiness_gates"]["self-contained HTML evidence report generated"] = (
        report.exists() and report.stat().st_size > 0
    )
    summary["evidence_status"] = (
        "READY-FOR-MANUSCRIPT"
        if all(summary["readiness_gates"].values())
        else "REVIEW-REQUIRED"
    )
    (destination / "publication_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    render_crossfit_report(
        destination,
        summary=summary,
        baselines=baselines,
        clustered_statistics=clustered,
        ood=ood_generalization,
        ood_detection=ood_detection,
        transition_audit=audit_table,
        shortcut=shortcut_table,
        calibration=calibration,
        latency=latency,
        literature_reference=literature_reference or {},
        crossfit_plan=plan,
    )
    return summary


run_final_benchmark = run_crossfit_benchmark
