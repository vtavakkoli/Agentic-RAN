"""Final scientific evaluation for the full COMMAG publication benchmark.

This module fixes the main inferential risks of the earlier benchmark:
- policy fitting and independent outcome/OPE fitting use disjoint training configs;
- validation is used to calibrate safety/OOD/planning/hysteresis parameters;
- test statistics are clustered by scenario/training-configuration/experiment;
- selected actions are scored by an independent outcome model, not the policy critics;
- OOD discrimination, transition persistence, shortcut sensitivity, and latency are audited;
- completion status is separated from manuscript-readiness gates.
"""
from __future__ import annotations

import json
import math
import os
import platform
import time
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold

from agentic_ran.offline_policy import (
    StudyConfig,
    _all_predictions,
    _candidate_matrix,
    _current_actions,
    _derive_targets,
    _fit_energy_scale,
    _normalize_q,
    _ood_scores,
    _state_matrix,
    _supported_mask,
    train_bundle,
)
from agentic_ran.publication_report import render_report
from agentic_ran.publication_v2 import PubConfig, fit_cql, fit_fqi, shortcut

CLUSTER_COLUMNS = ("scenario", "training_config", "experiment")


@dataclass(frozen=True)
class FinalScienceConfig:
    ope_config_count: int = 4
    evaluator_folds: int = 4
    evaluator_max_iter: int = 96
    target_validation_ood_fpr: float = 0.01
    reference_sla_threshold: float = 0.45
    sla_threshold_grid: tuple[float, ...] = (0.40, 0.45, 0.50)
    uncertainty_threshold_grid: tuple[float, ...] = (0.40, 0.50, 0.60)
    planning_weight_grid: tuple[float, ...] = (0.00, 0.03, 0.05)
    switch_margin_grid: tuple[float, ...] = (0.00, 0.005, 0.010, 0.020)
    cql_alpha_grid: tuple[float, ...] = (0.05, 0.15, 0.30)
    cql_tuning_epochs: int = 40
    bootstrap_samples: int = 5000
    permutation_samples: int = 10000
    calibration_sla_penalty: float = 0.15
    calibration_churn_penalty: float = 0.03
    calibration_rollback_penalty: float = 0.02
    max_validation_sla_regression: float = 0.01
    evaluator_support_gate: float = 0.95
    latency_samples: int = 24
    random_seed: int = 2026

    def validate(self) -> None:
        if self.ope_config_count < 2:
            raise ValueError("ope_config_count must be >=2")
        if self.evaluator_folds < 2:
            raise ValueError("evaluator_folds must be >=2")
        if not 0 < self.target_validation_ood_fpr < 0.5:
            raise ValueError("target_validation_ood_fpr must be in (0,0.5)")
        if self.bootstrap_samples < 200 or self.permutation_samples < 200:
            raise ValueError("statistical resampling requires >=200 samples")
        if not 0 < self.evaluator_support_gate <= 1:
            raise ValueError("evaluator_support_gate must be in (0,1]")
        for grid in (
            self.sla_threshold_grid,
            self.uncertainty_threshold_grid,
            self.planning_weight_grid,
            self.switch_margin_grid,
            self.cql_alpha_grid,
        ):
            if not grid:
                raise ValueError("calibration grids cannot be empty")


def load_science_config(path: str | Path) -> FinalScienceConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    data = payload.get("final_publication", payload)
    for key in (
        "sla_threshold_grid",
        "uncertainty_threshold_grid",
        "planning_weight_grid",
        "switch_margin_grid",
        "cql_alpha_grid",
    ):
        if key in data:
            data[key] = tuple(float(value) for value in data[key])
    cfg = FinalScienceConfig(**data)
    cfg.validate()
    return cfg


def _config_number(value: str) -> int:
    return int(str(value).removeprefix("tr"))


def _cluster_id(frame: pd.DataFrame) -> pd.Series:
    return frame[list(CLUSTER_COLUMNS)].astype(str).agg("/".join, axis=1)


def choose_ope_configs(train: pd.DataFrame, count: int) -> dict[str, Any]:
    """Choose a deterministic disjoint OPE subset that maximizes action coverage on both sides."""
    configs = sorted(train["training_config"].astype(str).unique(), key=_config_number)
    if len(configs) < count + 3:
        raise ValueError("not enough training configurations for disjoint policy/OPE fitting")
    count = min(count, len(configs) - 3)
    support_by_config: dict[str, set[str]] = {}
    rows_by_config: dict[str, int] = {}
    for name, group in train.groupby("training_config"):
        support_by_config[str(name)] = set(group["slice_type"].astype(str) + "|" + group["action"].astype(str))
        rows_by_config[str(name)] = len(group)
    all_support = set().union(*(support_by_config[name] for name in configs))
    total_rows = max(sum(rows_by_config.values()), 1)
    target_fraction = count / len(configs)
    best: tuple[float, tuple[str, ...]] | None = None
    for selected in combinations(configs, count):
        selected_set = set(selected)
        other = [name for name in configs if name not in selected_set]
        ope_support = set().union(*(support_by_config[name] for name in selected))
        policy_support = set().union(*(support_by_config[name] for name in other))
        ope_coverage = len(ope_support & all_support) / max(len(all_support), 1)
        policy_coverage = len(policy_support & all_support) / max(len(all_support), 1)
        row_fraction = sum(rows_by_config[name] for name in selected) / total_rows
        balance = 1.0 - abs(row_fraction - target_fraction)
        score = 4.0 * min(ope_coverage, policy_coverage) + ope_coverage + policy_coverage + 0.05 * balance
        candidate = (score, tuple(selected))
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    ope_configs = list(best[1])
    policy_configs = [name for name in configs if name not in set(ope_configs)]
    ope_support = set().union(*(support_by_config[name] for name in ope_configs))
    policy_support = set().union(*(support_by_config[name] for name in policy_configs))
    return {
        "policy_fit_configs": policy_configs,
        "ope_fit_configs": ope_configs,
        "all_action_cells": len(all_support),
        "policy_action_coverage": len(policy_support & all_support) / max(len(all_support), 1),
        "ope_action_coverage": len(ope_support & all_support) / max(len(all_support), 1),
    }


def _evaluator_model(seed: int, max_iter: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=max_iter,
        learning_rate=0.075,
        max_leaf_nodes=31,
        l2_regularization=0.10,
        random_state=seed,
    )


@dataclass
class IndependentOPE:
    actions: list[str]
    energy_scale: tuple[float, float]
    models: dict[str, Any]
    support_by_slice: dict[str, set[str]]
    fit_summary: dict[str, Any]

    def predict_all(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for objective, model in self.models.items():
            result[objective] = np.column_stack(
                [
                    np.clip(model.predict(_candidate_matrix(frame, self.actions, action)), 0.0, 1.0)
                    for action in self.actions
                ]
            ).astype(np.float32)
        return result

    def supported(self, frame: pd.DataFrame, actions: Sequence[str]) -> np.ndarray:
        slices = frame["slice_type"].astype(str).to_numpy()
        return np.asarray(
            [
                str(action) in self.support_by_slice.get(str(slice_type), set())
                for slice_type, action in zip(slices, actions, strict=True)
            ],
            dtype=bool,
        )


def fit_independent_ope(
    frame: pd.DataFrame,
    actions: Sequence[str],
    study_config: StudyConfig,
    science: FinalScienceConfig,
    seed: int,
) -> IndependentOPE:
    energy_scale = _fit_energy_scale(frame)
    targets = _derive_targets(frame, energy_scale, study_config)
    action_list = list(actions)
    x = _candidate_matrix(targets, action_list, targets["action"].astype(str).to_numpy())
    groups = _cluster_id(targets).to_numpy()
    unique_groups = np.unique(groups)
    folds = min(science.evaluator_folds, len(unique_groups))
    if folds < 2:
        raise ValueError("independent OPE requires at least two experimental clusters")
    y_by_objective = {
        "sla": targets["_sla_cost"].to_numpy(float),
        "energy": targets["_energy_cost"].to_numpy(float),
        "stability": targets["_stability_cost"].to_numpy(float),
    }
    oof = {name: np.full(len(targets), np.nan) for name in y_by_objective}
    splitter = GroupKFold(n_splits=folds)
    for fold_index, (fit_idx, hold_idx) in enumerate(splitter.split(x, groups=groups)):
        for objective_index, (name, values) in enumerate(y_by_objective.items()):
            model = _evaluator_model(
                seed + 100 * fold_index + objective_index,
                max(32, science.evaluator_max_iter // 2),
            )
            model.fit(x[fit_idx], values[fit_idx])
            oof[name][hold_idx] = model.predict(x[hold_idx])
    fit_metrics = {}
    final_models = {}
    for objective_index, (name, values) in enumerate(y_by_objective.items()):
        pred = oof[name]
        if not np.isfinite(pred).all():
            raise ValueError(f"non-finite cross-fitted OPE predictions for {name}")
        fit_metrics[name] = {
            "oof_mae": float(mean_absolute_error(values, pred)),
            "oof_r2": float(r2_score(values, pred)),
        }
        model = _evaluator_model(seed + 900 + objective_index, science.evaluator_max_iter)
        model.fit(x, values)
        final_models[name] = model
    support_by_slice = {
        str(slice_type): set(group["action"].astype(str))
        for slice_type, group in targets.groupby("slice_type")
    }
    summary = {
        "rows": len(targets),
        "clusters": int(len(unique_groups)),
        "folds": folds,
        "model": "HistGradientBoostingRegressor",
        "fit_metrics": fit_metrics,
        "energy_scale": [float(energy_scale[0]), float(energy_scale[1])],
        "support_by_slice": {name: sorted(values) for name, values in support_by_slice.items()},
        "role": "independent outcome/OPE evaluator; not used by policy selection",
    }
    return IndependentOPE(action_list, energy_scale, final_models, support_by_slice, summary)


def _action_indices(actions: Sequence[str], action_list: Sequence[str]) -> np.ndarray:
    index = {action: position for position, action in enumerate(action_list)}
    try:
        return np.asarray([index[str(action)] for action in actions], dtype=int)
    except KeyError as exc:
        raise ValueError(f"action outside policy action space: {exc}") from exc


def _score_from_predictions(
    evaluator: IndependentOPE,
    frame: pd.DataFrame,
    actions: Sequence[str],
    predictions: dict[str, np.ndarray],
    study_config: StudyConfig,
    name: str,
    *,
    rollback: Sequence[bool] | None = None,
    ood_score: Sequence[float] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    selected = np.asarray(actions, dtype=object)
    idx = _action_indices(selected, evaluator.actions)
    rows = np.arange(len(frame))
    sla = predictions["sla"][rows, idx].astype(float)
    energy = predictions["energy"][rows, idx].astype(float)
    stability = predictions["stability"][rows, idx].astype(float)
    denom = study_config.sla_weight + study_config.energy_weight + study_config.stability_weight
    utility = 1.0 - (
        study_config.sla_weight * sla
        + study_config.energy_weight * energy
        + study_config.stability_weight * stability
    ) / max(denom, 1e-9)
    current = _current_actions(frame)
    evaluator_supported = evaluator.supported(frame, selected)
    detail = pd.DataFrame(
        {
            "episode_id": frame["episode_id"].astype(str).to_numpy(),
            "scenario": frame["scenario"].astype(str).to_numpy(),
            "training_config": frame["training_config"].astype(str).to_numpy(),
            "experiment": frame["experiment"].astype(str).to_numpy(),
            "base_station": frame["base_station"].astype(str).to_numpy(),
            "model": name,
            "selected_action": selected,
            "utility": utility,
            "sla_cost": sla,
            "sla_violation": sla > study_config.sla_threshold,
            "energy": energy,
            "stability": stability,
            "policy_churn": selected != current,
            "evaluator_supported": evaluator_supported,
        }
    )
    detail["cluster_id"] = _cluster_id(detail)
    if rollback is not None:
        detail["rollback"] = np.asarray(rollback, dtype=bool)
    if ood_score is not None:
        detail["ood_score"] = np.asarray(ood_score, dtype=float)
    summary = {
        "model": name,
        "estimate_type": "independent_direct_method",
        "selected_ope_mean_utility": float(np.mean(utility)),
        "selected_ope_sla_violation_rate": float(np.mean(sla > study_config.sla_threshold)),
        "selected_ope_mean_energy_proxy": float(np.mean(energy)),
        "selected_ope_mean_stability_cost": float(np.mean(stability)),
        "policy_churn_rate": float(np.mean(selected != current)),
        "evaluator_support_rate": float(np.mean(evaluator_supported)),
    }
    if rollback is not None:
        summary["rollback_rate"] = float(np.mean(np.asarray(rollback, dtype=bool)))
    return summary, detail


def _observed_logged_summary(
    frame: pd.DataFrame,
    evaluator: IndependentOPE,
    study_config: StudyConfig,
) -> dict[str, Any]:
    targets = _derive_targets(frame, evaluator.energy_scale, study_config)
    return {
        "model": "logged_observed",
        "estimate_type": "observed_logged_outcome",
        "selected_ope_mean_utility": float(targets["_utility_reward"].mean()),
        "selected_ope_sla_violation_rate": float(np.mean(targets["_sla_cost"] > study_config.sla_threshold)),
        "selected_ope_mean_energy_proxy": float(targets["_energy_cost"].mean()),
        "selected_ope_mean_stability_cost": float(targets["_stability_cost"].mean()),
        "policy_churn_rate": float(
            np.mean(_current_actions(targets) != targets["action"].astype(str).to_numpy())
        ),
        "evaluator_support_rate": 1.0,
    }


def _policy_precompute(bundle: Any, frame: pd.DataFrame) -> dict[str, Any]:
    targets = _derive_targets(frame, bundle.energy_scale, bundle.config)
    predictions = _all_predictions(bundle, targets, real_data_adaptation=True)
    allowed = _supported_mask(targets, bundle.actions, bundle.actions_by_slice)
    return {
        "targets": targets,
        "predictions": predictions,
        "allowed": allowed,
        "q_norm": _normalize_q(predictions["q"], allowed),
        "ood": _ood_scores(bundle, targets),
        "current": _current_actions(targets),
    }


def _choose_from_precomputed(
    bundle: Any,
    pre: dict[str, Any],
    config: StudyConfig,
    switch_margin: float,
) -> pd.DataFrame:
    predictions = pre["predictions"]
    allowed = pre["allowed"]
    q_norm = pre["q_norm"]
    ood = pre["ood"]
    current = pre["current"]
    targets = pre["targets"]
    score = 1.0 - (
        config.sla_weight * predictions["sla"]
        + config.energy_weight * predictions["energy"]
        + config.stability_weight * predictions["stability"]
        + config.uncertainty_weight * predictions["uncertainty"]
    ) / max(config.immediate_weight_sum, 1e-9)
    score = score + config.planning_weight * q_norm
    score = np.where(allowed, score, -np.inf)
    safe = allowed & (predictions["sla"] <= config.sla_threshold) & (
        predictions["uncertainty"] <= config.uncertainty_threshold
    )
    safe_score = np.where(safe, score, -np.inf)
    has_safe = np.any(safe, axis=1)
    selected_idx = np.argmax(safe_score, axis=1)
    action_index = {action: idx for idx, action in enumerate(bundle.actions)}
    slices = targets["slice_type"].astype(str).to_numpy()
    fallback_idx = np.empty(len(targets), dtype=int)
    current_idx = np.full(len(targets), -1, dtype=int)
    for row, (slice_type, current_action) in enumerate(zip(slices, current, strict=True)):
        if str(current_action) in bundle.actions_by_slice.get(str(slice_type), set()):
            current_idx[row] = action_index[str(current_action)]
            fallback_idx[row] = current_idx[row]
        else:
            fallback_idx[row] = action_index[bundle.fallback_by_slice[str(slice_type)]]
    ood_fallback = ood > config.ood_threshold
    safety_fallback = ~has_safe
    rollback = ood_fallback | safety_fallback
    selected_idx[rollback] = fallback_idx[rollback]
    reason = np.full(len(targets), "", dtype=object)
    reason[ood_fallback] = "ood"
    reason[safety_fallback & ~ood_fallback] = "safety"
    can_hysteresis = (~rollback) & (current_idx >= 0) & (selected_idx != current_idx)
    positions = np.flatnonzero(can_hysteresis)
    if len(positions):
        improvement = score[positions, selected_idx[positions]] - score[positions, current_idx[positions]]
        hold = positions[improvement < switch_margin]
        selected_idx[hold] = current_idx[hold]
        reason[hold] = "hysteresis"
    selected = np.asarray(bundle.actions, dtype=object)[selected_idx]
    return pd.DataFrame(
        {
            "selected_action": selected,
            "rollback": rollback,
            "fallback_reason": reason,
            "ood_score": ood,
            "policy_churn": selected != current,
        },
        index=targets.index,
    )


def calibrate_controller(
    bundle: Any,
    validation: pd.DataFrame,
    evaluator: IndependentOPE,
    science: FinalScienceConfig,
) -> tuple[StudyConfig, float, pd.DataFrame, dict[str, Any]]:
    pre = _policy_precompute(bundle, validation)
    eval_predictions = evaluator.predict_all(validation)
    ood_threshold = float(np.quantile(pre["ood"], 1.0 - science.target_validation_ood_fpr))
    ood_threshold = float(np.clip(ood_threshold, 0.0, 1.0))
    observed = _derive_targets(validation, evaluator.energy_scale, bundle.config)
    observed_sla = float(np.mean(observed["_sla_cost"] > science.reference_sla_threshold))
    rows: list[dict[str, Any]] = []
    for sla_threshold in science.sla_threshold_grid:
        for uncertainty_threshold in science.uncertainty_threshold_grid:
            for planning_weight in science.planning_weight_grid:
                for switch_margin in science.switch_margin_grid:
                    candidate = replace(
                        bundle.config,
                        sla_threshold=float(sla_threshold),
                        uncertainty_threshold=float(uncertainty_threshold),
                        ood_threshold=ood_threshold,
                        planning_weight=float(planning_weight),
                    )
                    decisions = _choose_from_precomputed(bundle, pre, candidate, float(switch_margin))
                    idx = _action_indices(decisions["selected_action"], evaluator.actions)
                    positions = np.arange(len(validation))
                    sla = eval_predictions["sla"][positions, idx].astype(float)
                    energy = eval_predictions["energy"][positions, idx].astype(float)
                    stability = eval_predictions["stability"][positions, idx].astype(float)
                    denom = candidate.sla_weight + candidate.energy_weight + candidate.stability_weight
                    utility = 1.0 - (
                        candidate.sla_weight * sla
                        + candidate.energy_weight * energy
                        + candidate.stability_weight * stability
                    ) / max(denom, 1e-9)
                    churn = float(decisions["policy_churn"].mean())
                    rollback = float(decisions["rollback"].mean())
                    ref_sla_violation = float(np.mean(sla > science.reference_sla_threshold))
                    support = float(evaluator.supported(validation, decisions["selected_action"]).mean())
                    calibration_score = (
                        float(np.mean(utility))
                        - science.calibration_sla_penalty * ref_sla_violation
                        - science.calibration_churn_penalty * churn
                        - science.calibration_rollback_penalty * rollback
                    )
                    feasible = ref_sla_violation <= observed_sla + science.max_validation_sla_regression
                    rows.append(
                        {
                            "sla_threshold": sla_threshold,
                            "uncertainty_threshold": uncertainty_threshold,
                            "ood_threshold": ood_threshold,
                            "planning_weight": planning_weight,
                            "switch_margin": switch_margin,
                            "validation_ope_utility": float(np.mean(utility)),
                            "validation_reference_sla_violation": ref_sla_violation,
                            "validation_churn": churn,
                            "validation_rollback": rollback,
                            "evaluator_support_rate": support,
                            "calibration_score": calibration_score,
                            "feasible": feasible,
                        }
                    )
    table = pd.DataFrame(rows)
    feasible = table[
        table["feasible"] & (table["evaluator_support_rate"] >= science.evaluator_support_gate)
    ]
    pool = feasible if not feasible.empty else table
    selected = pool.sort_values(
        ["calibration_score", "validation_reference_sla_violation", "validation_churn"],
        ascending=[False, True, True],
    ).iloc[0]
    calibrated = replace(
        bundle.config,
        sla_threshold=float(selected["sla_threshold"]),
        uncertainty_threshold=float(selected["uncertainty_threshold"]),
        ood_threshold=float(selected["ood_threshold"]),
        planning_weight=float(selected["planning_weight"]),
    )
    selection = {
        "selected": {
            key: (bool(value) if isinstance(value, (np.bool_, bool)) else float(value))
            for key, value in selected.to_dict().items()
        },
        "observed_validation_reference_sla_violation": observed_sla,
        "ood_calibration_rule": (
            f"seen-validation quantile at target FPR={science.target_validation_ood_fpr}"
        ),
        "test_data_used_for_calibration": False,
    }
    return (
        calibrated,
        float(selected["switch_margin"]),
        table.sort_values("calibration_score", ascending=False),
        selection,
    )


def _normalize_baseline_actions(bundle: Any, frame: pd.DataFrame, actions: Sequence[str]) -> np.ndarray:
    slices = frame["slice_type"].astype(str).to_numpy()
    output = []
    for slice_type, action in zip(slices, actions, strict=True):
        action = str(action)
        if action in bundle.actions_by_slice.get(str(slice_type), set()):
            output.append(action)
        else:
            output.append(bundle.fallback_by_slice[str(slice_type)])
    return np.asarray(output, dtype=object)


def tune_cql_alpha(
    policy_targets: pd.DataFrame,
    validation: pd.DataFrame,
    bundle: Any,
    evaluator: IndependentOPE,
    pub_cfg: PubConfig,
    science: FinalScienceConfig,
    seed: int,
) -> tuple[float, pd.DataFrame]:
    eval_predictions = evaluator.predict_all(validation)
    allowed = _supported_mask(validation, bundle.actions, bundle.actions_by_slice)
    rows: list[dict[str, Any]] = []
    for index, alpha in enumerate(science.cql_alpha_grid):
        tuning_cfg = replace(
            pub_cfg,
            cql_alpha=float(alpha),
            cql_epochs=min(science.cql_tuning_epochs, pub_cfg.cql_epochs),
        )
        model = fit_cql(policy_targets, bundle, tuning_cfg, seed + 300 + index)
        q = np.where(allowed, model.q(validation), -np.inf)
        actions = np.asarray(bundle.actions, dtype=object)[np.argmax(q, axis=1)]
        actions = _normalize_baseline_actions(bundle, validation, actions)
        idx = _action_indices(actions, evaluator.actions)
        positions = np.arange(len(validation))
        sla = eval_predictions["sla"][positions, idx]
        energy = eval_predictions["energy"][positions, idx]
        stability = eval_predictions["stability"][positions, idx]
        denom = bundle.config.sla_weight + bundle.config.energy_weight + bundle.config.stability_weight
        utility = 1.0 - (
            bundle.config.sla_weight * sla
            + bundle.config.energy_weight * energy
            + bundle.config.stability_weight * stability
        ) / max(denom, 1e-9)
        sla_rate = float(np.mean(sla > science.reference_sla_threshold))
        score = float(np.mean(utility)) - science.calibration_sla_penalty * sla_rate
        rows.append(
            {
                "cql_alpha": alpha,
                "validation_ope_utility": float(np.mean(utility)),
                "validation_reference_sla_violation": sla_rate,
                "calibration_score": score,
            }
        )
    table = pd.DataFrame(rows).sort_values("calibration_score", ascending=False)
    return float(table.iloc[0]["cql_alpha"]), table


def _baseline_actions_fqi(model: Any, frame: pd.DataFrame, bundle: Any) -> np.ndarray:
    q = np.column_stack(
        [model.predict(_candidate_matrix(frame, bundle.actions, action)) for action in bundle.actions]
    )
    allowed = _supported_mask(frame, bundle.actions, bundle.actions_by_slice)
    q = np.where(allowed, q, -np.inf)
    return np.asarray(bundle.actions, dtype=object)[np.argmax(q, axis=1)]


def _baseline_actions_cql(model: Any, frame: pd.DataFrame, bundle: Any) -> np.ndarray:
    allowed = _supported_mask(frame, bundle.actions, bundle.actions_by_slice)
    q = np.where(allowed, model.q(frame), -np.inf)
    return np.asarray(bundle.actions, dtype=object)[np.argmax(q, axis=1)]


def _paired_group_statistics(
    proposed: pd.DataFrame,
    baseline: pd.DataFrame,
    science: FinalScienceConfig,
    seed: int,
    group_column: str,
) -> dict[str, Any]:
    left = proposed.groupby(group_column)["utility"].mean()
    right = baseline.groupby(group_column)["utility"].mean()
    common = left.index.intersection(right.index)
    diff = (left.loc[common] - right.loc[common]).to_numpy(float)
    if len(diff) < 2:
        raise ValueError(f"need >=2 paired {group_column} groups")
    rng = np.random.default_rng(seed)
    bootstrap = diff[
        rng.integers(0, len(diff), size=(science.bootstrap_samples, len(diff)))
    ].mean(axis=1)
    observed = float(diff.mean())
    if len(diff) <= 18:
        total = 1 << len(diff)
        extreme = 0
        for mask in range(total):
            signs = np.asarray(
                [1.0 if mask & (1 << bit) else -1.0 for bit in range(len(diff))]
            )
            if abs(float(np.mean(signs * diff))) >= abs(observed) - 1e-15:
                extreme += 1
        p_value = extreme / total
        permutation_method = "exact_random_sign"
        permutations = total
    else:
        extreme = 0
        remaining = science.permutation_samples
        while remaining:
            batch = min(1000, remaining)
            null = (
                rng.choice([-1.0, 1.0], size=(batch, len(diff))) * diff
            ).mean(axis=1)
            extreme += int(np.sum(np.abs(null) >= abs(observed)))
            remaining -= batch
        p_value = (extreme + 1) / (science.permutation_samples + 1)
        permutation_method = "monte_carlo_random_sign"
        permutations = science.permutation_samples
    std = float(np.std(diff, ddof=1))
    delta_key = "mean_cluster_delta_utility" if group_column == "cluster_id" else "mean_group_delta_utility"
    return {
        "unit": group_column,
        "groups": int(len(diff)),
        delta_key: observed,
        "bootstrap_ci95_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(bootstrap, 0.975)),
        "paired_sign_permutation_p": float(p_value),
        "p_display": (
            f"p < {1/(permutations+1):.4g}"
            if permutation_method.startswith("monte") and p_value <= 1 / (permutations + 1)
            else f"p = {p_value:.4g}"
        ),
        "permutation_method": permutation_method,
        "permutations": int(permutations),
        "cohens_dz": observed / std if std > 1e-12 else 0.0,
        "row_weighted_delta_utility": float(
            proposed["utility"].mean() - baseline["utility"].mean()
        ),
    }


def ood_detection_metrics(
    seen: pd.DataFrame,
    unseen: pd.DataFrame,
    threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    y = np.concatenate([np.zeros(len(seen), dtype=int), np.ones(len(unseen), dtype=int)])
    scores = np.concatenate(
        [seen["ood_score"].to_numpy(float), unseen["ood_score"].to_numpy(float)]
    )
    auroc = float(roc_auc_score(y, scores))
    auprc = float(average_precision_score(y, scores))
    fpr, tpr, _ = roc_curve(y, scores)
    positions = np.flatnonzero(tpr >= 0.95)
    fpr95 = float(fpr[positions[0]]) if len(positions) else 1.0
    overall = {
        "auroc": auroc,
        "auprc": auprc,
        "fpr_at_95_tpr": fpr95,
        "calibrated_threshold": float(threshold),
        "seen_detection_rate": float(np.mean(seen["ood_score"] > threshold)),
        "unseen_detection_rate": float(np.mean(unseen["ood_score"] > threshold)),
    }
    per_scenario: dict[str, Any] = {}
    for scenario, group in unseen.groupby("scenario"):
        ys = np.concatenate([np.zeros(len(seen), dtype=int), np.ones(len(group), dtype=int)])
        ss = np.concatenate(
            [seen["ood_score"].to_numpy(float), group["ood_score"].to_numpy(float)]
        )
        per_scenario[str(scenario)] = {
            "auroc_vs_seen": float(roc_auc_score(ys, ss)),
            "auprc_vs_seen": float(average_precision_score(ys, ss)),
            "detection_rate": float(np.mean(group["ood_score"] > threshold)),
        }
    sensitivity = []
    for value in np.linspace(0.30, 0.80, 11):
        sensitivity.append(
            {
                "threshold": float(value),
                "seen_false_positive_rate": float(np.mean(seen["ood_score"] > value)),
                "unseen_true_positive_rate": float(np.mean(unseen["ood_score"] > value)),
            }
        )
    return {"overall": overall, "per_unseen_scenario": per_scenario}, pd.DataFrame(sensitivity)


def transition_audit(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    local = frame.copy()
    current = _current_actions(local)
    next_action = local["action"].astype(str).to_numpy()
    local["joint_action_change"] = current != next_action
    local["scheduler_change"] = (
        local["scheduler_code"].round().to_numpy()
        != local["next_scheduler_code"].round().to_numpy()
    )
    local["prb_change"] = (
        local["slice_prb"].round().to_numpy() != local["next_slice_prb"].round().to_numpy()
    )
    group_columns = ["scenario", "training_config", "experiment", "base_station"]
    rows = []
    for key, group in local.groupby(group_columns, sort=True):
        rows.append(
            {
                **dict(zip(group_columns, key, strict=True)),
                "rows": len(group),
                "episodes": int(group["episode_id"].nunique()),
                "unique_scheduler": int(group["scheduler_code"].round().nunique()),
                "unique_prb": int(group["slice_prb"].round().nunique()),
                "scheduler_change_rows": int(group["scheduler_change"].sum()),
                "prb_change_rows": int(group["prb_change"].sum()),
                "joint_action_change_rows": int(group["joint_action_change"].sum()),
                "joint_action_change_rate": float(group["joint_action_change"].mean()),
            }
        )
    table = pd.DataFrame(rows)
    summary = {
        "rows": len(local),
        "scheduler_change_rows": int(local["scheduler_change"].sum()),
        "prb_change_rows": int(local["prb_change"].sum()),
        "joint_action_change_rows": int(local["joint_action_change"].sum()),
        "joint_action_change_rate": float(local["joint_action_change"].mean()),
        "within_episode_change_evidence_available": bool(local["joint_action_change"].any()),
    }
    return table, summary


def _augment_shortcut(table: pd.DataFrame, audit: dict[str, Any]) -> pd.DataFrame:
    output = table.copy()
    available = bool(audit.get("within_episode_change_evidence_available", False))
    output["change_evidence_available"] = available
    output["interpretation"] = (
        "policy-change performance available"
        if available
        else (
            "no within-episode joint-action changes observed; interpret feature-removal results "
            "as persistence/configuration shortcut sensitivity"
        )
    )
    return output


def _ood_generalization(
    seen: pd.DataFrame,
    unseen: pd.DataFrame,
    threshold: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, detail in (("seen", seen), ("unseen", unseen)):
        result[name] = {
            "mean_ood": float(detail["ood_score"].mean()),
            "p95_ood": float(detail["ood_score"].quantile(0.95)),
            "ood_detection_rate": float(np.mean(detail["ood_score"] > threshold)),
            "ood_fallback_rate": float(
                detail.get("ood_fallback", pd.Series(False, index=detail.index)).mean()
            ),
            "by_scenario": {
                str(scenario): {
                    "rows": len(group),
                    "mean_ood": float(group["ood_score"].mean()),
                    "p95_ood": float(group["ood_score"].quantile(0.95)),
                    "ood_detection_rate": float(np.mean(group["ood_score"] > threshold)),
                    "ood_fallback_rate": float(
                        group.get("ood_fallback", pd.Series(False, index=group.index)).mean()
                    ),
                }
                for scenario, group in detail.groupby("scenario")
            },
        }
    return result


def latency_profile(
    bundle: Any,
    frame: pd.DataFrame,
    calibrated: StudyConfig,
    switch_margin: float,
    science: FinalScienceConfig,
) -> dict[str, Any]:
    if frame.empty:
        return {}
    sample = frame.iloc[
        np.linspace(0, len(frame) - 1, num=min(128, len(frame)), dtype=int)
    ].copy()
    targets = _derive_targets(sample, bundle.energy_scale, bundle.config)
    start = time.perf_counter()
    predictions = _all_predictions(bundle, targets, real_data_adaptation=True)
    critic_ms = (time.perf_counter() - start) * 1000 / len(sample)
    allowed = _supported_mask(targets, bundle.actions, bundle.actions_by_slice)
    start = time.perf_counter()
    ood = _ood_scores(bundle, targets)
    ood_ms = (time.perf_counter() - start) * 1000 / len(sample)
    pre = {
        "targets": targets,
        "predictions": predictions,
        "allowed": allowed,
        "q_norm": _normalize_q(predictions["q"], allowed),
        "ood": ood,
        "current": _current_actions(targets),
    }
    start = time.perf_counter()
    _choose_from_precomputed(bundle, pre, calibrated, switch_margin)
    logic_ms = (time.perf_counter() - start) * 1000 / len(sample)
    count = min(science.latency_samples, len(frame))
    positions = np.linspace(0, len(frame) - 1, num=count, dtype=int)
    timings = []
    for position in positions:
        one = frame.iloc[[int(position)]]
        started = time.perf_counter()
        pre_one = _policy_precompute(bundle, one)
        _choose_from_precomputed(bundle, pre_one, calibrated, switch_margin)
        timings.append((time.perf_counter() - started) * 1000)
    values = np.asarray(timings, dtype=float)
    return {
        "critic_prediction_batch_ms_per_row": float(critic_ms),
        "ood_batch_ms_per_row": float(ood_ms),
        "selection_logic_batch_ms_per_row": float(logic_ms),
        "end_to_end_p50_ms": float(np.percentile(values, 50)),
        "end_to_end_p95_ms": float(np.percentile(values, 95)),
        "end_to_end_p99_ms": float(np.percentile(values, 99)),
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "cpu_count": os.cpu_count() or 0,
        "platform": platform.platform(),
        "note": (
            "component batch probes are not additive; end-to-end values are per-decision "
            "host/container timings"
        ),
    }


def _proposed_decisions(
    bundle: Any,
    frame: pd.DataFrame,
    calibrated: StudyConfig,
    switch_margin: float,
) -> pd.DataFrame:
    return _choose_from_precomputed(
        bundle,
        _policy_precompute(bundle, frame),
        calibrated,
        switch_margin,
    )


def _evaluate_split(
    name: str,
    frame: pd.DataFrame,
    bundle: Any,
    evaluator: IndependentOPE,
    calibrated: StudyConfig,
    switch_margin: float,
    fqi: Any,
    cql: Any,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    evaluator_predictions = evaluator.predict_all(frame)
    proposed_decisions = _proposed_decisions(bundle, frame, calibrated, switch_margin)
    proposed_summary, proposed_detail = _score_from_predictions(
        evaluator,
        frame,
        proposed_decisions["selected_action"],
        evaluator_predictions,
        calibrated,
        "proposed",
        rollback=proposed_decisions["rollback"],
        ood_score=proposed_decisions["ood_score"],
    )
    proposed_detail["ood_fallback"] = (
        proposed_decisions["fallback_reason"].eq("ood").to_numpy(dtype=bool)
    )
    proposed_detail["fallback_reason"] = proposed_decisions["fallback_reason"].astype(str).to_numpy()
    proposed_summary["ood_fallback_rate"] = float(np.mean(proposed_detail["ood_fallback"]))
    fqi_actions = _normalize_baseline_actions(
        bundle, frame, _baseline_actions_fqi(fqi, frame, bundle)
    )
    cql_actions = _normalize_baseline_actions(
        bundle, frame, _baseline_actions_cql(cql, frame, bundle)
    )
    hgb_actions = _normalize_baseline_actions(
        bundle,
        frame,
        bundle.baseline_models["hist_gradient_boosting"].predict(_state_matrix(frame)),
    )
    summaries = [proposed_summary]
    details = {"proposed": proposed_detail}
    for model_name, actions in (
        ("fqi", fqi_actions),
        ("cql_linear", cql_actions),
        ("behavior_hgb", hgb_actions),
    ):
        summary, detail = _score_from_predictions(
            evaluator,
            frame,
            actions,
            evaluator_predictions,
            calibrated,
            model_name,
        )
        summaries.append(summary)
        details[model_name] = detail
    summaries.append(_observed_logged_summary(frame, evaluator, calibrated))
    summary_table = pd.DataFrame(summaries)
    summary_table.insert(0, "split", name)
    combined = pd.concat(details.values(), ignore_index=True)
    combined.insert(0, "split", name)
    return summary_table, details, combined


def run_final_benchmark(
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

    partition = choose_ope_configs(train, science.ope_config_count)
    policy_fit = train[train["training_config"].isin(partition["policy_fit_configs"])].copy()
    ope_fit = train[train["training_config"].isin(partition["ope_fit_configs"])].copy()
    if set(policy_fit["episode_id"].astype(str)) & set(ope_fit["episode_id"].astype(str)):
        raise ValueError("policy-fit and OPE-fit episodes overlap")

    initial_config = StudyConfig()
    bundle = train_bundle(policy_fit.assign(split="train"), initial_config, seed)
    evaluator = fit_independent_ope(
        ope_fit,
        bundle.actions,
        bundle.config,
        science,
        seed + 1000,
    )
    calibrated, switch_margin, calibration_table, calibration_selection = calibrate_controller(
        bundle,
        validation,
        evaluator,
        science,
    )
    policy_targets = _derive_targets(policy_fit, bundle.energy_scale, bundle.config)
    selected_alpha, cql_table = tune_cql_alpha(
        policy_targets,
        validation,
        bundle,
        evaluator,
        pub_cfg,
        science,
        seed,
    )
    final_cql_cfg = replace(pub_cfg, cql_alpha=selected_alpha)
    fqi = fit_fqi(policy_targets, bundle, pub_cfg, seed + 2000)
    cql = fit_cql(policy_targets, bundle, final_cql_cfg, seed + 3000)

    baseline_tables = []
    decision_tables = []
    split_details: dict[str, dict[str, pd.DataFrame]] = {}
    proposed_ood: dict[str, pd.DataFrame] = {}
    for split_name, test in (("seen", seen), ("unseen", unseen)):
        summary_table, details, combined = _evaluate_split(
            split_name,
            test,
            bundle,
            evaluator,
            calibrated,
            switch_margin,
            fqi,
            cql,
        )
        baseline_tables.append(summary_table)
        decision_tables.append(combined)
        split_details[split_name] = details
        proposed_ood[split_name] = details["proposed"]

    baselines = pd.concat(baseline_tables, ignore_index=True)
    decisions = pd.concat(decision_tables, ignore_index=True)

    clustered: dict[str, Any] = {}
    episode_secondary: dict[str, Any] = {}
    stat_seed = seed + 5000
    for split_name in ("seen", "unseen"):
        proposed = split_details[split_name]["proposed"]
        for offset, baseline_name in enumerate(("fqi", "cql_linear", "behavior_hgb")):
            baseline = split_details[split_name][baseline_name]
            key = f"{split_name}:proposed_vs_{baseline_name}"
            clustered[key] = _paired_group_statistics(
                proposed,
                baseline,
                science,
                stat_seed + offset,
                "cluster_id",
            )
            episode_secondary[key] = _paired_group_statistics(
                proposed,
                baseline,
                science,
                stat_seed + 100 + offset,
                "episode_id",
            )

    threshold = calibrated.ood_threshold
    ood_detection, ood_sensitivity = ood_detection_metrics(
        proposed_ood["seen"],
        proposed_ood["unseen"],
        threshold,
    )
    ood_generalization = _ood_generalization(
        proposed_ood["seen"],
        proposed_ood["unseen"],
        threshold,
    )

    audit_table, audit_summary = transition_audit(pd.concat([seen, unseen], ignore_index=True))
    shortcut_tables = []
    for split_name, test in (("seen", seen), ("unseen", unseen)):
        _, split_audit_summary = transition_audit(test)
        diagnostic = shortcut(policy_fit, test, seed)
        diagnostic.insert(0, "split", split_name)
        shortcut_tables.append(_augment_shortcut(diagnostic, split_audit_summary))
    shortcut_table = pd.concat(shortcut_tables, ignore_index=True)
    latency = latency_profile(bundle, seen, calibrated, switch_margin, science)

    proposed_support_seen = float(
        baselines[
            (baselines["split"] == "seen") & (baselines["model"] == "proposed")
        ]["evaluator_support_rate"].iloc[0]
    )
    proposed_support_unseen = float(
        baselines[
            (baselines["split"] == "unseen") & (baselines["model"] == "proposed")
        ]["evaluator_support_rate"].iloc[0]
    )
    partition_disjoint = not bool(
        set(partition["policy_fit_configs"]) & set(partition["ope_fit_configs"])
    )
    stat_clusters_ok = all(int(item["groups"]) >= 8 for item in clustered.values())
    fit_finite = all(
        math.isfinite(float(metric[value]))
        for metric in evaluator.fit_summary["fit_metrics"].values()
        for value in ("oof_mae", "oof_r2")
    )
    readiness_gates = {
        "policy/OPE fitting roles are disjoint": partition_disjoint,
        "validation calibration completed without test data": not calibration_selection[
            "test_data_used_for_calibration"
        ],
        "independent OPE cross-fit metrics are finite": fit_finite,
        f"proposed evaluator support >= {science.evaluator_support_gate:.0%} on seen test": (
            proposed_support_seen >= science.evaluator_support_gate
        ),
        f"proposed evaluator support >= {science.evaluator_support_gate:.0%} on unseen test": (
            proposed_support_unseen >= science.evaluator_support_gate
        ),
        "primary inference uses >=8 experimental clusters per comparison": stat_clusters_ok,
        "OOD discrimination metrics computed on held-out seen vs unseen": math.isfinite(
            float(ood_detection["overall"]["auroc"])
        ),
        "transition/persistence audit completed": not audit_table.empty,
        "counterfactual outcomes use independent evaluator": True,
    }
    warnings: list[str] = []
    if not audit_summary["within_episode_change_evidence_available"]:
        warnings.append(
            "No within-episode joint scheduler+PRB action changes were observed after preparation; "
            "policy-change-only accuracy is unavailable. Treat feature-removal results as "
            "persistence/configuration shortcut sensitivity."
        )
    unseen_fqi = clustered.get("unseen:proposed_vs_fqi", {})
    if abs(float(unseen_fqi.get("cohens_dz", 0.0))) < 0.2:
        warnings.append(
            "The unseen proposed-vs-FQI effect is statistically estimable but small "
            "(|Cohen's dz| < 0.2); do not describe it as a large practical gain."
        )
    if float(latency.get("end_to_end_p95_ms", 0.0)) > 500:
        warnings.append(
            "Host/container p95 decision latency exceeds 500 ms; report hardware/runtime context "
            "and avoid claiming real-time RIC latency without deployment measurements."
        )
    if float(ood_detection["overall"]["auroc"]) < 0.70:
        warnings.append(
            "OOD AUROC is below 0.70; frame OOD results as limited discrimination rather than "
            "strong detection."
        )

    summary: dict[str, Any] = {
        "run_status": "EXPERIMENT-COMPLETE",
        "evidence_status": (
            "READY-FOR-MANUSCRIPT" if all(readiness_gates.values()) else "REVIEW-REQUIRED"
        ),
        "policy_fit_rows": len(policy_fit),
        "ope_fit_rows": len(ope_fit),
        "validation_rows": len(validation),
        "seen_test_rows": len(seen),
        "unseen_test_rows": len(unseen),
        "partition": partition,
        "calibrated_controller": {
            "sla_threshold": calibrated.sla_threshold,
            "uncertainty_threshold": calibrated.uncertainty_threshold,
            "ood_threshold": calibrated.ood_threshold,
            "planning_weight": calibrated.planning_weight,
            "switch_margin": switch_margin,
            "selected_cql_alpha": selected_alpha,
        },
        "readiness_gates": readiness_gates,
        "warnings": warnings,
        "clustered_statistics": clustered,
        "ood_detection": ood_detection,
        "transition_audit": audit_summary,
        "limitations": [
            "alternative selected-action outcomes are independent direct-method/OPE estimates, not causal online intervention effects",
            "energy is a normalized proxy rather than measured joules",
            "the original COMMAG PPO findings are literature-reference values and are not reproduced or used in paired tests",
            "primary inferential units are scenario/training-configuration/experiment clusters; UE-level statistics are secondary only",
        ],
    }
    if literature_reference is not None:
        summary["literature_reference"] = literature_reference

    baselines.to_csv(destination / "publication_baselines.csv", index=False)
    decisions.to_csv(
        destination / "publication_decisions.csv.gz",
        index=False,
        compression="gzip",
    )
    calibration_table.to_csv(destination / "validation_calibration.csv", index=False)
    cql_table.to_csv(destination / "cql_validation.csv", index=False)
    audit_table.to_csv(destination / "transition_audit.csv", index=False)
    shortcut_table.to_csv(destination / "policy_shortcut_test.csv", index=False)
    ood_sensitivity.to_csv(destination / "ood_threshold_sensitivity.csv", index=False)
    (destination / "clustered_statistics.json").write_text(
        json.dumps(clustered, indent=2), encoding="utf-8"
    )
    (destination / "paired_statistics.json").write_text(
        json.dumps(clustered, indent=2), encoding="utf-8"
    )
    (destination / "episode_statistics_secondary.json").write_text(
        json.dumps(episode_secondary, indent=2), encoding="utf-8"
    )
    (destination / "ood_detection.json").write_text(
        json.dumps(ood_detection, indent=2), encoding="utf-8"
    )
    (destination / "ood_generalization.json").write_text(
        json.dumps(ood_generalization, indent=2), encoding="utf-8"
    )
    (destination / "validation_selection.json").write_text(
        json.dumps(calibration_selection, indent=2), encoding="utf-8"
    )
    (destination / "independent_ope_fit.json").write_text(
        json.dumps(evaluator.fit_summary, indent=2), encoding="utf-8"
    )
    (destination / "latency_profile.json").write_text(
        json.dumps(latency, indent=2), encoding="utf-8"
    )
    (destination / "partition.json").write_text(
        json.dumps(partition, indent=2), encoding="utf-8"
    )
    if literature_reference is not None:
        (destination / "literature_reference.json").write_text(
            json.dumps(literature_reference, indent=2), encoding="utf-8"
        )
    joblib.dump(
        {
            "policy_bundle": bundle,
            "independent_ope": evaluator,
            "fqi": fqi,
            "cql": cql,
            "calibrated_controller": summary["calibrated_controller"],
        },
        destination / "publication_models.joblib",
    )
    (destination / "publication_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    report = render_report(
        destination,
        summary=summary,
        baselines=baselines,
        clustered_statistics=clustered,
        ood=ood_generalization,
        ood_detection=ood_detection,
        transition_audit=audit_table,
        shortcut=shortcut_table,
        calibration=calibration_table,
        latency=latency,
        literature_reference=literature_reference or {},
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
    render_report(
        destination,
        summary=summary,
        baselines=baselines,
        clustered_statistics=clustered,
        ood=ood_generalization,
        ood_detection=ood_detection,
        transition_audit=audit_table,
        shortcut=shortcut_table,
        calibration=calibration_table,
        latency=latency,
        literature_reference=literature_reference or {},
    )
    return summary
