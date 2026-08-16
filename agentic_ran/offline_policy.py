"""Paper-oriented offline multi-objective RAN policy study for COMMAG traces.

This module intentionally keeps observational and counterfactual evidence separate.
Observed metrics are only reported for the logged action. Metrics for a newly selected
policy are direct-method estimates from models trained on fixed logs and are never
presented as causal intervention effects.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

SCHEDULERS = {0: "round_robin", 1: "waterfilling", 2: "proportional_fair"}
SLICE_TYPES = ("eMBB", "mMTC", "URLLC")
STATE_COLUMNS = [
    "num_ues",
    "slice_prb",
    "power_multiplier",
    "scheduler_code",
    "dl_mcs",
    "dl_buffer_bytes",
    "dl_bitrate_mbps",
    "dl_errors_pct",
    "dl_cqi",
    "ul_mcs",
    "ul_buffer_bytes",
    "ul_bitrate_mbps",
    "ul_errors_pct",
    "ul_sinr",
    "requested_prbs",
    "granted_prbs",
    "grant_ratio",
]
OBJECTIVES = ("sla", "energy", "stability")
BASELINE_NAMES = ("logistic_regression", "random_forest", "extra_trees", "hist_gradient_boosting")


@dataclass(frozen=True)
class StudyConfig:
    """Configuration frozen into each trained study bundle."""

    sla_weight: float = 0.50
    energy_weight: float = 0.20
    stability_weight: float = 0.15
    uncertainty_weight: float = 0.10
    planning_weight: float = 0.05
    sla_threshold: float = 0.45
    uncertainty_threshold: float = 0.50
    ood_threshold: float = 0.65
    q_gamma: float = 0.97
    q_iterations: int = 4
    estimators: int = 96
    min_samples_leaf: int = 3
    oof_folds: int = 4
    robustness_seeds: int = 30
    latency_samples: int = 128
    parallel_jobs: int = 1

    def validate(self) -> None:
        weights = [
            self.sla_weight,
            self.energy_weight,
            self.stability_weight,
            self.uncertainty_weight,
            self.planning_weight,
        ]
        if any(value < 0 for value in weights):
            raise ValueError("All objective/planning weights must be non-negative")
        if sum(weights[:4]) <= 0:
            raise ValueError("At least one immediate objective weight must be positive")
        for name in ("sla_threshold", "uncertainty_threshold", "ood_threshold"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0 < self.q_gamma <= 1:
            raise ValueError("q_gamma must be in (0, 1]")
        if self.q_iterations < 1 or self.estimators < 8 or self.min_samples_leaf < 1:
            raise ValueError("Model sizing parameters are too small")
        if self.oof_folds < 2:
            raise ValueError("oof_folds must be at least 2")
        if self.robustness_seeds < 1:
            raise ValueError("robustness_seeds must be positive")
        if self.parallel_jobs == 0 or self.parallel_jobs < -1:
            raise ValueError("parallel_jobs must be -1 or a positive integer")

    @property
    def immediate_weight_sum(self) -> float:
        return self.sla_weight + self.energy_weight + self.stability_weight + self.uncertainty_weight


@dataclass
class OfflinePolicyBundle:
    """Serializable model bundle used by the held-out evaluation."""

    config: StudyConfig
    seed: int
    actions: list[str]
    actions_by_slice: dict[str, set[str]]
    fallback_by_slice: dict[str, str]
    critic_models: dict[str, Any]
    uncertainty_critic: Any
    q_model: Any
    ood_scaler: Any
    ood_model: Any
    ood_q05: float
    ood_scale: float
    adaptation: dict[str, dict[str, float]]
    energy_scale: tuple[float, float]
    baseline_models: dict[str, Any]
    training_summary: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _current_actions(frame: pd.DataFrame, prefix: str = "") -> np.ndarray:
    scheduler = frame[f"{prefix}scheduler_code"].round().astype(int).map(SCHEDULERS).fillna("unknown")
    prb = frame[f"{prefix}slice_prb"].round().astype(int).astype(str)
    return (scheduler + ":prb=" + prb).to_numpy(dtype=object)


def _required_columns() -> set[str]:
    return {
        "episode_id",
        "slice_type",
        "action",
        "reward",
        "done",
        "split",
        *STATE_COLUMNS,
        *[f"next_{column}" for column in STATE_COLUMNS],
    }


def validate_dataset(frame: pd.DataFrame) -> dict[str, bool]:
    missing = sorted(_required_columns() - set(frame.columns))
    if missing:
        raise ValueError(f"COMMAG transition table is missing: {', '.join(missing)}")
    numeric = [*STATE_COLUMNS, *[f"next_{item}" for item in STATE_COLUMNS], "reward"]
    finite = bool(np.isfinite(frame[numeric].to_numpy(dtype=float)).all())
    train = frame[frame["split"].eq("train")]
    test = frame[frame["split"].eq("test")]
    train_episodes = set(train["episode_id"].astype(str))
    test_episodes = set(test["episode_id"].astype(str))
    actions = set(train["action"].astype(str))
    unsupported = set(test["action"].astype(str)) - actions
    checks = {
        "at least 100 training transitions": len(train) >= 100,
        "at least 100 held-out transitions": len(test) >= 100,
        "at least three discrete actions": len(actions) >= 3,
        "training and held-out episodes are disjoint": not bool(train_episodes & test_episodes),
        "all held-out actions have training support": not bool(unsupported),
        "all numeric transition values are finite": finite,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError("Invalid offline study dataset: " + "; ".join(failed))
    return checks


def _state_matrix(frame: pd.DataFrame, prefix: str = "") -> np.ndarray:
    numeric = frame[[f"{prefix}{column}" for column in STATE_COLUMNS]].to_numpy(dtype=float)
    slices = np.column_stack([frame["slice_type"].eq(name).to_numpy(dtype=float) for name in SLICE_TYPES])
    return np.column_stack([numeric, slices])


def _candidate_matrix(
    frame: pd.DataFrame,
    actions: Sequence[str],
    candidate_action: str | Sequence[str],
    prefix: str = "",
) -> np.ndarray:
    state = _state_matrix(frame, prefix=prefix)
    if isinstance(candidate_action, str):
        action_values = np.repeat(candidate_action, len(frame))
    else:
        action_values = np.asarray(candidate_action, dtype=object)
    encoded = np.column_stack([action_values == action for action in actions]).astype(float)
    return np.column_stack([state, encoded])


def _energy_raw(frame: pd.DataFrame) -> np.ndarray:
    power = np.maximum(frame["next_power_multiplier"].to_numpy(dtype=float), 0.0)
    granted = np.maximum(frame["next_granted_prbs"].to_numpy(dtype=float), 0.0)
    requested = np.maximum(frame["next_requested_prbs"].to_numpy(dtype=float), 1.0)
    radio_load = np.clip(granted / requested, 0.0, 1.5)
    prb_pressure = np.clip(
        granted / np.maximum(frame["next_slice_prb"].to_numpy(dtype=float), 1.0),
        0.0,
        2.0,
    )
    return power * (0.30 + 0.45 * radio_load + 0.25 * np.minimum(prb_pressure, 1.0))


def _fit_energy_scale(train: pd.DataFrame) -> tuple[float, float]:
    raw = _energy_raw(train)
    low, high = np.quantile(raw, [0.02, 0.98])
    if high - low < 1e-9:
        high = low + 1.0
    return float(low), float(high)


def _derive_targets(frame: pd.DataFrame, energy_scale: tuple[float, float], config: StudyConfig) -> pd.DataFrame:
    output = frame.copy()
    downlink = np.maximum(output["next_dl_bitrate_mbps"].to_numpy(dtype=float), 0.0)
    grant = np.clip(output["next_grant_ratio"].to_numpy(dtype=float), 0.0, 1.0)
    errors = np.clip(output["next_dl_errors_pct"].to_numpy(dtype=float) / 100.0, 0.0, 1.0)
    buffer_values = np.maximum(output["next_dl_buffer_bytes"].to_numpy(dtype=float), 0.0)
    buffer_pressure = buffer_values / (buffer_values + 1000.0)
    slices = output["slice_type"].astype(str).to_numpy()

    sla = np.zeros(len(output), dtype=float)
    embb = slices == "eMBB"
    mmtc = slices == "mMTC"
    urllc = slices == "URLLC"
    sla[embb] = 0.75 * (1.0 - np.clip(downlink[embb] / 1.0, 0.0, 1.0)) + 0.25 * errors[embb]
    sla[mmtc] = (
        0.70 * (1.0 - np.clip(downlink[mmtc] / 0.03, 0.0, 1.0))
        + 0.20 * (1.0 - grant[mmtc])
        + 0.10 * errors[mmtc]
    )
    sla[urllc] = 0.65 * (1.0 - grant[urllc]) + 0.20 * errors[urllc] + 0.15 * buffer_pressure[urllc]
    sla = np.clip(sla, 0.0, 1.0)

    low, high = energy_scale
    energy = np.clip((_energy_raw(output) - low) / max(high - low, 1e-9), 0.0, 1.0)

    current_actions = _current_actions(output)
    logged_actions = output["action"].astype(str).to_numpy(dtype=object)
    churn = (current_actions != logged_actions).astype(float)
    grant_delta = np.clip(np.abs(output["next_grant_ratio"] - output["grant_ratio"]).to_numpy(dtype=float), 0.0, 1.0)
    throughput_delta = np.clip(
        np.abs(output["next_dl_bitrate_mbps"] - output["dl_bitrate_mbps"]).to_numpy(dtype=float)
        / (np.abs(output["dl_bitrate_mbps"].to_numpy(dtype=float)) + 0.10),
        0.0,
        1.0,
    )
    buffer_delta = np.clip(
        np.abs(output["next_dl_buffer_bytes"] - output["dl_buffer_bytes"]).to_numpy(dtype=float)
        / (np.abs(output["dl_buffer_bytes"].to_numpy(dtype=float)) + 100.0),
        0.0,
        1.0,
    )
    stability = np.clip(0.60 * churn + 0.20 * grant_delta + 0.10 * throughput_delta + 0.10 * buffer_delta, 0, 1)

    denom = config.sla_weight + config.energy_weight + config.stability_weight
    immediate_cost = (
        config.sla_weight * sla + config.energy_weight * energy + config.stability_weight * stability
    ) / max(denom, 1e-9)
    utility = 1.0 - immediate_cost

    output["_sla_cost"] = sla
    output["_energy_cost"] = energy
    output["_stability_cost"] = stability
    output["_utility_reward"] = utility
    output["_current_action"] = current_actions
    return output


def _regressor(seed: int, config: StudyConfig, estimators: int | None = None) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=estimators or config.estimators,
        min_samples_leaf=config.min_samples_leaf,
        max_features=0.85,
        random_state=seed,
        n_jobs=config.parallel_jobs,
    )


def _baseline_models(seed: int, config: StudyConfig) -> dict[str, Any]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1800,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=config.estimators,
            min_samples_leaf=config.min_samples_leaf,
            class_weight="balanced_subsample",
            random_state=seed + 11,
            n_jobs=config.parallel_jobs,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=config.estimators,
            min_samples_leaf=config.min_samples_leaf,
            class_weight="balanced",
            random_state=seed + 23,
            n_jobs=config.parallel_jobs,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=max(32, config.estimators),
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=0.08,
            random_state=seed + 37,
        ),
    }


def _candidate_predictions(
    model: Any,
    frame: pd.DataFrame,
    actions: Sequence[str],
    *,
    prefix: str = "",
) -> np.ndarray:
    return np.column_stack(
        [model.predict(_candidate_matrix(frame, actions, action, prefix=prefix)) for action in actions]
    )


def _supported_mask(
    frame: pd.DataFrame,
    actions: Sequence[str],
    actions_by_slice: dict[str, set[str]],
) -> np.ndarray:
    return np.column_stack(
        [
            frame["slice_type"].astype(str).map(
                lambda value, action=action: action in actions_by_slice.get(value, set())
            )
            for action in actions
        ]
    ).astype(bool)


def _oof_predictions(
    train: pd.DataFrame,
    actions: Sequence[str],
    config: StudyConfig,
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    x = _candidate_matrix(train, actions, train["action"].astype(str).to_numpy())
    groups = train["episode_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    folds = min(config.oof_folds, len(unique_groups))
    targets = {
        "sla": train["_sla_cost"].to_numpy(dtype=float),
        "energy": train["_energy_cost"].to_numpy(dtype=float),
        "stability": train["_stability_cost"].to_numpy(dtype=float),
    }
    predictions = {name: np.full(len(train), np.nan, dtype=float) for name in OBJECTIVES}
    if folds >= 2:
        splitter = GroupKFold(n_splits=folds)
        for fold_index, (fit_idx, val_idx) in enumerate(splitter.split(x, groups=groups)):
            for target_index, name in enumerate(OBJECTIVES):
                model = _regressor(
                    seed + 1000 + fold_index * 17 + target_index,
                    config,
                    estimators=max(12, config.estimators // 3),
                )
                model.fit(x[fit_idx], targets[name][fit_idx])
                predictions[name][val_idx] = model.predict(x[val_idx])
    else:  # pragma: no cover - protected by benchmark gates in real use
        for target_index, name in enumerate(OBJECTIVES):
            model = _regressor(seed + 1000 + target_index, config, estimators=max(12, config.estimators // 3))
            model.fit(x, targets[name])
            predictions[name] = model.predict(x)

    residual_parts: list[np.ndarray] = []
    for name in OBJECTIVES:
        missing = ~np.isfinite(predictions[name])
        if missing.any():
            predictions[name][missing] = np.nanmean(predictions[name])
        scale = max(float(np.std(targets[name])), 0.05)
        residual_parts.append(np.abs(targets[name] - predictions[name]) / scale)
    uncertainty = np.mean(np.column_stack(residual_parts), axis=1)
    cap = max(float(np.quantile(uncertainty, 0.98)), 1e-6)
    uncertainty = np.clip(uncertainty / cap, 0.0, 1.0)
    return predictions, uncertainty


def _fit_adaptation(train: pd.DataFrame, oof: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    adaptation: dict[str, dict[str, float]] = {}
    keys = train["slice_type"].astype(str) + "|" + train["action"].astype(str)
    for name, column in (("sla", "_sla_cost"), ("energy", "_energy_cost"), ("stability", "_stability_cost")):
        residual = train[column].to_numpy(dtype=float) - oof[name]
        grouped = pd.DataFrame({"key": keys, "residual": residual}).groupby("key")["residual"].mean()
        adaptation[name] = {str(key): float(value) for key, value in grouped.items()}
    return adaptation


def _fit_q_model(
    train: pd.DataFrame,
    actions: Sequence[str],
    actions_by_slice: dict[str, set[str]],
    config: StudyConfig,
    seed: int,
) -> Any:
    x = _candidate_matrix(train, actions, train["action"].astype(str).to_numpy())
    reward = train["_utility_reward"].to_numpy(dtype=float)
    targets = reward.copy()
    model: Any | None = None
    allowed = _supported_mask(train, actions, actions_by_slice)
    not_done = (~train["done"].astype(bool)).to_numpy(dtype=float)
    for iteration in range(config.q_iterations):
        model = _regressor(seed + 200 + iteration, config)
        model.fit(x, targets)
        next_q = _candidate_predictions(model, train, actions, prefix="next_")
        next_q = np.where(allowed, next_q, -np.inf)
        continuation = np.max(next_q, axis=1)
        continuation[~np.isfinite(continuation)] = 0.0
        targets = reward + config.q_gamma * continuation * not_done
    return model


def train_bundle(train: pd.DataFrame, config: StudyConfig, seed: int = 42) -> OfflinePolicyBundle:
    config.validate()
    if train.empty:
        raise ValueError("Training split is empty")
    actions = sorted(train["action"].astype(str).unique())
    if len(actions) < 3:
        raise ValueError("Offline policy study requires at least three actions")
    energy_scale = _fit_energy_scale(train)
    train_targets = _derive_targets(train, energy_scale, config)
    actions_by_slice = {
        str(slice_type): set(group["action"].astype(str))
        for slice_type, group in train_targets.groupby("slice_type", sort=True)
    }
    x = _candidate_matrix(train_targets, actions, train_targets["action"].astype(str).to_numpy())

    oof, uncertainty_target = _oof_predictions(train_targets, actions, config, seed)
    adaptation = _fit_adaptation(train_targets, oof)

    critic_models: dict[str, Any] = {}
    critic_fit: dict[str, dict[str, float | None]] = {}
    for index, (name, column) in enumerate(
        (("sla", "_sla_cost"), ("energy", "_energy_cost"), ("stability", "_stability_cost"))
    ):
        target = train_targets[column].to_numpy(dtype=float)
        model = _regressor(seed + index * 31, config)
        model.fit(x, target)
        prediction = model.predict(x)
        critic_models[name] = model
        critic_fit[name] = {
            "train_mae": float(mean_absolute_error(target, prediction)),
            "train_r2": _finite(r2_score(target, prediction)),
            "oof_mae": float(mean_absolute_error(target, oof[name])),
            "oof_r2": _finite(r2_score(target, oof[name])),
        }

    uncertainty_critic = _regressor(seed + 131, config)
    uncertainty_critic.fit(x, uncertainty_target)
    uncertainty_fit = uncertainty_critic.predict(x)

    q_model = _fit_q_model(train_targets, actions, actions_by_slice, config, seed)

    state = _state_matrix(train_targets)
    ood_scaler = RobustScaler(quantile_range=(10.0, 90.0)).fit(state)
    scaled_state = ood_scaler.transform(state)
    ood_model = IsolationForest(
        n_estimators=max(32, config.estimators),
        max_samples="auto",
        contamination="auto",
        random_state=seed + 173,
        n_jobs=config.parallel_jobs,
    ).fit(scaled_state)
    train_decision = ood_model.decision_function(scaled_state)
    q05 = float(np.quantile(train_decision, 0.05))
    median = float(np.median(train_decision))
    ood_scale = max(median - q05, 1e-6)

    summary_rows = train_targets.groupby(["slice_type", "action"], sort=True).agg(
        support=("action", "size"),
        sla=("_sla_cost", "mean"),
        energy=("_energy_cost", "mean"),
        stability=("_stability_cost", "mean"),
    )
    fallback_by_slice: dict[str, str] = {}
    for slice_type, group in summary_rows.groupby(level=0):
        local = group.reset_index()
        local["risk"] = 0.70 * local["sla"] + 0.20 * local["energy"] + 0.10 * local["stability"]
        local = local.sort_values(["risk", "support"], ascending=[True, False])
        fallback_by_slice[str(slice_type)] = str(local.iloc[0]["action"])

    baseline_models = _baseline_models(seed, config)
    baseline_x = _state_matrix(train_targets)
    labels = train_targets["action"].astype(str).to_numpy()
    for model in baseline_models.values():
        model.fit(baseline_x, labels)

    training_summary = {
        "seed": seed,
        "rows": len(train_targets),
        "episodes": int(train_targets["episode_id"].nunique()),
        "actions": actions,
        "action_support": train_targets["action"].value_counts().sort_index().to_dict(),
        "slice_support": train_targets["slice_type"].value_counts().sort_index().to_dict(),
        "critic_fit": critic_fit,
        "uncertainty_critic_train_mae": float(mean_absolute_error(uncertainty_target, uncertainty_fit)),
        "uncertainty_target_mean": float(np.mean(uncertainty_target)),
        "ood_train_decision_q05": q05,
        "ood_train_decision_median": median,
        "fallback_by_slice": fallback_by_slice,
        "adaptation_cells": {name: len(values) for name, values in adaptation.items()},
        "energy_proxy_scale": [float(energy_scale[0]), float(energy_scale[1])],
    }
    return OfflinePolicyBundle(
        config=config,
        seed=seed,
        actions=actions,
        actions_by_slice=actions_by_slice,
        fallback_by_slice=fallback_by_slice,
        critic_models=critic_models,
        uncertainty_critic=uncertainty_critic,
        q_model=q_model,
        ood_scaler=ood_scaler,
        ood_model=ood_model,
        ood_q05=q05,
        ood_scale=ood_scale,
        adaptation=adaptation,
        energy_scale=energy_scale,
        baseline_models=baseline_models,
        training_summary=training_summary,
    )


def _ood_scores(bundle: OfflinePolicyBundle, frame: pd.DataFrame) -> np.ndarray:
    state = bundle.ood_scaler.transform(_state_matrix(frame))
    decision = bundle.ood_model.decision_function(state)
    distance = (bundle.ood_q05 - decision) / bundle.ood_scale
    return _sigmoid(distance)


def _apply_adaptation(
    predictions: np.ndarray,
    frame: pd.DataFrame,
    actions: Sequence[str],
    corrections: dict[str, float],
) -> np.ndarray:
    output = predictions.copy()
    slices = frame["slice_type"].astype(str).to_numpy()
    for action_index, action in enumerate(actions):
        output[:, action_index] += np.asarray(
            [corrections.get(f"{slice_type}|{action}", 0.0) for slice_type in slices],
            dtype=float,
        )
    return np.clip(output, 0.0, 1.0)


def _all_predictions(
    bundle: OfflinePolicyBundle,
    frame: pd.DataFrame,
    *,
    real_data_adaptation: bool,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for name in OBJECTIVES:
        values = _candidate_predictions(bundle.critic_models[name], frame, bundle.actions)
        if real_data_adaptation:
            values = _apply_adaptation(values, frame, bundle.actions, bundle.adaptation[name])
        output[name] = np.clip(values, 0.0, 1.0)
    output["uncertainty"] = np.clip(
        _candidate_predictions(bundle.uncertainty_critic, frame, bundle.actions),
        0.0,
        1.0,
    )
    output["q"] = _candidate_predictions(bundle.q_model, frame, bundle.actions)
    return output


def _normalize_q(q_values: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    masked = np.where(allowed, q_values, np.nan)
    row_min = np.nanmin(masked, axis=1)
    row_max = np.nanmax(masked, axis=1)
    span = row_max - row_min
    normalized = np.full_like(q_values, 0.5, dtype=float)
    good = span > 1e-9
    normalized[good] = (q_values[good] - row_min[good, None]) / span[good, None]
    return np.clip(normalized, 0.0, 1.0)


def select_policies(
    bundle: OfflinePolicyBundle,
    frame: pd.DataFrame,
    *,
    safety_gate: bool = True,
    ood_gate: bool = True,
    planning: bool = True,
    real_data_adaptation: bool = True,
) -> pd.DataFrame:
    predictions = _all_predictions(bundle, frame, real_data_adaptation=real_data_adaptation)
    allowed = _supported_mask(frame, bundle.actions, bundle.actions_by_slice)
    q_norm = _normalize_q(predictions["q"], allowed)
    config = bundle.config
    score = 1.0 - (
        config.sla_weight * predictions["sla"]
        + config.energy_weight * predictions["energy"]
        + config.stability_weight * predictions["stability"]
        + config.uncertainty_weight * predictions["uncertainty"]
    ) / config.immediate_weight_sum
    if planning:
        score = score + config.planning_weight * q_norm
    score = np.where(allowed, score, -np.inf)

    safe = allowed.copy()
    if safety_gate:
        safe &= predictions["sla"] <= config.sla_threshold
        safe &= predictions["uncertainty"] <= config.uncertainty_threshold

    ood = _ood_scores(bundle, frame)
    current = _current_actions(frame)
    slices = frame["slice_type"].astype(str).to_numpy()
    action_index = {action: index for index, action in enumerate(bundle.actions)}

    selected: list[str] = []
    fallback_reason: list[str] = []
    rollback = np.zeros(len(frame), dtype=bool)
    safety_override = np.zeros(len(frame), dtype=bool)
    unconstrained: list[str] = []

    for row in range(len(frame)):
        allowed_indices = np.flatnonzero(allowed[row])
        if not len(allowed_indices):  # pragma: no cover - train/test support gate should prevent this
            raise ValueError(f"No supported action for slice {slices[row]}")
        best_unconstrained_index = allowed_indices[int(np.argmax(score[row, allowed_indices]))]
        best_unconstrained = bundle.actions[best_unconstrained_index]
        unconstrained.append(best_unconstrained)

        current_action = str(current[row])
        fallback = (
            current_action
            if current_action in bundle.actions_by_slice.get(slices[row], set())
            else bundle.fallback_by_slice[slices[row]]
        )
        if ood_gate and ood[row] > config.ood_threshold:
            selected.append(fallback)
            fallback_reason.append("ood")
            rollback[row] = True
            continue

        candidate_indices = np.flatnonzero(safe[row])
        if not len(candidate_indices):
            selected.append(fallback)
            fallback_reason.append("safety")
            rollback[row] = True
            safety_override[row] = fallback != best_unconstrained
            continue
        selected_index = candidate_indices[int(np.argmax(score[row, candidate_indices]))]
        chosen = bundle.actions[selected_index]
        selected.append(chosen)
        fallback_reason.append("")
        safety_override[row] = chosen != best_unconstrained

    selected_array = np.asarray(selected, dtype=object)
    selected_indices = np.asarray([action_index[action] for action in selected_array], dtype=int)
    rows = np.arange(len(frame))
    result = pd.DataFrame(
        {
            "selected_action": selected_array,
            "unconstrained_action": np.asarray(unconstrained, dtype=object),
            "fallback_reason": fallback_reason,
            "rollback": rollback,
            "safety_override": safety_override,
            "ood_score": ood,
            "selected_sla_cost": predictions["sla"][rows, selected_indices],
            "selected_energy_cost": predictions["energy"][rows, selected_indices],
            "selected_stability_cost": predictions["stability"][rows, selected_indices],
            "selected_uncertainty": predictions["uncertainty"][rows, selected_indices],
            "selected_q": predictions["q"][rows, selected_indices],
        },
        index=frame.index,
    )
    result["policy_churn"] = result["selected_action"].to_numpy(dtype=object) != current
    result["logged_action_agreement"] = (
        result["selected_action"].to_numpy(dtype=object) == frame["action"].astype(str).to_numpy(dtype=object)
    )
    denom = config.sla_weight + config.energy_weight + config.stability_weight
    result["selected_direct_utility"] = 1.0 - (
        config.sla_weight * result["selected_sla_cost"]
        + config.energy_weight * result["selected_energy_cost"]
        + config.stability_weight * result["selected_stability_cost"]
    ) / max(denom, 1e-9)
    return result


def _latency_profile(bundle: OfflinePolicyBundle, frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0, "throughput_decisions_s": 0.0}
    start = time.perf_counter()
    select_policies(bundle, frame)
    batch_elapsed = max(time.perf_counter() - start, 1e-9)
    count = min(bundle.config.latency_samples, len(frame))
    positions = np.linspace(0, len(frame) - 1, num=count, dtype=int)
    timings: list[float] = []
    for position in positions:
        sample = frame.iloc[[int(position)]]
        started = time.perf_counter()
        select_policies(bundle, sample)
        timings.append((time.perf_counter() - started) * 1000.0)
    values = np.asarray(timings, dtype=float)
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(np.max(values)),
        "throughput_decisions_s": float(len(frame) / batch_elapsed),
    }


def evaluate_bundle(
    bundle: OfflinePolicyBundle,
    frame: pd.DataFrame,
    *,
    variant: str = "full",
    measure_latency: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    variant_options = {
        "full": {},
        "without_safety_gate": {"safety_gate": False},
        "without_ood_gate": {"ood_gate": False},
        "without_planning": {"planning": False},
        "without_real_data_adaptation": {"real_data_adaptation": False},
    }
    if variant not in variant_options:
        raise ValueError(f"Unknown ablation variant: {variant}")
    targets = _derive_targets(frame, bundle.energy_scale, bundle.config)
    started = time.perf_counter()
    decisions = select_policies(bundle, targets, **variant_options[variant])
    elapsed = max(time.perf_counter() - started, 1e-9)
    logged_utility = targets["_utility_reward"].to_numpy(dtype=float)
    selected_utility = decisions["selected_direct_utility"].to_numpy(dtype=float)
    metrics: dict[str, Any] = {
        "variant": variant,
        "rows": len(targets),
        "episodes": int(targets["episode_id"].nunique()),
        "logged_observed_sla_violation_rate": float(np.mean(targets["_sla_cost"] > bundle.config.sla_threshold)),
        "logged_observed_mean_energy_proxy": float(targets["_energy_cost"].mean()),
        "logged_observed_mean_stability_cost": float(targets["_stability_cost"].mean()),
        "logged_observed_mean_utility": float(np.mean(logged_utility)),
        "selected_dm_sla_violation_rate": float(
            np.mean(decisions["selected_sla_cost"] > bundle.config.sla_threshold)
        ),
        "selected_dm_mean_energy_proxy": float(decisions["selected_energy_cost"].mean()),
        "selected_dm_mean_stability_cost": float(decisions["selected_stability_cost"].mean()),
        "selected_dm_mean_utility": float(np.mean(selected_utility)),
        "selected_dm_estimated_utility_uplift": float(np.mean(selected_utility - logged_utility)),
        "policy_churn_rate": float(decisions["policy_churn"].mean()),
        "rollback_rate": float(decisions["rollback"].mean()),
        "ood_fallback_rate": float(decisions["fallback_reason"].eq("ood").mean()),
        "safety_fallback_rate": float(decisions["fallback_reason"].eq("safety").mean()),
        "safety_override_rate": float(decisions["safety_override"].mean()),
        "logged_action_agreement": float(decisions["logged_action_agreement"].mean()),
        "mean_ood_score": float(decisions["ood_score"].mean()),
        "p95_ood_score": float(decisions["ood_score"].quantile(0.95)),
        "mean_uncertainty": float(decisions["selected_uncertainty"].mean()),
        "batch_throughput_decisions_s": float(len(targets) / elapsed),
        "action_distribution": decisions["selected_action"].value_counts().sort_index().to_dict(),
    }
    if measure_latency:
        metrics["latency"] = _latency_profile(bundle, targets)
    return metrics, decisions


def _baseline_comparison(
    bundle: OfflinePolicyBundle,
    frame: pd.DataFrame,
    *,
    full_metrics: dict[str, Any] | None = None,
) -> pd.DataFrame:
    targets = _derive_targets(frame, bundle.energy_scale, bundle.config)
    predictions = _all_predictions(bundle, targets, real_data_adaptation=True)
    action_index = {action: idx for idx, action in enumerate(bundle.actions)}
    rows = np.arange(len(targets))
    result_rows: list[dict[str, Any]] = []
    x = _state_matrix(targets)
    logged = targets["action"].astype(str).to_numpy(dtype=object)
    current = _current_actions(targets)
    slices = targets["slice_type"].astype(str).to_numpy()

    for name, model in bundle.baseline_models.items():
        started = time.perf_counter()
        predicted = np.asarray(model.predict(x), dtype=object)
        elapsed = max(time.perf_counter() - started, 1e-9)
        corrected = predicted.copy()
        fallback_count = 0
        for index, action in enumerate(predicted):
            if action not in bundle.actions_by_slice.get(slices[index], set()):
                fallback_count += 1
                corrected[index] = bundle.fallback_by_slice[slices[index]]
        indices = np.asarray([action_index[action] for action in corrected], dtype=int)
        sla = predictions["sla"][rows, indices]
        energy = predictions["energy"][rows, indices]
        stability = predictions["stability"][rows, indices]
        denom = bundle.config.sla_weight + bundle.config.energy_weight + bundle.config.stability_weight
        utility = 1.0 - (
            bundle.config.sla_weight * sla
            + bundle.config.energy_weight * energy
            + bundle.config.stability_weight * stability
        ) / max(denom, 1e-9)
        result_rows.append(
            {
                "model": name,
                "logged_action_macro_f1": float(f1_score(logged, corrected, average="macro", zero_division=0)),
                "logged_action_agreement": float(np.mean(logged == corrected)),
                "selected_dm_sla_violation_rate": float(np.mean(sla > bundle.config.sla_threshold)),
                "selected_dm_mean_energy_proxy": float(np.mean(energy)),
                "selected_dm_mean_stability_cost": float(np.mean(stability)),
                "selected_dm_mean_utility": float(np.mean(utility)),
                "policy_churn_rate": float(np.mean(corrected != current)),
                "unsupported_action_fallback_rate": float(fallback_count / max(len(corrected), 1)),
                "prediction_throughput_decisions_s": float(len(corrected) / elapsed),
            }
        )
    if full_metrics is None:
        full_metrics, _ = evaluate_bundle(bundle, targets, measure_latency=False)
    result_rows.append(
        {
            "model": "multi_objective_offline_policy",
            "logged_action_macro_f1": np.nan,
            "logged_action_agreement": full_metrics["logged_action_agreement"],
            "selected_dm_sla_violation_rate": full_metrics["selected_dm_sla_violation_rate"],
            "selected_dm_mean_energy_proxy": full_metrics["selected_dm_mean_energy_proxy"],
            "selected_dm_mean_stability_cost": full_metrics["selected_dm_mean_stability_cost"],
            "selected_dm_mean_utility": full_metrics["selected_dm_mean_utility"],
            "policy_churn_rate": full_metrics["policy_churn_rate"],
            "unsupported_action_fallback_rate": 0.0,
            "prediction_throughput_decisions_s": full_metrics["batch_throughput_decisions_s"],
        }
    )
    return pd.DataFrame(result_rows)


def _episode_metrics(bundle: OfflinePolicyBundle, frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for episode_id, episode in frame.groupby("episode_id", sort=True):
        metrics, _ = evaluate_bundle(bundle, episode, measure_latency=False)
        rows.append({"episode_id": str(episode_id), **{k: v for k, v in metrics.items() if not isinstance(v, dict)}})
    return pd.DataFrame(rows)


def _summary_stats(frame: pd.DataFrame, group_columns: Sequence[str], metric_columns: Sequence[str]) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    grouped: Iterable[tuple[Any, pd.DataFrame]]
    if group_columns:
        grouped = frame.groupby(list(group_columns), sort=True, dropna=False)
    else:
        grouped = [((), frame)]
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        base = dict(zip(group_columns, group_key, strict=True))
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            output.append(
                {
                    **base,
                    "metric": metric,
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "std": std,
                    "ci95_half_width": float(1.96 * std / math.sqrt(len(values))) if len(values) > 1 else 0.0,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
    return pd.DataFrame(output)


def run_seed_study(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: StudyConfig,
    *,
    base_seed: int,
    seeds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    ablation_variants = [
        "without_safety_gate",
        "without_ood_gate",
        "without_planning",
        "without_real_data_adaptation",
    ]
    for offset in range(seeds):
        seed = base_seed + offset
        bundle = train_bundle(train, config, seed=seed)
        full_metrics, _ = evaluate_bundle(bundle, test, variant="full", measure_latency=False)
        flat_full = {k: v for k, v in full_metrics.items() if not isinstance(v, dict)}
        seed_rows.append({"seed": seed, **flat_full})
        ablation_rows.append({"seed": seed, **flat_full})
        for variant in ablation_variants:
            metrics, _ = evaluate_bundle(bundle, test, variant=variant, measure_latency=False)
            ablation_rows.append(
                {"seed": seed, **{k: v for k, v in metrics.items() if not isinstance(v, dict)}}
            )
        comparison = _baseline_comparison(bundle, test, full_metrics=full_metrics)
        comparison.insert(0, "seed", seed)
        baseline_rows.extend(comparison.to_dict(orient="records"))
    return pd.DataFrame(seed_rows), pd.DataFrame(ablation_rows), pd.DataFrame(baseline_rows)


def _table_html(frame: pd.DataFrame, *, digits: int = 4, limit: int | None = None) -> str:
    if frame.empty:
        return "<p>No rows.</p>"
    local = frame.head(limit) if limit else frame
    headers = "".join(f"<th>{html.escape(str(column))}</th>" for column in local.columns)
    body_rows = []
    for _, row in local.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                text = "—" if not math.isfinite(value) else f"{value:.{digits}f}"
            else:
                text = str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<div class='table-wrap'><table><thead><tr>"
        f"{headers}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def _page(title: str, eyebrow: str, verdict: str, body: str) -> str:
    verdict_class = "ok" if verdict in {"BENCHMARK-READY", "TRAINED", "VALID"} else "warn"
    style = """
<style>
:root { color-scheme: light; }
body { margin:0; background:#f4f6f8; color:#17202a;
  font:15px/1.55 Inter,system-ui,-apple-system,sans-serif; }
main { max-width:1180px; margin:auto; padding:38px 20px 64px; }
h1 { font-size:38px; line-height:1.1; margin:6px 0 12px; } h2 { margin-top:0; }
.eyebrow { text-transform:uppercase; letter-spacing:.12em; font-weight:800; color:#607080; }
.verdict { display:inline-block; padding:8px 12px; border-radius:10px; font-weight:900; }
.ok { background:#dcfce7; color:#166534; } .warn { background:#ffedd5; color:#9a3412; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin:20px 0; }
.card, section { background:white; border:1px solid #dce3e9; border-radius:14px;
  padding:18px; margin:14px 0; box-shadow:0 7px 28px #2030400b; }
.value { font-size:26px; font-weight:850; margin-top:4px; } .label { color:#667788; font-size:13px; }
.table-wrap { overflow-x:auto; } table { width:100%; border-collapse:collapse; font-size:13px; }
th, td { padding:8px 9px; border-bottom:1px solid #e7edf2; text-align:left; white-space:nowrap; }
th { color:#536879; background:#f8fafc; } code, pre { background:#eef2f5; border-radius:7px; }
code { padding:2px 5px; } pre { padding:13px; overflow:auto; }
.note { border-left:4px solid #f59e0b; background:#fff7ed; padding:12px 14px; border-radius:8px; }
.good { border-left:4px solid #10b981; background:#ecfdf5; padding:12px 14px; border-radius:8px; }
</style>
"""
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title>{style}</head><body><main>"
        f"<div class='eyebrow'>{html.escape(eyebrow)}</div><h1>{html.escape(title)}</h1>"
        f"<span class='verdict {verdict_class}'>{html.escape(verdict)}</span>{body}</main></body></html>"
    )


def prepare_report(
    data_path: Path | str,
    manifest_path: Path | str | None,
    output_dir: Path | str,
) -> dict[str, Any]:
    data_path = Path(data_path)
    frame = pd.read_csv(data_path)
    checks = validate_dataset(frame)
    train = frame[frame["split"].eq("train")]
    test = frame[frame["split"].eq("test")]
    manifest = {}
    if manifest_path and Path(manifest_path).exists():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    summary = {
        "verdict": "BENCHMARK-READY" if all(checks.values()) else "INVALID",
        "dataset": str(data_path),
        "dataset_sha256": _sha256(data_path),
        "rows": len(frame),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_episodes": int(train["episode_id"].nunique()),
        "test_episodes": int(test["episode_id"].nunique()),
        "actions": sorted(train["action"].astype(str).unique()),
        "checks": checks,
        "source_repository": manifest.get("source_repository"),
        "source_revision": manifest.get("source_revision"),
        "source_license": manifest.get("source_license"),
        "prepared_sha256_from_manifest": manifest.get("prepared_sha256"),
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    action_table = (
        train.groupby(["slice_type", "action"], sort=True).size().rename("train_rows").reset_index()
    )
    action_table.to_csv(destination / "action_support.csv", index=False)
    cards = "".join(
        [
            f"<div class='card'><div class='label'>Train rows</div><div class='value'>{len(train)}</div></div>",
            f"<div class='card'><div class='label'>Held-out rows</div><div class='value'>{len(test)}</div></div>",
            "<div class='card'><div class='label'>Train episodes</div>"
            f"<div class='value'>{train['episode_id'].nunique()}</div></div>",
            "<div class='card'><div class='label'>Held-out episodes</div>"
            f"<div class='value'>{test['episode_id'].nunique()}</div></div>",
            "<div class='card'><div class='label'>Actions</div>"
            f"<div class='value'>{train['action'].nunique()}</div></div>",
        ]
    )
    gates = "".join(
        f"<li>{'PASS' if passed else 'FAIL'} — {html.escape(name)}</li>"
        for name, passed in checks.items()
    )
    provenance_fields = ("source_repository", "source_revision", "source_license")
    provenance = html.escape(json.dumps({key: summary[key] for key in provenance_fields}, indent=2))
    body = (
        f"<div class='grid'>{cards}</div>"
        f"<section><h2>Validity gates</h2><ul>{gates}</ul></section>"
        f"<section><h2>Action support</h2>{_table_html(action_table)}</section>"
        f"<section><h2>Provenance</h2><pre>{provenance}</pre>"
        "<p class='note'>Raw COMMAG files are not redistributed. Keep the upstream GPL-3.0 "
        "provenance with any derived dataset and paper artifact.</p></section>"
    )
    (destination / "report.html").write_text(
        _page(
            "COMMAG data preparation report",
            "Pinned real-data profile · offline study",
            summary["verdict"],
            body,
        ),
        encoding="utf-8",
    )
    return summary


def train_study(
    data_path: Path | str,
    artifacts_dir: Path | str,
    output_dir: Path | str,
    *,
    config: StudyConfig,
    seed: int,
) -> dict[str, Any]:
    frame = pd.read_csv(data_path)
    checks = validate_dataset(frame)
    train = frame[frame["split"].eq("train")].reset_index(drop=True)
    bundle = train_bundle(train, config, seed=seed)
    artifact_root = Path(artifacts_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    bundle_path = artifact_root / "offline_policy_bundle.joblib"
    joblib.dump(bundle, bundle_path, compress=3)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    training_summary = {
        "verdict": "TRAINED",
        "bundle": str(bundle_path),
        "bundle_sha256": _sha256(bundle_path),
        "dataset_sha256": _sha256(Path(data_path)),
        "config": asdict(config),
        "validity_gates": checks,
        **bundle.training_summary,
    }
    (output / "metrics.json").write_text(json.dumps(training_summary, indent=2), encoding="utf-8")
    support = train.groupby(["slice_type", "action"], sort=True).size().rename("rows").reset_index()
    support.to_csv(output / "action_support.csv", index=False)
    critic_frame = pd.DataFrame.from_dict(
        bundle.training_summary["critic_fit"], orient="index"
    ).reset_index(names="critic")
    critic_frame.to_csv(output / "critic_fit.csv", index=False)
    cards = "".join(
        [
            f"<div class='card'><div class='label'>Train rows</div><div class='value'>{len(train)}</div></div>",
            "<div class='card'><div class='label'>Episodes</div>"
            f"<div class='value'>{train['episode_id'].nunique()}</div></div>",
            f"<div class='card'><div class='label'>Actions</div><div class='value'>{len(bundle.actions)}</div></div>",
            f"<div class='card'><div class='label'>Seed</div><div class='value'>{seed}</div></div>",
        ]
    )
    body = (
        f"<div class='grid'>{cards}</div>"
        "<section><h2>Independent critics</h2>"
        "<p>SLA, energy-proxy and stability critics are trained independently. "
        "The uncertainty critic is trained on grouped out-of-fold residual magnitude.</p>"
        f"{_table_html(critic_frame)}</section>"
        f"<section><h2>Action support</h2>{_table_html(support)}</section>"
        "<section><h2>Conservative fallback</h2><pre>"
        f"{html.escape(json.dumps(bundle.fallback_by_slice, indent=2))}</pre></section>"
        "<section><h2>Training semantics</h2><p class='note'>Energy is a trace-derived proxy, "
        "not measured joules. Stability penalizes policy changes and abrupt KPI movement. "
        "The OOD model is an Isolation Forest on robust-scaled state features. Planning uses "
        "Fitted-Q over the composite immediate utility.</p></section>"
    )
    (output / "report.html").write_text(
        _page(
            "Offline RAN policy training report",
            "COMMAG · multi-objective critic training",
            "TRAINED",
            body,
        ),
        encoding="utf-8",
    )
    return training_summary


def evaluate_study(
    data_path: Path | str,
    artifacts_dir: Path | str,
    output_dir: Path | str,
    *,
    seeds: int,
    base_seed: int,
) -> dict[str, Any]:
    frame = pd.read_csv(data_path)
    checks = validate_dataset(frame)
    train = frame[frame["split"].eq("train")].reset_index(drop=True)
    test = frame[frame["split"].eq("test")].reset_index(drop=True)
    bundle_path = Path(artifacts_dir) / "offline_policy_bundle.joblib"
    bundle: OfflinePolicyBundle = joblib.load(bundle_path)

    main, decisions = evaluate_bundle(bundle, test, variant="full", measure_latency=True)
    baseline = _baseline_comparison(bundle, test)
    episodes = _episode_metrics(bundle, test)
    final_ablations = []
    for variant in (
        "full",
        "without_safety_gate",
        "without_ood_gate",
        "without_planning",
        "without_real_data_adaptation",
    ):
        metrics, _ = evaluate_bundle(bundle, test, variant=variant, measure_latency=False)
        final_ablations.append({k: v for k, v in metrics.items() if not isinstance(v, dict)})
    ablation_final = pd.DataFrame(final_ablations)

    seed_frame, ablation_seed_frame, baseline_seed_frame = run_seed_study(
        train,
        test,
        bundle.config,
        base_seed=base_seed,
        seeds=seeds,
    )
    key_metrics = [
        "selected_dm_sla_violation_rate",
        "selected_dm_mean_energy_proxy",
        "selected_dm_mean_stability_cost",
        "selected_dm_mean_utility",
        "selected_dm_estimated_utility_uplift",
        "policy_churn_rate",
        "rollback_rate",
        "mean_ood_score",
        "mean_uncertainty",
        "batch_throughput_decisions_s",
    ]
    seed_summary = _summary_stats(seed_frame, [], key_metrics)
    ablation_summary = _summary_stats(ablation_seed_frame, ["variant"], key_metrics[:-1])
    baseline_summary = _summary_stats(
        baseline_seed_frame,
        ["model"],
        [
            "logged_action_macro_f1",
            "logged_action_agreement",
            "selected_dm_sla_violation_rate",
            "selected_dm_mean_energy_proxy",
            "selected_dm_mean_stability_cost",
            "selected_dm_mean_utility",
            "policy_churn_rate",
            "prediction_throughput_decisions_s",
        ],
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    decisions_output = test[["episode_id", "timestamp_s", "slice_type", "action"]].copy()
    decisions_output = pd.concat([decisions_output.reset_index(drop=True), decisions.reset_index(drop=True)], axis=1)
    decisions_output.to_csv(destination / "decisions.csv.gz", index=False, compression="gzip")
    baseline.to_csv(destination / "baseline_comparison.csv", index=False)
    episodes.to_csv(destination / "per_episode.csv", index=False)
    ablation_final.to_csv(destination / "ablations.csv", index=False)
    seed_frame.to_csv(destination / "per_seed.csv", index=False)
    ablation_seed_frame.to_csv(destination / "ablations_per_seed.csv", index=False)
    baseline_seed_frame.to_csv(destination / "baselines_per_seed.csv", index=False)
    seed_summary.to_csv(destination / "seed_summary.csv", index=False)
    ablation_summary.to_csv(destination / "ablation_summary.csv", index=False)
    baseline_summary.to_csv(destination / "baseline_summary.csv", index=False)

    structural = {
        **checks,
        "robustness study uses at least 20 independent seeds": seeds >= 20,
        "all requested baseline families are present": set(BASELINE_NAMES).issubset(set(baseline["model"])),
        "all four requested ablations are present": set(ablation_final["variant"]) >= {
            "without_safety_gate",
            "without_ood_gate",
            "without_planning",
            "without_real_data_adaptation",
        },
        "latency and throughput are measured": bool(main.get("latency")) and main["batch_throughput_decisions_s"] > 0,
    }
    verdict = "BENCHMARK-READY" if all(structural.values()) else "INCOMPLETE"
    summary = {
        "verdict": verdict,
        "data_path": str(data_path),
        "data_sha256": _sha256(Path(data_path)),
        "bundle_sha256": _sha256(bundle_path),
        "seed_count": seeds,
        "seed_range": [base_seed, base_seed + seeds - 1],
        "train_rows": len(train),
        "test_rows": len(test),
        "train_episodes": int(train["episode_id"].nunique()),
        "test_episodes": int(test["episode_id"].nunique()),
        "main": main,
        "structural_checks": structural,
        "interpretation": {
            "observed": "Logged-action outcome columns are observations from held-out traces.",
            "direct_method": (
                "Selected-policy SLA, energy, stability, utility and uplift are model-based direct-method estimates. "
                "They are not observed interventions and must not be described as causal online gains."
            ),
            "energy": "Energy is a normalized proxy derived from power multiplier, PRB grant/load and slice pressure.",
            "latency": "Latency is host/container inference latency, not RIC-to-gNB end-to-end control latency.",
        },
    }
    (destination / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    latency = main["latency"]
    cards = "".join(
        [
            f"<div class='card'><div class='label'>Seeds</div><div class='value'>{seeds}</div></div>",
            "<div class='card'><div class='label'>Held-out episodes</div>"
            f"<div class='value'>{test['episode_id'].nunique()}</div></div>",
            "<div class='card'><div class='label'>DM SLA violation</div>"
            f"<div class='value'>{main['selected_dm_sla_violation_rate']:.3f}</div></div>",
            "<div class='card'><div class='label'>Energy proxy</div>"
            f"<div class='value'>{main['selected_dm_mean_energy_proxy']:.3f}</div></div>",
            "<div class='card'><div class='label'>Policy churn</div>"
            f"<div class='value'>{main['policy_churn_rate']:.3f}</div></div>",
            "<div class='card'><div class='label'>Rollback</div>"
            f"<div class='value'>{main['rollback_rate']:.3f}</div></div>",
            "<div class='card'><div class='label'>p95 decision latency</div>"
            f"<div class='value'>{latency['p95_ms']:.2f} ms</div></div>",
            "<div class='card'><div class='label'>Batch throughput</div>"
            f"<div class='value'>{main['batch_throughput_decisions_s']:.0f}/s</div></div>",
        ]
    )
    gates = "".join(
        f"<li>{'PASS' if passed else 'FAIL'} — {html.escape(name)}</li>"
        for name, passed in structural.items()
    )
    body = (
        f"<div class='grid'>{cards}</div>"
        "<p class='note'><strong>Counterfactual warning:</strong> selected-policy SLA, energy, "
        "stability, utility and uplift are direct-method estimates from fixed logs, not observed "
        "intervention effects.</p>"
        f"<section><h2>Validity gates</h2><ul>{gates}</ul></section>"
        "<section><h2>Main held-out evaluation</h2>"
        f"<pre>{html.escape(json.dumps(main, indent=2))}</pre></section>"
        f"<section><h2>Baseline comparison</h2>{_table_html(baseline)}</section>"
        f"<section><h2>Ablations</h2>{_table_html(ablation_final)}</section>"
        f"<section><h2>{seeds}-seed robustness summary</h2>{_table_html(seed_summary)}</section>"
        f"<section><h2>Ablation robustness</h2>{_table_html(ablation_summary)}</section>"
        f"<section><h2>Baseline robustness</h2>{_table_html(baseline_summary)}</section>"
        f"<section><h2>Held-out episode results</h2>{_table_html(episodes, limit=40)}</section>"
        "<section><h2>Paper-use guidance</h2><ul>"
        "<li>Use <code>per_seed.csv</code> for confidence intervals and statistical tests.</li>"
        "<li>Use <code>ablations_per_seed.csv</code> for matched ablation comparisons.</li>"
        "<li>Use <code>baselines_per_seed.csv</code> for model-family comparisons.</li>"
        "<li>Use <code>per_episode.csv</code> to show experiment/episode heterogeneity.</li>"
        "<li>Report energy explicitly as a proxy and uplift explicitly as direct-method estimated uplift.</li>"
        "</ul></section>"
    )
    (destination / "report.html").write_text(
        _page(
            "COMMAG multi-objective offline policy report",
            "Experiment-separated offline RAN policy learning",
            verdict,
            body,
        ),
        encoding="utf-8",
    )
    return summary


def load_config(path: Path | str | None) -> StudyConfig:
    if path is None:
        config = StudyConfig()
    else:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if "offline_policy" in payload:
            payload = payload["offline_policy"] or {}
        allowed = set(StudyConfig.__dataclass_fields__)
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown offline-policy config keys: {', '.join(unknown)}")
        config = StudyConfig(**payload)
    config.validate()
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="COMMAG multi-objective offline policy study")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-report", help="Validate prepared COMMAG data and write preparation evidence")
    prepare.add_argument("--data", default="data/prepared/commag/commag_transitions.csv.gz")
    prepare.add_argument("--manifest", default="data/prepared/commag/commag_manifest.json")
    prepare.add_argument("--output", default="results/prepare-data")

    train = sub.add_parser("train", help="Train independent critics, OOD detector, planning model and baselines")
    train.add_argument("--data", default="data/prepared/commag/commag_transitions.csv.gz")
    train.add_argument("--artifacts", default="artifacts/offline-policy")
    train.add_argument("--output", default="results/train")
    train.add_argument("--config", default="configs/offline_policy.yaml")
    train.add_argument("--seed", type=int, default=42)

    evaluate = sub.add_parser("evaluate", help="Held-out evaluation, baselines, ablations and multi-seed study")
    evaluate.add_argument("--data", default="data/prepared/commag/commag_transitions.csv.gz")
    evaluate.add_argument("--artifacts", default="artifacts/offline-policy")
    evaluate.add_argument("--output", default="results/test")
    evaluate.add_argument("--seeds", type=int, default=0, help="0 means use robustness_seeds from config")
    evaluate.add_argument("--base-seed", type=int, default=1000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-report":
        result = prepare_report(args.data, args.manifest, args.output)
    elif args.command == "train":
        config = load_config(args.config)
        result = train_study(args.data, args.artifacts, args.output, config=config, seed=args.seed)
    elif args.command == "evaluate":
        bundle: OfflinePolicyBundle = joblib.load(Path(args.artifacts) / "offline_policy_bundle.joblib")
        seeds = args.seeds or bundle.config.robustness_seeds
        result = evaluate_study(args.data, args.artifacts, args.output, seeds=seeds, base_seed=args.base_seed)
    else:  # pragma: no cover
        raise RuntimeError(args.command)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
