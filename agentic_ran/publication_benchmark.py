"""Publication-oriented COMMAG benchmark utilities.

This module builds on :mod:`agentic_ran.commag` and :mod:`agentic_ran.offline_policy`
without changing the compact benchmark defaults.  It adds explicit scenario/config
splits, offline-RL baselines, policy-shortcut diagnostics, and paired statistics.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier
from sklearn.metrics import f1_score

from agentic_ran.commag import (
    COMMAG_REVISION,
    STATE_COLUMNS,
    _read_commag_trace,
    _sha256,
    _to_transitions,
    download_commag_core,
)
from agentic_ran.offline_policy import (
    StudyConfig,
    _candidate_matrix,
    _current_actions,
    _derive_targets,
    _fit_energy_scale,
    _state_matrix,
    evaluate_bundle,
    fit_bundle,
)

SCENARIOS = (
    "rome_static_close",
    "rome_slow_close",
    "rome_static_medium",
    "rome_slow_medium",
    "rome_static_far",
    "rome_slow_far",
)

# COMMAG IMSIs are grouped by BS in blocks of ten.  Only traffic-slice UEs are
# required by this study; each BS therefore contributes the three representative
# per-slice metric files used by the compact benchmark.
CORE_UE_FILES_ALL_BS = {
    1: ("1010123456002", "1010123456003", "1010123456004"),
    2: ("1010123456012", "1010123456013", "1010123456014"),
    3: ("1010123456022", "1010123456023", "1010123456024"),
    4: ("1010123456035", "1010123456036", "1010123456037"),
}


@dataclass(frozen=True)
class PublicationConfig:
    train_configs: tuple[int, ...]
    validation_configs: tuple[int, ...]
    test_configs: tuple[int, ...]
    train_scenarios: tuple[str, ...]
    unseen_scenarios: tuple[str, ...]
    base_stations: tuple[int, ...]
    experiments: tuple[int, ...]
    bootstrap_samples: int = 5000
    permutation_samples: int = 10000
    random_seed: int = 2026
    cql_alpha: float = 0.15
    fqi_iterations: int = 6
    gamma: float = 0.97

    def validate(self) -> None:
        if set(self.train_configs) & set(self.validation_configs):
            raise ValueError("train_configs and validation_configs overlap")
        if set(self.train_configs) & set(self.test_configs):
            raise ValueError("train_configs and test_configs overlap")
        if set(self.validation_configs) & set(self.test_configs):
            raise ValueError("validation_configs and test_configs overlap")
        for value in (*self.train_configs, *self.validation_configs, *self.test_configs):
            if not 0 <= int(value) <= 17:
                raise ValueError("COMMAG training configurations must be in [0,17]")
        unknown = (set(self.train_scenarios) | set(self.unseen_scenarios)) - set(SCENARIOS)
        if unknown:
            raise ValueError(f"Unknown COMMAG scenarios: {sorted(unknown)}")
        if set(self.train_scenarios) & set(self.unseen_scenarios):
            raise ValueError("train_scenarios and unseen_scenarios overlap")
        if not set(self.base_stations).issubset(CORE_UE_FILES_ALL_BS):
            raise ValueError("base_stations must be a subset of 1,2,3,4")
        if self.bootstrap_samples < 100 or self.permutation_samples < 100:
            raise ValueError("publication statistics require at least 100 resamples")


def load_publication_config(path: str | Path) -> PublicationConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    payload = payload.get("publication_benchmark", payload)
    tuple_fields = {
        "train_configs",
        "validation_configs",
        "test_configs",
        "train_scenarios",
        "unseen_scenarios",
        "base_stations",
        "experiments",
    }
    for name in tuple_fields:
        if name in payload:
            payload[name] = tuple(payload[name])
    config = PublicationConfig(**payload)
    config.validate()
    return config


def publication_paths(
    scenarios: Sequence[str],
    train_configs: Sequence[int],
    experiments: Sequence[int],
    base_stations: Sequence[int],
) -> list[str]:
    paths: list[str] = []
    for scenario in scenarios:
        if scenario not in SCENARIOS:
            raise ValueError(f"Unsupported COMMAG scenario: {scenario}")
        for training in train_configs:
            for experiment in experiments:
                for base_station in base_stations:
                    for imsi in CORE_UE_FILES_ALL_BS[int(base_station)]:
                        paths.append(
                            "slice_traffic/"
                            f"{scenario}/tr{int(training)}/exp{int(experiment)}/bs{int(base_station)}/"
                            f"slices_bs{int(base_station)}/{imsi}_metrics.csv"
                        )
    return paths


def _tag_split(frame: pd.DataFrame, config: PublicationConfig) -> pd.DataFrame:
    out = frame.copy()
    tr = out["training_config"].astype(str).str.removeprefix("tr").astype(int)
    scenario = out["scenario"].astype(str)
    split = np.full(len(out), "excluded", dtype=object)
    split[np.isin(tr, config.train_configs) & scenario.isin(config.train_scenarios)] = "train"
    split[np.isin(tr, config.validation_configs) & scenario.isin(config.train_scenarios)] = "validation"
    split[np.isin(tr, config.test_configs) & scenario.isin(config.train_scenarios)] = "test_seen"
    split[scenario.isin(config.unseen_scenarios)] = "test_unseen"
    out["publication_split"] = split
    return out[out["publication_split"].ne("excluded")].reset_index(drop=True)


def prepare_full_commag(
    raw_dir: str | Path,
    output_dir: str | Path,
    config: PublicationConfig,
    *,
    revision: str = COMMAG_REVISION,
    workers: int = 4,
    max_rows_per_file: int | None = None,
) -> dict[str, Any]:
    scenarios = tuple(dict.fromkeys((*config.train_scenarios, *config.unseen_scenarios)))
    configs = tuple(dict.fromkeys((*config.train_configs, *config.validation_configs, *config.test_configs)))
    paths = publication_paths(scenarios, configs, config.experiments, config.base_stations)
    files = download_commag_core(raw_dir, paths, revision=revision, workers=workers)
    measurements = pd.concat(
        [
            _read_commag_trace(path, relative, max_rows=max_rows_per_file)
            for path, relative in zip(files, paths, strict=True)
        ],
        ignore_index=True,
    )
    transitions = _tag_split(_to_transitions(measurements), config)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    dataset = destination / "commag_publication_transitions.csv.gz"
    transitions.to_csv(dataset, index=False, compression={"method": "gzip", "compresslevel": 9, "mtime": 0})
    manifest = {
        "source_revision": revision,
        "profile": "full-publication",
        "raw_files": len(files),
        "scenarios": list(scenarios),
        "training_configs": list(configs),
        "base_stations": list(config.base_stations),
        "experiments": list(config.experiments),
        "rows": len(transitions),
        "split_rows": transitions["publication_split"].value_counts().sort_index().to_dict(),
        "split_episodes": transitions.groupby("publication_split")["episode_id"].nunique().to_dict(),
        "dataset_sha256": _sha256(dataset),
        "important_note": "Counterfactual policy outcomes remain model estimates unless the logged action matches.",
    }
    (destination / "commag_publication_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _fqi_fit(
    train: pd.DataFrame,
    actions: Sequence[str],
    config: StudyConfig,
    *,
    iterations: int,
    gamma: float,
    seed: int,
    conservative_alpha: float = 0.0,
) -> ExtraTreesRegressor:
    action_values = train["action"].astype(str).to_numpy(dtype=object)
    rewards = train["_utility_reward"].to_numpy(dtype=float)
    done = train["done"].astype(float).to_numpy(dtype=float)
    x = _candidate_matrix(train, actions, action_values)
    target = rewards.copy()
    model: ExtraTreesRegressor | None = None
    action_counts = train["action"].value_counts().to_dict()
    total = max(len(train), 1)
    penalties = np.asarray(
        [conservative_alpha * -math.log(max(action_counts.get(str(a), 0), 1) / total) for a in action_values],
        dtype=float,
    )
    for step in range(max(1, int(iterations))):
        model = ExtraTreesRegressor(
            n_estimators=config.estimators,
            min_samples_leaf=config.min_samples_leaf,
            max_features=0.85,
            random_state=seed + step,
            n_jobs=config.parallel_jobs,
        )
        model.fit(x, target - penalties)
        next_q = np.column_stack(
            [model.predict(_candidate_matrix(train, actions, action, prefix="next_")) for action in actions]
        )
        target = rewards + gamma * (1.0 - done) * np.max(next_q, axis=1)
    assert model is not None
    return model


def _evaluate_q_policy(
    model: ExtraTreesRegressor,
    bundle: Any,
    frame: pd.DataFrame,
    name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = _derive_targets(frame, bundle.energy_scale, bundle.config)
    q = np.column_stack([model.predict(_candidate_matrix(targets, bundle.actions, action)) for action in bundle.actions])
    allowed = np.column_stack(
        [targets["slice_type"].astype(str).map(lambda s, a=action: a in bundle.actions_by_slice.get(s, set())).to_numpy() for action in bundle.actions]
    )
    q = np.where(allowed, q, -np.inf)
    idx = np.argmax(q, axis=1)
    selected = np.asarray(bundle.actions, dtype=object)[idx]
    # Score with the same direct-method critics used for the proposed policy.
    from agentic_ran.offline_policy import _all_predictions

    pred = _all_predictions(bundle, targets, real_data_adaptation=True)
    rows = np.arange(len(targets))
    sla = pred["sla"][rows, idx]
    energy = pred["energy"][rows, idx]
    stability = pred["stability"][rows, idx]
    denom = bundle.config.sla_weight + bundle.config.energy_weight + bundle.config.stability_weight
    utility = 1.0 - (
        bundle.config.sla_weight * sla + bundle.config.energy_weight * energy + bundle.config.stability_weight * stability
    ) / denom
    decisions = pd.DataFrame(
        {
            "episode_id": targets["episode_id"].astype(str).to_numpy(),
            "selected_action": selected,
            "utility": utility,
            "sla_violation": sla > bundle.config.sla_threshold,
            "energy": energy,
            "stability": stability,
            "policy_churn": selected != _current_actions(targets),
        }
    )
    summary = pd.DataFrame(
        [
            {
                "model": name,
                "selected_dm_mean_utility": float(np.mean(utility)),
                "selected_dm_sla_violation_rate": float(np.mean(sla > bundle.config.sla_threshold)),
                "selected_dm_mean_energy_proxy": float(np.mean(energy)),
                "selected_dm_mean_stability_cost": float(np.mean(stability)),
                "policy_churn_rate": float(np.mean(selected != _current_actions(targets))),
            }
        ]
    )
    return summary, decisions


def _shortcut_diagnostics(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> pd.DataFrame:
    variants = {
        "full_state": STATE_COLUMNS,
        "without_scheduler": [c for c in STATE_COLUMNS if c != "scheduler_code"],
        "without_prb": [c for c in STATE_COLUMNS if c != "slice_prb"],
        "without_scheduler_and_prb": [c for c in STATE_COLUMNS if c not in {"scheduler_code", "slice_prb"}],
    }
    rows: list[dict[str, Any]] = []
    y_train = train["action"].astype(str)
    y_test = test["action"].astype(str)
    changed = _current_actions(test) != y_test.to_numpy(dtype=object)
    for name, columns in variants.items():
        x_train = np.column_stack(
            [train[columns].to_numpy(dtype=float), pd.get_dummies(train["slice_type"]).reindex(columns=["eMBB", "mMTC", "URLLC"], fill_value=0).to_numpy(dtype=float)]
        )
        x_test = np.column_stack(
            [test[columns].to_numpy(dtype=float), pd.get_dummies(test["slice_type"]).reindex(columns=["eMBB", "mMTC", "URLLC"], fill_value=0).to_numpy(dtype=float)]
        )
        model = HistGradientBoostingClassifier(random_state=seed).fit(x_train, y_train)
        pred = model.predict(x_test)
        rows.append(
            {
                "variant": name,
                "macro_f1_all": float(f1_score(y_test, pred, average="macro", zero_division=0)),
                "agreement_all": float(np.mean(pred == y_test.to_numpy(dtype=object))),
                "policy_change_rows": int(np.sum(changed)),
                "macro_f1_policy_change": float(f1_score(y_test[changed], pred[changed], average="macro", zero_division=0)) if np.any(changed) else None,
                "agreement_policy_change": float(np.mean(pred[changed] == y_test.to_numpy(dtype=object)[changed])) if np.any(changed) else None,
            }
        )
    return pd.DataFrame(rows)


def _paired_episode_stats(
    proposed: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> dict[str, float]:
    left = proposed.groupby("episode_id")["utility"].mean()
    right = baseline.groupby("episode_id")["utility"].mean()
    common = left.index.intersection(right.index)
    diff = (left.loc[common] - right.loc[common]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.asarray([np.mean(rng.choice(diff, size=len(diff), replace=True)) for _ in range(bootstrap_samples)])
    observed = float(np.mean(diff))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(permutation_samples, len(diff)))
    null = np.mean(signs * diff[None, :], axis=1)
    p = float((np.sum(np.abs(null) >= abs(observed)) + 1) / (permutation_samples + 1))
    return {
        "episodes": int(len(diff)),
        "mean_paired_delta_utility": observed,
        "bootstrap_ci95_low": float(np.quantile(boot, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(boot, 0.975)),
        "paired_sign_permutation_p": p,
        "cohens_dz": float(observed / np.std(diff, ddof=1)) if len(diff) > 1 and np.std(diff, ddof=1) > 0 else 0.0,
    }


def evaluate_publication(
    data_path: str | Path,
    output_dir: str | Path,
    config: PublicationConfig,
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    seed = config.random_seed if seed is None else int(seed)
    frame = pd.read_csv(data_path, low_memory=False)
    train = frame[frame["publication_split"].eq("train")].copy()
    validation = frame[frame["publication_split"].eq("validation")].copy()
    seen = frame[frame["publication_split"].eq("test_seen")].copy()
    unseen = frame[frame["publication_split"].eq("test_unseen")].copy()
    if min(len(train), len(validation), len(seen), len(unseen)) == 0:
        raise ValueError("Publication dataset requires train, validation, seen-test and unseen-test rows")

    study_config = StudyConfig()
    # The existing bundle API uses split=train/test.  Validation is reserved for threshold/weight
    # selection by the paper workflow; the bundle itself is fit on publication train only.
    fit_frame = pd.concat(
        [train.assign(split="train"), validation.assign(split="test")], ignore_index=True
    )
    energy_scale = _fit_energy_scale(fit_frame[fit_frame["split"].eq("train")])
    fit_targets = _derive_targets(fit_frame, energy_scale, study_config)
    bundle = fit_bundle(fit_targets, config=study_config, seed=seed)

    train_targets = _derive_targets(train, bundle.energy_scale, bundle.config)
    fqi = _fqi_fit(train_targets, bundle.actions, bundle.config, iterations=config.fqi_iterations, gamma=config.gamma, seed=seed)
    cql = _fqi_fit(
        train_targets,
        bundle.actions,
        bundle.config,
        iterations=config.fqi_iterations,
        gamma=config.gamma,
        seed=seed + 100,
        conservative_alpha=config.cql_alpha,
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    shortcut = _shortcut_diagnostics(train, seen, seed)
    shortcut.to_csv(destination / "policy_shortcut_test.csv", index=False)

    all_summary: list[pd.DataFrame] = []
    statistics: dict[str, Any] = {}
    for split_name, test in (("seen", seen), ("unseen", unseen)):
        eval_test = test.assign(split="test")
        full_metrics, proposed_decisions = evaluate_bundle(bundle, eval_test, variant="full", measure_latency=True)
        proposed = pd.DataFrame(
            {
                "episode_id": eval_test["episode_id"].astype(str).to_numpy(),
                "utility": proposed_decisions["selected_direct_utility"].to_numpy(dtype=float),
            }
        )
        full_row = pd.DataFrame([{"split": split_name, "model": "proposed", **full_metrics}])
        all_summary.append(full_row)
        for model_name, model in (("fqi", fqi), ("cql_conservative_fqi", cql)):
            summary, decisions = _evaluate_q_policy(model, bundle, eval_test, model_name)
            summary.insert(0, "split", split_name)
            all_summary.append(summary)
            stats = _paired_episode_stats(
                proposed,
                decisions,
                bootstrap_samples=config.bootstrap_samples,
                permutation_samples=config.permutation_samples,
                seed=seed,
            )
            statistics[f"{split_name}:proposed_vs_{model_name}"] = stats

        ood_scores = proposed_decisions["ood_score"].to_numpy(dtype=float)
        statistics[f"{split_name}:ood"] = {
            "mean": float(np.mean(ood_scores)),
            "p95": float(np.quantile(ood_scores, 0.95)),
            "fallback_rate": float(proposed_decisions["fallback_reason"].eq("ood").mean()),
        }

    summary = pd.concat(all_summary, ignore_index=True)
    summary.to_csv(destination / "publication_baselines.csv", index=False)
    (destination / "paired_statistics.json").write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    joblib.dump({"bundle": bundle, "fqi": fqi, "cql": cql}, destination / "publication_models.joblib")

    ppo = {
        "status": "compatibility-hook",
        "reason": (
            "The upstream COMMAG repository distributes original PPO agents using the historical "
            "TensorFlow/stable-baselines stack. This benchmark records the baseline explicitly but does not "
            "silently reimplement incompatible weights. Use the optional legacy PPO container adapter for "
            "exact upstream inference when those dependencies are available."
        ),
    }
    (destination / "original_ppo_status.json").write_text(json.dumps(ppo, indent=2), encoding="utf-8")
    result = {
        "verdict": "PUBLICATION-BENCHMARK-READY",
        "train_rows": len(train),
        "validation_rows": len(validation),
        "seen_test_rows": len(seen),
        "unseen_test_rows": len(unseen),
        "shortcut_test": shortcut.to_dict(orient="records"),
        "paired_statistics": statistics,
        "original_ppo": ppo,
    }
    (destination / "publication_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Full COMMAG publication benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--raw-dir", default="data/raw/commag")
    prepare.add_argument("--output", default="data/prepared/commag-publication")
    prepare.add_argument("--config", default="configs/full_commag_publication.yaml")
    prepare.add_argument("--workers", type=int, default=4)
    prepare.add_argument("--max-rows-per-file", type=int, default=0)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--data", default="data/prepared/commag-publication/commag_publication_transitions.csv.gz")
    evaluate.add_argument("--output", default="results/publication")
    evaluate.add_argument("--config", default="configs/full_commag_publication.yaml")
    evaluate.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_publication_config(args.config)
    if args.command == "prepare":
        result = prepare_full_commag(
            args.raw_dir,
            args.output,
            config,
            workers=args.workers,
            max_rows_per_file=args.max_rows_per_file or None,
        )
    else:
        result = evaluate_publication(args.data, args.output, config, seed=args.seed or None)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
