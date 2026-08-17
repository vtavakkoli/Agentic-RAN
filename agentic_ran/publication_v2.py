"""Full slice-traffic COMMAG publication benchmark.

Covers every existing UE metrics file in the configured COMMAG scenario/config/
experiment/BS cells, adds seen/unseen condition tests, FQI and discrete CQL
baselines, policy-shortcut diagnostics, paired episode statistics, and an exact
upstream-PPO export hook.
"""
from __future__ import annotations

import argparse, json, urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier
from sklearn.metrics import f1_score

from agentic_ran.commag import (
    COMMAG_REPOSITORY, COMMAG_REVISION, STATE_COLUMNS,
    _read_commag_trace, _sha256, _to_transitions, download_commag_core,
)
from agentic_ran.offline_policy import (
    StudyConfig, _all_predictions, _candidate_matrix, _current_actions,
    _derive_targets, _state_matrix, evaluate_bundle, train_bundle,
)

SCENARIOS = {"rome_static_close", "rome_static_medium", "rome_static_far", "rome_slow_close"}
SHIFT = {"rome_static_far": "rf_distance_shift", "rome_slow_close": "mobility_shift"}
SCHEDULERS = {0: "round_robin", 1: "waterfilling", 2: "proportional_fair"}


@dataclass(frozen=True)
class PubConfig:
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
    fqi_iterations: int = 6
    gamma: float = 0.97
    cql_alpha: float = 0.15
    cql_epochs: int = 180
    cql_learning_rate: float = 0.01

    def validate(self) -> None:
        a, b, c = map(set, (self.train_configs, self.validation_configs, self.test_configs))
        if a & b or a & c or b & c:
            raise ValueError("train/validation/test config groups overlap")
        if any(not 0 <= int(x) <= 17 for x in (*a, *b, *c)):
            raise ValueError("COMMAG configs must be in [0,17]")
        if (set(self.train_scenarios) | set(self.unseen_scenarios)) - SCENARIOS:
            raise ValueError("unknown COMMAG scenario")
        if set(self.train_scenarios) & set(self.unseen_scenarios):
            raise ValueError("seen and unseen scenarios overlap")
        if not set(self.base_stations).issubset({1, 2, 3, 4}):
            raise ValueError("BS must be in {1,2,3,4}")
        if self.bootstrap_samples < 100 or self.permutation_samples < 100:
            raise ValueError("statistics require >=100 resamples")


def load_config(path: str | Path) -> PubConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    data = data.get("publication_benchmark", data)
    for key in (
        "train_configs", "validation_configs", "test_configs", "train_scenarios",
        "unseen_scenarios", "base_stations", "experiments",
    ):
        data[key] = tuple(data[key])
    cfg = PubConfig(**data)
    cfg.validate()
    return cfg


def _tree(raw_dir: str | Path) -> list[str]:
    cache = Path(raw_dir) / f".tree-{COMMAG_REVISION}.json"
    if cache.exists() and cache.stat().st_size:
        payload = json.loads(cache.read_text(encoding="utf-8"))
    else:
        cache.parent.mkdir(parents=True, exist_ok=True)
        url = (
            "https://api.github.com/repos/wineslab/colosseum-oran-commag-dataset/"
            f"git/trees/{COMMAG_REVISION}?recursive=1"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Agentic-RAN publication benchmark"})
        with urllib.request.urlopen(req, timeout=180) as r:
            payload = json.loads(r.read().decode())
        cache.write_text(json.dumps(payload), encoding="utf-8")
    if payload.get("truncated"):
        raise RuntimeError("upstream tree truncated; refusing incomplete full benchmark")
    return [x["path"] for x in payload.get("tree", []) if x.get("type") == "blob"]


def filter_paths(tree: Sequence[str], cfg: PubConfig) -> list[str]:
    scenarios = set(cfg.train_scenarios) | set(cfg.unseen_scenarios)
    configs = {f"tr{x}" for x in (*cfg.train_configs, *cfg.validation_configs, *cfg.test_configs)}
    exps = {f"exp{x}" for x in cfg.experiments}
    bss = {f"bs{x}" for x in cfg.base_stations}
    out: list[str] = []
    for path in tree:
        p = Path(path).parts
        if len(p) < 7 or p[0] != "slice_traffic":
            continue
        scenario, tr, exp, bs = p[1:5]
        if (
            scenario in scenarios
            and tr in configs
            and exp in exps
            and bs in bss
            and p[5] == f"slices_{bs}"
            and p[-1].endswith("_metrics.csv")
        ):
            out.append(path)
    expected = len(scenarios) * len(configs) * len(exps) * len(bss)
    cells = {tuple(Path(x).parts[1:5]) for x in out}
    if len(cells) != expected:
        raise RuntimeError(f"COMMAG discovery incomplete: {len(cells)}/{expected} scenario/config/exp/BS cells")
    return sorted(out)


def _split(frame: pd.DataFrame, cfg: PubConfig) -> pd.DataFrame:
    out = frame.copy()
    tr = out.training_config.str.removeprefix("tr").astype(int)
    sc = out.scenario.astype(str)
    split = np.full(len(out), "excluded", object)
    split[np.isin(tr, cfg.train_configs) & sc.isin(cfg.train_scenarios)] = "train"
    split[np.isin(tr, cfg.validation_configs) & sc.isin(cfg.train_scenarios)] = "validation"
    split[np.isin(tr, cfg.test_configs) & sc.isin(cfg.train_scenarios)] = "test_seen"
    split[sc.isin(cfg.unseen_scenarios)] = "test_unseen"
    out["publication_split"] = split
    out["shift_type"] = sc.map(SHIFT).fillna("seen_condition")
    return out[out.publication_split.ne("excluded")].reset_index(drop=True)


def prepare(
    raw_dir: str | Path,
    output: str | Path,
    cfg: PubConfig,
    workers: int = 4,
    max_rows: int | None = None,
) -> dict[str, Any]:
    paths = filter_paths(_tree(raw_dir), cfg)
    files = download_commag_core(raw_dir, paths, revision=COMMAG_REVISION, workers=workers)
    obs = pd.concat(
        [_read_commag_trace(f, p, max_rows=max_rows) for f, p in zip(files, paths, strict=True)],
        ignore_index=True,
    )
    data = _split(_to_transitions(obs), cfg)
    required = {"train", "validation", "test_seen", "test_unseen"}
    if not required.issubset(set(data.publication_split)):
        raise ValueError("one or more publication splits are empty")
    sets = {k: set(g.episode_id.astype(str)) for k, g in data.groupby("publication_split")}
    overlap = {
        f"{a}__{b}": len(sets[a] & sets[b])
        for i, a in enumerate(sorted(sets))
        for b in sorted(sets)[i + 1 :]
    }
    if any(overlap.values()):
        raise ValueError(f"episode leakage: {overlap}")
    dest = Path(output)
    dest.mkdir(parents=True, exist_ok=True)
    fn = dest / "commag_publication_transitions.csv.gz"
    data.to_csv(fn, index=False, compression={"method": "gzip", "compresslevel": 9, "mtime": 0})
    manifest = {
        "source_repository": COMMAG_REPOSITORY,
        "source_revision": COMMAG_REVISION,
        "profile": "full-slice-traffic-publication",
        "raw_files": len(files),
        "raw_bytes": int(sum(f.stat().st_size for f in files)),
        "rows": len(data),
        "scenarios": sorted(data.scenario.unique()),
        "training_configs": sorted(data.training_config.unique()),
        "base_stations": sorted(data.base_station.unique()),
        "experiments": sorted(data.experiment.unique()),
        "split_rows": data.publication_split.value_counts().sort_index().to_dict(),
        "split_episodes": data.groupby("publication_split").episode_id.nunique().to_dict(),
        "episode_overlap": overlap,
        "prepared_sha256": _sha256(fn),
    }
    (dest / "commag_publication_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _support(bundle: Any, frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            frame.slice_type.astype(str).map(
                lambda s, action=action: action in bundle.actions_by_slice.get(s, set())
            )
            for action in bundle.actions
        ]
    ).astype(bool)


def fit_fqi(train: pd.DataFrame, bundle: Any, cfg: PubConfig, seed: int) -> ExtraTreesRegressor:
    actions = train.action.astype(str).to_numpy()
    x = _candidate_matrix(train, bundle.actions, actions)
    reward = train._utility_reward.to_numpy(float)
    done = train.done.astype(bool).to_numpy()
    target = reward.copy()
    model: ExtraTreesRegressor | None = None
    for iteration in range(cfg.fqi_iterations):
        model = ExtraTreesRegressor(
            n_estimators=bundle.config.estimators,
            min_samples_leaf=bundle.config.min_samples_leaf,
            max_features=0.85,
            random_state=seed + iteration,
            n_jobs=bundle.config.parallel_jobs,
        ).fit(x, target)
        next_q = np.column_stack(
            [model.predict(_candidate_matrix(train, bundle.actions, action, prefix="next_")) for action in bundle.actions]
        )
        target = reward + cfg.gamma * (~done).astype(float) * np.max(next_q, axis=1)
    assert model is not None
    return model


@dataclass
class LinearCQL:
    actions: list[str]
    mean: np.ndarray
    scale: np.ndarray
    w: np.ndarray
    b: np.ndarray

    def q(self, frame: pd.DataFrame, prefix: str = "") -> np.ndarray:
        x = (_state_matrix(frame, prefix=prefix) - self.mean) / self.scale
        return x @ self.w.T + self.b


def fit_cql(train: pd.DataFrame, bundle: Any, cfg: PubConfig, seed: int) -> LinearCQL:
    rng = np.random.default_rng(seed)
    x0 = _state_matrix(train)
    xn0 = _state_matrix(train, prefix="next_")
    mean = x0.mean(0)
    scale = x0.std(0)
    scale[scale < 1e-6] = 1
    x = (x0 - mean) / scale
    xn = (xn0 - mean) / scale
    n, d = x.shape
    k = len(bundle.actions)
    index = {action: i for i, action in enumerate(bundle.actions)}
    logged = np.array([index[str(action)] for action in train.action])
    one_hot = np.eye(k)[logged]
    reward = train._utility_reward.to_numpy(float)
    done = train.done.astype(bool).to_numpy()
    w = rng.normal(0, 0.01, (k, d))
    b = np.zeros(k)
    mw = np.zeros_like(w)
    vw = np.zeros_like(w)
    mb = np.zeros_like(b)
    vb = np.zeros_like(b)
    beta1, beta2 = 0.9, 0.999
    for step in range(1, cfg.cql_epochs + 1):
        q = x @ w.T + b
        next_q = xn @ w.T + b
        target = reward + cfg.gamma * (~done).astype(float) * np.max(next_q, 1)
        residual = q[np.arange(n), logged] - target
        shifted = q - q.max(1, keepdims=True)
        softmax = np.exp(shifted)
        softmax /= softmax.sum(1, keepdims=True)
        grad_q = 2 * residual[:, None] * one_hot / n + cfg.cql_alpha * (softmax - one_hot) / n
        grad_w = grad_q.T @ x + 1e-5 * w
        grad_b = grad_q.sum(0)
        mw = beta1 * mw + (1 - beta1) * grad_w
        vw = beta2 * vw + (1 - beta2) * grad_w * grad_w
        mb = beta1 * mb + (1 - beta1) * grad_b
        vb = beta2 * vb + (1 - beta2) * grad_b * grad_b
        w -= cfg.cql_learning_rate * (mw / (1 - beta1**step)) / (np.sqrt(vw / (1 - beta2**step)) + 1e-8)
        b -= cfg.cql_learning_rate * (mb / (1 - beta1**step)) / (np.sqrt(vb / (1 - beta2**step)) + 1e-8)
    return LinearCQL(list(bundle.actions), mean, scale, w, b)


def score(bundle: Any, frame: pd.DataFrame, actions: Sequence[str], name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = _derive_targets(frame, bundle.energy_scale, bundle.config)
    predictions = _all_predictions(bundle, targets, real_data_adaptation=True)
    action_index = {action: idx for idx, action in enumerate(bundle.actions)}
    slices = targets.slice_type.astype(str).to_numpy()
    final: list[str] = []
    fallback: list[bool] = []
    for row, action in enumerate(actions):
        action = str(action)
        bad = action not in action_index or action not in bundle.actions_by_slice.get(slices[row], set())
        final.append(bundle.fallback_by_slice[slices[row]] if bad else action)
        fallback.append(bad)
    indices = np.array([action_index[action] for action in final])
    rows = np.arange(len(targets))
    sla = predictions["sla"][rows, indices]
    energy = predictions["energy"][rows, indices]
    stability = predictions["stability"][rows, indices]
    denom = bundle.config.sla_weight + bundle.config.energy_weight + bundle.config.stability_weight
    utility = 1 - (
        bundle.config.sla_weight * sla
        + bundle.config.energy_weight * energy
        + bundle.config.stability_weight * stability
    ) / denom
    current = _current_actions(targets)
    selected = np.array(final, object)
    detail = pd.DataFrame(
        {
            "episode_id": targets.episode_id.astype(str),
            "scenario": targets.scenario.astype(str),
            "shift_type": targets.shift_type.astype(str),
            "model": name,
            "selected_action": selected,
            "utility": utility,
            "sla_violation": sla > bundle.config.sla_threshold,
            "energy": energy,
            "stability": stability,
            "policy_churn": selected != current,
            "unsupported_fallback": fallback,
        }
    )
    summary = pd.DataFrame(
        [
            {
                "model": name,
                "selected_dm_mean_utility": float(utility.mean()),
                "selected_dm_sla_violation_rate": float((sla > bundle.config.sla_threshold).mean()),
                "selected_dm_mean_energy_proxy": float(energy.mean()),
                "selected_dm_mean_stability_cost": float(stability.mean()),
                "policy_churn_rate": float((selected != current).mean()),
                "unsupported_action_fallback_rate": float(np.mean(fallback)),
            }
        ]
    )
    return summary, detail


def paired(proposed: pd.DataFrame, baseline: pd.DataFrame, cfg: PubConfig, seed: int) -> dict[str, float | int]:
    left = proposed.groupby("episode_id").utility.mean()
    right = baseline.groupby("episode_id").utility.mean()
    common = left.index.intersection(right.index)
    diff = (left.loc[common] - right.loc[common]).to_numpy(float)
    if len(diff) < 2:
        raise ValueError("need >=2 paired episodes")
    rng = np.random.default_rng(seed)
    bootstrap = diff[rng.integers(0, len(diff), (cfg.bootstrap_samples, len(diff)))].mean(1)
    observed = float(diff.mean())
    extreme = 0
    remaining = cfg.permutation_samples
    while remaining:
        batch = min(1000, remaining)
        null = (rng.choice([-1.0, 1.0], (batch, len(diff))) * diff).mean(1)
        extreme += int((np.abs(null) >= abs(observed)).sum())
        remaining -= batch
    std = float(diff.std(ddof=1))
    return {
        "episodes": len(diff),
        "mean_paired_delta_utility": observed,
        "bootstrap_ci95_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(bootstrap, 0.975)),
        "paired_sign_permutation_p": float((extreme + 1) / (cfg.permutation_samples + 1)),
        "cohens_dz": observed / std if std > 1e-12 else 0.0,
    }


def shortcut(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> pd.DataFrame:
    variants = {
        "full_state": STATE_COLUMNS,
        "without_scheduler": [c for c in STATE_COLUMNS if c != "scheduler_code"],
        "without_prb": [c for c in STATE_COLUMNS if c != "slice_prb"],
        "without_scheduler_and_prb": [c for c in STATE_COLUMNS if c not in {"scheduler_code", "slice_prb"}],
    }
    y_train = train.action.astype(str).to_numpy()
    y_test = test.action.astype(str).to_numpy()
    changed = _current_actions(test) != y_test
    output = []
    for name, columns in variants.items():
        def matrix(frame: pd.DataFrame) -> np.ndarray:
            slices = pd.get_dummies(frame.slice_type).reindex(columns=["eMBB", "mMTC", "URLLC"], fill_value=0)
            return np.column_stack([frame[columns].to_numpy(float), slices.to_numpy(float)])
        prediction = HistGradientBoostingClassifier(random_state=seed).fit(matrix(train), y_train).predict(matrix(test))
        output.append(
            {
                "variant": name,
                "macro_f1_all": float(f1_score(y_test, prediction, average="macro", zero_division=0)),
                "agreement_all": float(np.mean(y_test == prediction)),
                "policy_change_rows": int(changed.sum()),
                "macro_f1_policy_change": (
                    float(f1_score(y_test[changed], prediction[changed], average="macro", zero_division=0))
                    if changed.any() else None
                ),
                "agreement_policy_change": float(np.mean(y_test[changed] == prediction[changed])) if changed.any() else None,
            }
        )
    return pd.DataFrame(output)


def ppo_actions(path: Path, frame: pd.DataFrame) -> np.ndarray | None:
    if not path.exists():
        return None
    exported = pd.read_csv(path, low_memory=False)
    key = ["scenario", "training_config", "experiment", "base_station", "timestamp_s", "slice_type"]
    missing = set(key + ["scheduler_code"]) - set(exported)
    if missing:
        raise ValueError(f"PPO export missing {sorted(missing)}")
    left = frame.reset_index(drop=True).copy()
    left["_order"] = np.arange(len(left))
    merged = left.merge(
        exported[key + ["scheduler_code"]].rename(columns={"scheduler_code": "ppo_scheduler_code"}),
        on=key,
        how="left",
        validate="many_to_one",
    ).sort_values("_order")
    if merged.ppo_scheduler_code.isna().any():
        return None
    scheduler = merged.ppo_scheduler_code.round().astype(int).map(SCHEDULERS)
    prb = merged.next_slice_prb.round().astype(int).astype(str)
    return (scheduler + ":prb=" + prb).to_numpy(object)


def evaluate(
    data_path: str | Path,
    output: str | Path,
    cfg: PubConfig,
    seed: int | None = None,
    ppo_export: str | Path | None = None,
) -> dict[str, Any]:
    seed = cfg.random_seed if seed is None else seed
    frame = pd.read_csv(data_path, low_memory=False)
    train = frame[frame.publication_split.eq("train")].copy()
    validation = frame[frame.publication_split.eq("validation")].copy()
    seen = frame[frame.publication_split.eq("test_seen")].copy()
    unseen = frame[frame.publication_split.eq("test_unseen")].copy()
    if min(map(len, (train, validation, seen, unseen))) == 0:
        raise ValueError("empty publication split")
    bundle = train_bundle(train.assign(split="train"), StudyConfig(), seed)
    train_targets = _derive_targets(train, bundle.energy_scale, bundle.config)
    fqi = fit_fqi(train_targets, bundle, cfg, seed + 10)
    cql = fit_cql(train_targets, bundle, cfg, seed + 20)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    ppo_path = Path(ppo_export) if ppo_export else Path(data_path).with_name("original_ppo_actions.csv.gz")
    shortcut_tables = []
    summaries = []
    details = []
    statistics: dict[str, Any] = {}
    ood: dict[str, Any] = {}
    ppo_ok = False
    for split_name, test in (("seen", seen), ("unseen", unseen)):
        shortcut_table = shortcut(train, test, seed)
        shortcut_table.insert(0, "split", split_name)
        shortcut_tables.append(shortcut_table)
        eval_frame = test.assign(split="test")
        metrics, decisions = evaluate_bundle(bundle, eval_frame, variant="full", measure_latency=True)
        proposed = pd.DataFrame(
            {
                "episode_id": eval_frame.episode_id.astype(str),
                "scenario": eval_frame.scenario.astype(str),
                "shift_type": eval_frame.shift_type.astype(str),
                "model": "proposed",
                "selected_action": decisions.selected_action.astype(str),
                "utility": decisions.selected_direct_utility,
                "sla_violation": decisions.selected_sla_cost > bundle.config.sla_threshold,
                "energy": decisions.selected_energy_cost,
                "stability": decisions.selected_stability_cost,
                "policy_churn": decisions.policy_churn,
                "ood_score": decisions.ood_score,
                "ood_fallback": decisions.fallback_reason.eq("ood"),
            }
        )
        details.append(proposed)
        summaries.append(
            pd.DataFrame(
                [
                    {
                        "split": split_name,
                        "model": "proposed",
                        "selected_dm_mean_utility": metrics["selected_dm_mean_utility"],
                        "selected_dm_sla_violation_rate": metrics["selected_dm_sla_violation_rate"],
                        "selected_dm_mean_energy_proxy": metrics["selected_dm_mean_energy_proxy"],
                        "selected_dm_mean_stability_cost": metrics["selected_dm_mean_stability_cost"],
                        "policy_churn_rate": metrics["policy_churn_rate"],
                        "rollback_rate": metrics["rollback_rate"],
                        "ood_fallback_rate": metrics["ood_fallback_rate"],
                        "p95_latency_ms": metrics["latency"]["p95_ms"],
                    }
                ]
            )
        )
        fqi_q = np.column_stack(
            [fqi.predict(_candidate_matrix(eval_frame, bundle.actions, action)) for action in bundle.actions]
        )
        fqi_q = np.where(_support(bundle, eval_frame), fqi_q, -np.inf)
        fqi_summary, fqi_detail = score(
            bundle, eval_frame, np.array(bundle.actions, object)[fqi_q.argmax(1)], "fqi"
        )
        cql_q = np.where(_support(bundle, eval_frame), cql.q(eval_frame), -np.inf)
        cql_summary, cql_detail = score(
            bundle, eval_frame, np.array(bundle.actions, object)[cql_q.argmax(1)], "cql_linear"
        )
        behavior_actions = bundle.baseline_models["hist_gradient_boosting"].predict(_state_matrix(eval_frame))
        behavior_summary, behavior_detail = score(bundle, eval_frame, behavior_actions, "behavior_hgb")
        baselines = [
            ("fqi", fqi_summary, fqi_detail),
            ("cql_linear", cql_summary, cql_detail),
            ("behavior_hgb", behavior_summary, behavior_detail),
        ]
        original_actions = ppo_actions(ppo_path, eval_frame)
        if original_actions is not None:
            ppo_summary, ppo_detail = score(
                bundle, eval_frame, original_actions, "original_commag_ppo_scheduler"
            )
            baselines.append(("original_commag_ppo_scheduler", ppo_summary, ppo_detail))
            ppo_ok = True
        for name, summary, detail in baselines:
            summary.insert(0, "split", split_name)
            summaries.append(summary)
            details.append(detail)
            statistics[f"{split_name}:proposed_vs_{name}"] = paired(proposed, detail, cfg, seed)
        ood[split_name] = {
            "mean_ood": float(proposed.ood_score.mean()),
            "p95_ood": float(proposed.ood_score.quantile(0.95)),
            "ood_fallback_rate": float(proposed.ood_fallback.mean()),
            "by_scenario": {
                scenario: {
                    "rows": len(group),
                    "mean_ood": float(group.ood_score.mean()),
                    "p95_ood": float(group.ood_score.quantile(0.95)),
                    "ood_fallback_rate": float(group.ood_fallback.mean()),
                }
                for scenario, group in proposed.groupby("scenario")
            },
        }
    pd.concat(shortcut_tables, ignore_index=True).to_csv(destination / "policy_shortcut_test.csv", index=False)
    pd.concat(summaries, ignore_index=True).to_csv(destination / "publication_baselines.csv", index=False)
    pd.concat(details, ignore_index=True).to_csv(
        destination / "publication_decisions.csv.gz", index=False, compression="gzip"
    )
    (destination / "paired_statistics.json").write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    (destination / "ood_generalization.json").write_text(json.dumps(ood, indent=2), encoding="utf-8")
    joblib.dump({"bundle": bundle, "fqi": fqi, "cql": cql}, destination / "publication_models.joblib")
    result = {
        "verdict": "PUBLICATION-BENCHMARK-READY" if ppo_ok else "PUBLICATION-BENCHMARK-PARTIAL",
        "train_rows": len(train),
        "validation_rows": len(validation),
        "seen_test_rows": len(seen),
        "unseen_test_rows": len(unseen),
        "original_ppo_available": ppo_ok,
        "paired_statistics": statistics,
        "ood_generalization": ood,
        "limitations": [
            "counterfactual selected-policy outcomes are direct-method estimates",
            "energy is a normalized proxy",
            "original COMMAG PPO controls scheduler while RBG resizing is exogenous",
        ],
    }
    (destination / "publication_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    prepare_cmd = sub.add_parser("prepare")
    prepare_cmd.add_argument("--raw-dir", default="data/raw/commag")
    prepare_cmd.add_argument("--output", default="data/prepared/commag-publication")
    prepare_cmd.add_argument("--config", default="configs/full_commag_publication.yaml")
    prepare_cmd.add_argument("--workers", type=int, default=4)
    prepare_cmd.add_argument("--max-rows-per-file", type=int, default=0)
    evaluate_cmd = sub.add_parser("evaluate")
    evaluate_cmd.add_argument("--data", default="data/prepared/commag-publication/commag_publication_transitions.csv.gz")
    evaluate_cmd.add_argument("--output", default="results/publication")
    evaluate_cmd.add_argument("--config", default="configs/full_commag_publication.yaml")
    evaluate_cmd.add_argument("--seed", type=int, default=0)
    evaluate_cmd.add_argument("--original-ppo-export", default="")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.cmd == "prepare":
        result = prepare(args.raw_dir, args.output, cfg, args.workers, args.max_rows_per_file or None)
    else:
        result = evaluate(
            args.data,
            args.output,
            cfg,
            args.seed or None,
            args.original_ppo_export or None,
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
