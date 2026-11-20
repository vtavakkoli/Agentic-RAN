"""Matched-seed Gymnasium benchmark for *executed* RAN controller actions.

This module never maps COMMAG scheduler/PRB actions onto unrelated
traffic-steering or energy-saving Gym action spaces. Caller-provided factories
must implement a scientifically documented compatible observation/action
adapter. A shadow CSV or a toy backend is not publication evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_factory(spec: str) -> tuple[Any, dict[str, str]]:
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError("factory must be an explicit module:function")
    module, name = spec.split(":", 1)
    if not module or not name.isidentifier() or name.startswith("_"):
        raise ValueError("invalid module:function factory")
    function = getattr(importlib.import_module(module), name)
    if not callable(function):
        raise TypeError(f"{spec} is not callable")
    source = inspect.getsourcefile(function)
    if not source or not Path(source).is_file():
        raise ValueError(f"{spec}: a readable source file is required for provenance")
    return function, {
        "entrypoint": spec,
        "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
    }


def normalized(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [normalized(x) for x in value]
    if isinstance(value, dict):
        return {str(k): normalized(v) for k, v in value.items()}
    return value


def same_action(left: Any, right: Any) -> bool:
    return json.dumps(normalized(left), sort_keys=True) == json.dumps(
        normalized(right), sort_keys=True
    )


def finite_scalar(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a numeric scalar")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} is nonfinite")
    return number


def episode(
    env_factory: Any, policy_factory: Any, parameters: dict[str, Any], *,
    label: str, scenario: str, seed: int, steps: int,
    strict_receipt: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One new environment per policy/scenario/seed; never replay rewards."""
    env = env_factory(scenario, seed, parameters)
    lines: list[dict[str, Any]] = []
    try:
        if not hasattr(env, "reset") or not hasattr(env, "step") or not hasattr(env, "action_space"):
            raise TypeError("factory did not return a Gymnasium-like environment")
        policy = policy_factory({**parameters, "seed": seed}, env.action_space)
        if not callable(policy):
            raise TypeError(f"{label}: policy factory must return an action callable")
        observation, info = env.reset(seed=seed)
        if not isinstance(info, dict):
            raise ValueError("Gymnasium reset must return (observation, info dict)")
        previous = None
        t0 = time.monotonic()
        for index in range(steps):
            action = policy(observation, previous, index, info)
            if not bool(env.action_space.contains(action)):
                raise ValueError(
                    f"{label}: action incompatible with environment action_space; "
                    "do not silently project a COMMAG action into a different RAN use case"
                )
            transition = env.step(action)
            if not isinstance(transition, (tuple, list)) or len(transition) != 5:
                raise ValueError("Gymnasium step must return observation,reward,terminated,truncated,info")
            next_observation, reward, terminated, truncated, next_info = transition
            reward = finite_scalar(reward, "realized Gym reward")
            if not isinstance(next_info, dict):
                raise ValueError("Gym info must be a dict")
            if not isinstance(terminated, (bool, np.bool_)) or not isinstance(truncated, (bool, np.bool_)):
                raise ValueError("Gym termination flags must be bool")
            receipt = next_info.get("applied_action")
            confirmed = receipt is not None and same_action(receipt, action)
            if strict_receipt and not confirmed:
                raise ValueError(
                    "No matching applied_action receipt: external action execution cannot be verified"
                )
            for key in ("throughput_mbps", "latency_ms", "sla_violation_rate"):
                if key in next_info:
                    finite_scalar(next_info[key], key)
            lines.append({
                "policy": label, "scenario": scenario, "seed": seed, "step": index,
                "action": json.dumps(normalized(action), sort_keys=True),
                "reward": reward, "action_changed": previous is not None and not same_action(action, previous),
                "receipt_verified": confirmed,
                "throughput_mbps": next_info.get("throughput_mbps"),
                "latency_ms": next_info.get("latency_ms"),
                "sla_violation_rate": next_info.get("sla_violation_rate"),
                "terminated": bool(terminated), "truncated": bool(truncated),
            })
            previous = action
            observation, info = next_observation, next_info
            if terminated or truncated:
                break
        if not lines:
            raise RuntimeError("environment produced no executed steps")
        return {
            "policy": label, "scenario": scenario, "seed": seed, "steps": len(lines),
            "total_reward": sum(row["reward"] for row in lines),
            "mean_reward": float(np.mean([row["reward"] for row in lines])),
            "switch_rate": float(np.mean([row["action_changed"] for row in lines])),
            "receipt_fraction": float(np.mean([row["receipt_verified"] for row in lines])),
            "unique_actions": len({row["action"] for row in lines}),
            "elapsed_seconds": time.monotonic() - t0,
            **{
                "mean_" + k: float(np.mean([row[k] for row in lines]))
                if all(row[k] is not None for row in lines) else None
                for k in ("throughput_mbps", "latency_ms", "sla_violation_rate")
            },
        }, lines
    finally:
        env.close()


def paired_statistics(candidate: pd.DataFrame, baseline: pd.DataFrame, *,
                      seed: int = 2026, draws: int = 10000) -> dict[str, Any]:
    merged = candidate.merge(
        baseline, on=["scenario", "seed"], suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    if len(merged) != len(candidate) or len(merged) != len(baseline):
        raise ValueError("unmatched policy episodes; refuse unpaired performance claims")
    differences = (merged["total_reward_candidate"] - merged["total_reward_baseline"]).to_numpy(float)
    if not len(differences) or not np.isfinite(differences).all():
        raise ValueError("missing/nonfinite paired reward")
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(differences), size=(draws, len(differences)))
    means = differences[sample_indices].mean(axis=1)
    if len(differences) <= 18:
        # Exact sign-flip randomization on matched *episodes*, not individual steps.
        signs = np.arange(1 << len(differences), dtype=np.uint32)
        flipped = np.array([
            np.mean(differences * (2 * ((i >> np.arange(len(differences))) & 1).astype(int) - 1))
            for i in signs
        ])
    else:
        flipped = (
            rng.choice([-1.0, 1.0], size=(draws, len(differences))) * differences
        ).mean(axis=1)
    actual = float(np.mean(differences))
    p_value = float((np.sum(flipped >= actual - 1e-12) + (0 if len(differences) <= 18 else 1)) /
                    (len(flipped) + (0 if len(differences) <= 18 else 1)))
    sla_non_regression = None
    sla_ci = None
    if (merged["mean_sla_violation_rate_candidate"].notna().all()
            and merged["mean_sla_violation_rate_baseline"].notna().all()):
        sla_diff = (
            merged["mean_sla_violation_rate_candidate"]
            - merged["mean_sla_violation_rate_baseline"]
        ).to_numpy(dtype=float)
        sla_ci = [float(x) for x in np.quantile(sla_diff[sample_indices].mean(axis=1), [0.025, 0.975])]
        sla_non_regression = sla_ci[1] <= 0.01
    return {
        "paired_episodes": int(len(differences)),
        "mean_total_reward_delta": actual,
        "bootstrap_ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])],
        "one_sided_sign_flip_p": p_value,
        "p_method": "exact" if len(differences) <= 18 else "monte_carlo",
        "sla_regression_ci95": sla_ci,
        "sla_non_regression": sla_non_regression,
        "candidate_mean_reward": float(merged["total_reward_candidate"].mean()),
        "baseline_mean_reward": float(merged["total_reward_baseline"].mean()),
    }


def run(config_file: Path, output_dir: Path, *, require_superiority: bool = False) -> dict[str, Any]:
    params = json.loads(config_file.read_text(encoding="utf-8"))
    if params.get("schema") != "agentic-ran-gym-benchmark-v1":
        raise ValueError("unsupported RAN Gym evaluation config schema")
    backend = str(params.get("backend", ""))
    if backend not in ("toy-gym-smoke", "ns-o-ran-gym", "custom-external-gym"):
        raise ValueError("backend must explicitly identify toy or external RAN Gym")
    factories = params.get("policies")
    if not isinstance(factories, dict) or "candidate" not in factories or len(factories) < 2:
        raise ValueError("candidate plus at least one named baseline policy are required")
    if len(set(factories)) != len(factories):
        raise ValueError("duplicate policy names")
    scenarios = params.get("scenarios")
    seeds = params.get("test_seeds")
    if not isinstance(scenarios, list) or not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("nonempty distinct scenarios required")
    if not isinstance(seeds, list) or len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(
        type(x) is not int or x < 0 for x in seeds
    ):
        raise ValueError("at least two unique, nonnegative integer final-test seeds required")
    training = params.get("training_seeds", [])
    validation = params.get("validation_seeds", [])
    if set(seeds) & (set(training) | set(validation)) or set(training) & set(validation):
        raise ValueError("training/validation/final test seed leakage")
    steps = params.get("max_steps", 24)
    if type(steps) is not int or not 1 <= steps <= 10000:
        raise ValueError("max_steps outside [1, 10000]")
    if backend != "toy-gym-smoke":
        if not params.get("observation_action_mapping_documented"):
            raise ValueError("external RAN Gym needs explicit scientifically validated observation/action mapping")
        if not params.get("simulator_revision"):
            raise ValueError("external simulator revision must be recorded")
        if not params.get("frozen_model_sha256"):
            raise ValueError("external candidate must identify the frozen model SHA-256")
    if backend != "toy-gym-smoke":
        model_path = Path(params.get("frozen_model_path", ""))
        if not model_path.is_file():
            raise FileNotFoundError("External comparison requires readable frozen_model_path")
        digest = hashlib.sha256()
        with model_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        observed_sha = digest.hexdigest()
        if observed_sha != params["frozen_model_sha256"]:
            raise ValueError("Frozen model SHA-256 does not match configured model")
    env_factory, env_source = load_factory(params["environment_factory"])
    loaded = {key: load_factory(spec) for key, spec in factories.items()}
    strict_receipt = bool(params.get("require_applied_action_receipt", backend != "toy-gym-smoke"))
    episode_rows = []
    step_rows = []
    for scenario in scenarios:
        for seed in seeds:
            for label, (policy_factory, _) in loaded.items():
                print(f"[RAN Gym] {backend} {scenario} seed={seed} policy={label}", flush=True)
                result, trace = episode(
                    env_factory, policy_factory, params, label=label,
                    scenario=scenario, seed=seed, steps=steps,
                    strict_receipt=strict_receipt,
                )
                episode_rows.append(result)
                step_rows.extend(trace)
    episodes = pd.DataFrame(episode_rows)
    steps_df = pd.DataFrame(step_rows)
    candidate = episodes[episodes["policy"].eq("candidate")]
    comparisons = {}
    for baseline in factories:
        if baseline != "candidate":
            comparisons[baseline] = paired_statistics(
                candidate, episodes[episodes["policy"].eq(baseline)]
            )
    # Holm-Bonferroni adjustment of one-sided paired p-values.
    ordered = sorted(comparisons, key=lambda name: comparisons[name]["one_sided_sign_flip_p"])
    running = 0.0
    for i, name in enumerate(ordered):
        running = max(running, min(1.0, comparisons[name]["one_sided_sign_flip_p"] * (len(ordered) - i)))
        comparisons[name]["holm_adjusted_p"] = running
        c = comparisons[name]
        c["significant_gain"] = bool(c["bootstrap_ci95"][0] > 0 and running < 0.05)
    service_complete = bool(
        episodes[["mean_throughput_mbps", "mean_latency_ms", "mean_sla_violation_rate"]]
        .notna().all().all()
    )
    sla_ok = bool(comparisons) and all(
        d["sla_non_regression"] is True for d in comparisons.values()
    )
    # A *realized* advantage is only claimed when every matched comparison passes.
    all_superior = bool(comparisons) and all(
        d["significant_gain"] for d in comparisons.values()
    )
    if strict_receipt and not steps_df["receipt_verified"].all():
        raise AssertionError("receipt enforcement unexpectedly bypassed")
    report = {
        "schema": "agentic-ran-gym-benchmark-v1", "backend": backend,
        "executed_gym_step": True,
        "backend_identity_independently_verified": False,
        "real_hardware_experiment": False,
        "is_toy_smoke": backend == "toy-gym-smoke",
        "receipt_enforced": strict_receipt,
        "observation_action_mapping_documented": bool(params.get("observation_action_mapping_documented")),
        "simulator_revision": params.get("simulator_revision"),
        "frozen_model_sha256": params.get("frozen_model_sha256"),
        "environment_factory": env_source,
        "policy_factories": {key: provenance for key, (_, provenance) in loaded.items()},
        "heldout_scenarios": scenarios, "heldout_seeds": seeds,
        "episodes": len(episode_rows), "steps": len(step_rows),
        "compared_methods": list(factories), "comparisons": comparisons,
        "all_baselines_significantly_outperformed": all_superior,
        "service_metrics_complete": service_complete,
        "all_baselines_sla_nonregression": sla_ok,
        "publication_superiority_ready": (
            all_superior and sla_ok and service_complete
            and backend != "toy-gym-smoke" and strict_receipt
        ),
        "limitations": [
            "The repository cannot independently attest the external simulator backend.",
            "Gym.step feedback is simulator reward, not real-network performance.",
            "Comparability of COMMAG-trained scheduler/PRB actions must be demonstrated by adapter documentation.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(output_dir / "ran_gym_episodes.csv", index=False)
    steps_df.to_csv(output_dir / "ran_gym_steps.csv.gz", index=False, compression="gzip")
    (output_dir / "ran_gym_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if require_superiority and not report["publication_superiority_ready"]:
        raise RuntimeError("Active RAN Gym superiority NOT demonstrated: inspect ran_gym_summary.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched-seed RAN Gym online-action benchmark")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-superiority", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output, require_superiority=args.require_superiority),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
