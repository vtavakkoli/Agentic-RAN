"""Adapters for RAN Gym benchmark: explicit toy smoke and ns-O-RAN TS.

The ns-O-RAN TrafficSteeringEnv acts on handovers, not COMMAG scheduler/PRB.
A frozen Agentic-RAN controller MUST NOT be relabeled as a TS controller
without a separately documented and validated observation/action mapping.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from agentic_ran.research_twin_environment import ACTIONS, MultiSliceToyTwin


def make_toy_env(scenario: str, seed: int, config: dict[str, Any]) -> Any:
    """Minimal valid Gymnasium API around the project's action-dependent toy."""
    from gymnasium import spaces

    class TwinGym:
        action_space = spaces.Discrete(len(ACTIONS))

        def __init__(self) -> None:
            self.engine = MultiSliceToyTwin(seed=seed, scenario=scenario)
            self.steps = 0

        def reset(self, *, seed: int | None = None, options: Any = None):
            self.engine = MultiSliceToyTwin(seed=seed if seed is not None else 0, scenario=scenario)
            self.steps = 0
            return self.engine.state.copy(), {}

        def step(self, action: int):
            new_state, reward, flags = self.engine.step(int(action))
            self.steps += 1
            info = {
                "applied_action": int(action),
                "sla_violation_rate": float(flags.mean()),
            }
            return new_state, reward, False, self.steps >= config.get("max_steps", 24), info

        def close(self) -> None:
            pass

    return TwinGym()


def make_toy_queue_policy(config: dict[str, Any], action_space: Any) -> Any:
    """Rule-based *toy* candidate, NOT a COMMAG-trained Agentic-RAN model."""
    if int(action_space.n) != len(ACTIONS):
        raise ValueError("toy policy action mismatch")

    def decide(observation: Any, previous: Any, step: int, info: dict) -> int:
        state = np.asarray(observation, dtype=float)
        if state.shape != (9,) or not np.isfinite(state).all():
            raise ValueError("toy queue policy requires 9 finite values")
        if state[2] > 0.45:
            return 1
        if state[0] > 1.0:
            return 2
        return 0

    return decide


def make_noop_policy(config: dict[str, Any], action_space: Any) -> Any:
    """Use the explicit no-op action only if the environment supports it."""
    import gymnasium as gym

    if not isinstance(action_space, gym.spaces.Discrete):
        raise ValueError("no-op example supports only Discrete Gym action spaces")
    noop = int(config.get("noop_action", 0))
    if not action_space.contains(noop):
        raise ValueError("configured no-op is outside environment action space")
    return lambda observation, previous, step, info: noop


def make_random_policy(config: dict[str, Any], action_space: Any) -> Any:
    """Independent deterministic baseline RNG (not the simulator RNG)."""
    import gymnasium as gym

    if not isinstance(action_space, gym.spaces.Discrete):
        raise ValueError("random example supports only Discrete Gym action spaces")
    rng = np.random.default_rng(int(config["seed"]) + 413)
    return lambda observation, previous, step, info: int(rng.integers(action_space.n)) + int(action_space.start)


def make_nsoran_traffic_steering(scenario: str, seed: int, config: dict[str, Any]) -> Any:
    """Construct actual ns-o-ran-gym TrafficSteeringEnv with external ns-3.

    This does not translate the COMMAG PRB/scheduler policy. Supply a separate
    validated policy factory and action mapping to use it scientifically.
    """
    if scenario != "scenario-one":
        raise ValueError("stock ns-O-RAN TS environment supports scenario-one only")
    from environments.ts_env import TrafficSteeringEnv

    ns3_path = Path(config["ns3_path"])
    scenario_file = Path(config["scenario_configuration"])
    if not ns3_path.is_dir() or not scenario_file.is_file():
        raise FileNotFoundError("ns-O-RAN simulator and scenario configuration must be mounted")
    scenario_configuration = json.loads(scenario_file.read_text(encoding="utf-8"))
    # Gymnasium's reset(seed=...) does not pass ns-3 RngRun to upstream
    # by default. Bind each matched seed to its simulator disturbance stream.
    scenario_configuration["RngRun"] = [int(seed)]
    run_dir = Path(config.get("run_root", "/workspace/results/ran-gym-native")) / scenario / str(seed) / str(config.get("policy_label", "unknown-policy"))
    run_dir.mkdir(parents=True, exist_ok=True)

    class PrebuiltTrafficSteeringEnv(TrafficSteeringEnv):
        def configure_and_build_ns3(self) -> None:
            if not bool(config.get("skip_ns3_rebuild", True)):
                return super().configure_and_build_ns3()
            # Compiled during Docker build. Avoid rebuilding the whole
            # simulator for each independently seeded policy episode.
            # The pinned ns-O-RAN tree is Waf-based; modern ns-3 is CMake.
            import sys

            simulator_root = Path(self.ns3_path)
            if (simulator_root / "ns3").is_file():
                build_marker = simulator_root / f".lock-ns3_{sys.platform}_build"
            else:
                # NsOranEnv itself loads this Waf-generated Python module to
                # resolve the compiled scenario-one executable.
                build_marker = simulator_root / (
                    "build/optimized/build-status.py"
                    if self.optimized else "build/build-status.py"
                )
            if not build_marker.is_file():
                raise FileNotFoundError(
                    f"prebuilt ns-O-RAN status missing: {build_marker}; "
                    "rebuild Docker target nsoran with the pinned Waf toolchain"
                )

    return PrebuiltTrafficSteeringEnv(
        ns3_path=str(ns3_path),
        scenario_configuration=scenario_configuration,
        output_folder=str(run_dir),
        optimized=bool(config.get("optimized", True)),
        verbose=False,
    )


from functools import lru_cache


@lru_cache(maxsize=2)
def _load_trusted_publication_models(path: str, sha256: str) -> Any:
    import joblib
    # The command-line runner checks the sha256 before calling this factory.
    return joblib.load(path)


def make_frozen_commag_agentic_policy(config: dict[str, Any], action_space: Any) -> Any:
    """Evaluate the *actual* frozen cross-fitted COMMAG policy, not a toy heuristic.

    The observation mapper must produce authentic COMMAG-equivalent
    per-slice 17-feature rows and the action mapper must implement actual
    scheduler/PRB actuation on a COMPATIBLE environment. This is deliberately
    incompatible with stock ns-o-ran TrafficSteeringEnv's handover actions.
    The caller and publication authors must validate these mappings.
    """
    from dataclasses import fields, replace
    import pandas as pd

    from agentic_ran.offline_policy import STATE_COLUMNS
    from agentic_ran.publication_science import (
        _choose_from_precomputed, _policy_precompute,
    )
    from agentic_ran.research_ran_gym import load_factory

    if config.get("environment_factory", "").endswith(":make_nsoran_traffic_steering"):
        raise ValueError(
            "COMMAG-trained Agentic-RAN scheduler/PRB policy cannot control "
            "stock ns-O-RAN traffic steering handover action space"
        )
    model_path = Path(config["frozen_model_path"])
    fold_name = str(config.get("policy_fold", ""))
    if not fold_name:
        raise ValueError("select an explicitly frozen policy_fold, e.g. policy_exp1")
    obs_mapper, _ = load_factory(config["observation_mapper"])
    action_mapper, _ = load_factory(config["action_mapper"])
    # Only locally trusted checksum-verified artifacts may be used.
    # Cache the frozen model across repeated matched simulator episodes.
    fitted = _load_trusted_publication_models(
        str(model_path.resolve()), str(config["frozen_model_sha256"])
    )
    if not isinstance(fitted, dict) or not isinstance(fitted.get("folds"), dict):
        raise ValueError("expected cross-fitted publication_models.joblib")
    if fold_name not in fitted["folds"]:
        raise ValueError(f"unknown policy fold: {fold_name}")
    fold = fitted["folds"][fold_name]
    bundle = fold["policy_bundle"]
    calibration = fold["calibrated_controller"]
    permitted = {field.name for field in fields(bundle.config)}
    settings = {name: calibration[name] for name in
                ("sla_threshold", "uncertainty_threshold", "ood_threshold", "planning_weight")
                if name in permitted and name in calibration}
    calibrated = replace(bundle.config, **settings)
    switching_margin = float(calibration["switch_margin"])

    def decide(observation: Any, previous: Any, step: int, info: dict) -> Any:
        frame = obs_mapper(observation, previous, step, info, config)
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError("observation mapper must return a non-empty COMMAG feature DataFrame")
        mandatory = set(STATE_COLUMNS) | {"slice_type", "action", "episode_id", "timestamp_s"}
        if not mandatory.issubset(frame.columns):
            raise ValueError(f"observation mapper omitted COMMAG fields: {sorted(mandatory - set(frame.columns))}")
        if not np.isfinite(frame[STATE_COLUMNS].to_numpy(dtype=float)).all():
            raise ValueError("nonfinite mapped COMMAG observations")
        pre = _policy_precompute(bundle, frame)
        decisions = _choose_from_precomputed(bundle, pre, calibrated, switching_margin)
        chosen = decisions["selected_action"].astype(str).to_list()
        action = action_mapper(chosen, frame, observation, previous, step, info, config)
        if not bool(action_space.contains(action)):
            raise ValueError("physical action mapper produced incompatible Gym action")
        return action

    return decide
