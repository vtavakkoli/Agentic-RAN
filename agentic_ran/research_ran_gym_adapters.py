"""Research Gymnasium adapters: explicit synthetic tests and strict native boundaries.

The verified ns-3.49 UDP backend is a separate module and must never be
misrepresented as a COMMAG PRB/scheduler or O-RAN traffic-steering backend.
"""
from __future__ import annotations

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
    """Legacy Waf/ns-O-RAN fork removed; fail explicitly rather than falling back."""
    raise NotImplementedError(
        "Legacy wineslab/ns-O-RAN Gym is no longer a supported runtime dependency. "
        "Use the independently verified ns-3.49 UDP integration for UDP actions, "
        "or provide a separately validated NR PRB/scheduler simulator. "
        "No silent traffic-steering-to-PRB mapping exists."
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

    if config.get("environment_factory", "").endswith((
        ":make_nsoran_traffic_steering", ":make_verified_ns3_udp_env"
    )):
        raise ValueError(
            "COMMAG-trained Agentic-RAN scheduler/PRB policy cannot control "
            "legacy handover or generic ns-3 UDP packet-count action space"
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
