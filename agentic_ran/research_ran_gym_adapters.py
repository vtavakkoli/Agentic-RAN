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
    run_dir = Path(config.get("run_root", "/workspace/results/ran-gym-native")) / scenario / str(seed) / str(config.get("policy_label", "unknown-policy"))
    run_dir.mkdir(parents=True, exist_ok=True)
    return TrafficSteeringEnv(
        ns3_path=str(ns3_path),
        scenario_configuration=scenario_configuration,
        output_folder=str(run_dir),
        optimized=bool(config.get("optimized", True)),
        verbose=False,
    )
