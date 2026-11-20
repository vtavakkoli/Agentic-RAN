"""Smoke-test the real upstream ns-O-RAN Gym and compiled ns-3 in one image.

The stock environment implements TRAFFIC STEERING (handover) actions, not
COMMAG scheduler/PRB actions; even a successful step is not an Agentic-RAN
performance result. Records backend provenance and one or more realized steps.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from agentic_ran.research_ran_gym_adapters import make_nsoran_traffic_steering


def run(output: Path, steps: int = 2) -> dict:
    config = {
        "ns3_path": "/opt/ns3",
        "scenario_configuration": "/workspace/configs/nsoran_ts_container.json",
        "run_root": "/workspace/results/nsoran-native",
        "optimized": True,
        "skip_ns3_rebuild": True,
        "policy_label": "stock-traffic-steering-no-op",
    }
    env = make_nsoran_traffic_steering("scenario-one", 2026, config)
    try:
        started = time.monotonic()
        obs, info = env.reset(seed=2026)
        if not isinstance(info, dict):
            raise ValueError("native Gym reset must return a dict for info")
        step_rewards = []
        for index in range(steps):
            # In the upstream TrafficSteeringEnv, zero explicitly means
            # NO handover for every UE, not a scheduler or PRB action.
            action = np.zeros(env.action_space.shape, dtype=np.int64)
            assert env.action_space.contains(action)
            obs, reward, terminated, truncated, info = env.step(action)
            if not np.isfinite(float(reward)):
                raise ValueError("native simulator returned a nonfinite reward")
            step_rewards.append(float(reward))
            print(f"[ns-O-RAN native] t={index} realized_reward={reward}", flush=True)
            if terminated or truncated:
                break
        if not step_rewards:
            raise RuntimeError("native simulator supplied no step observations")
        report = {
            "backend": "stock-ns-o-ran-gym",
            "test": "traffic-steering-no-op",
            "simulator": "compiled_ns3_inside_same_container",
            "actual_gym_steps": len(step_rewards),
            "observed_rewards": step_rewards,
            "initial_seed": 2026,
            "backend_identity_independently_verified": False,
            "agentic_ran_prb_controller_evaluated": False,
            "publication_superiority_ready": False,
            "applied_action_receipts_verified": False,
            "elapsed_seconds": time.monotonic() - started,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(
        "/workspace/results/nsoran-native/native_gym_smoke.json"
    ))
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.steps <= 50:
        parser.error("--steps must be between 1 and 50")
    print(json.dumps(run(args.output, args.steps), indent=2))


if __name__ == "__main__":
    main()
