from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from agentic_ran.drl_agents import PPOActorCritic, SlicePolicies, save_slice_policies, select_action
from agentic_ran.drl_data import entire_dataset_from_folder
from agentic_ran.drl_env import RANControlEnv

SIMULATION_DURATION_SECONDS = 50 * 60


def _limit_to_duration(dataset: pd.DataFrame, duration_seconds: int = SIMULATION_DURATION_SECONDS) -> pd.DataFrame:
    if dataset.empty:
        return dataset
    work = dataset.copy()
    work["Timestamp"] = pd.to_datetime(work["Timestamp"], errors="coerce")
    work = work.dropna(subset=["Timestamp"]).sort_values(["Timestamp", "source_file"], kind="stable").reset_index(drop=True)
    if work.empty:
        return work
    t0 = work["Timestamp"].iloc[0]
    mask = (work["Timestamp"] - t0).dt.total_seconds() <= float(duration_seconds)
    limited = work.loc[mask].copy()
    return limited.reset_index(drop=True)


def _policy_from_action(action: int) -> str:
    if action in {1, 10}:
        return "RR"
    if action == 2:
        return "WF"
    return "PF"


def _temperature_proxy(row: pd.Series) -> float:
    buffer_pressure = float(row.get("dl_buffer [bytes]", 0.0)) + float(row.get("ul_buffer [bytes]", 0.0))
    error_pressure = float(row.get("tx_errors downlink (%)", 0.0)) + float(row.get("rx_errors uplink (%)", 0.0))
    return float(25.0 + 0.15 * buffer_pressure + 0.05 * error_pressure)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _policy_for_slice(policies: SlicePolicies, sid: int) -> PPOActorCritic:
    if sid == 0:
        return policies.embb_policy
    if sid == 1:
        return policies.mtc_policy
    return policies.urllc_policy


def run_one_seed(dataset: pd.DataFrame, seed: int, out_root: Path) -> dict:
    _set_seed(seed)
    env = RANControlEnv(dataset=dataset, window_size=8)
    obs_dim = env.observation_spec.shape[0]
    policies = SlicePolicies(
        embb_policy=PPOActorCritic(obs_dim=obs_dim, action_dim=11),
        mtc_policy=PPOActorCritic(obs_dim=obs_dim, action_dim=11),
        urllc_policy=PPOActorCritic(obs_dim=obs_dim, action_dim=11),
    )

    obs = env.reset()
    done = False
    rewards = []
    records = []
    while not done:
        sid = int(dataset.iloc[min(env.cursor - 1, len(dataset) - 1)]["slice_id"])
        actor = _policy_for_slice(policies, sid)
        action = select_action(actor, obs)
        next_obs, reward, done, info = env.step(action)
        rewards.append(reward)
        row = dataset.iloc[min(env.cursor - 1, len(dataset) - 1)]
        records.append(
            {
                "sample_id": int(row.get("sample_id", 0)),
                "global_index": int(row.get("global_index", 0)),
                "Timestamp": str(row.get("Timestamp", "")),
                "source_file": row.get("source_file", "unknown"),
                "y_true": float(row.get("tx_brate downlink [Mbps]", 0.0)),
                "y_pred": float(row.get("tx_brate downlink [Mbps]", 0.0)),
                "scenario": "base_case",
                "model_type": "drl",
                "action": int(action),
                "action_name": info.get("action_name", "keep_current"),
                "policy_name": _policy_from_action(action),
                "temperature_c": _temperature_proxy(row),
                "reward": float(reward),
            }
        )
        obs = next_obs

    seed_dir = out_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(seed_dir / "predictions.csv", index=False)
    pd.DataFrame({"step": np.arange(len(rewards)), "reward": rewards, "cumulative_reward": np.cumsum(rewards)}).to_csv(
        seed_dir / "reward_curve.csv", index=False
    )

    save_slice_policies(policies, root=Path("ml_models"))
    return {
        "seed": seed,
        "average_reward": float(np.mean(rewards)) if rewards else 0.0,
        "cumulative_reward": float(np.sum(rewards)),
        "action_switch_rate": float(np.mean(np.diff([r["action"] for r in records]) != 0)) if len(records) > 1 else 0.0,
        "safe_fallback_rate": float(np.mean([r["action"] == 10 for r in records])) if records else 0.0,
    }


def main() -> None:
    dataset = entire_dataset_from_folder(Path("dataset"))
    dataset = _limit_to_duration(dataset, duration_seconds=SIMULATION_DURATION_SECONDS)
    if len(dataset) < 10:
        raise RuntimeError("Dataset window for DRL simulation is too small after applying 50-minute duration cap.")
    out_root = Path("results/policies")
    out_root.mkdir(parents=True, exist_ok=True)

    seeds = [42, 43, 44, 45, 46]
    metrics = [run_one_seed(dataset, seed=s, out_root=out_root) for s in seeds]
    (Path("results/tables")).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv("results/tables/drl_seed_metrics.csv", index=False)
    Path("results/policies/drl_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
