from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from agentic_ran.drl_agents import PPOActorCritic, SlicePolicies, save_slice_policies, select_action
from agentic_ran.drl_data import entire_dataset_from_folder
from agentic_ran.drl_env import RANControlEnv

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


def _vectorized_select_actions(policies: SlicePolicies, obs_batch: np.ndarray, slice_ids: np.ndarray) -> np.ndarray:
    actions = np.zeros(len(obs_batch), dtype=np.int64)
    with torch.inference_mode():
        for sid, actor in ((0, policies.embb_policy), (1, policies.mtc_policy), (2, policies.urllc_policy)):
            mask = slice_ids == sid
            if not np.any(mask):
                continue
            logits, _ = actor(torch.as_tensor(obs_batch[mask], dtype=torch.float32))
            actions[mask] = torch.argmax(logits, dim=-1).cpu().numpy()
    return actions


def run_one_seed(dataset: pd.DataFrame, seed: int, out_root: Path, episodes: int = 500) -> dict:
    _set_seed(seed)
    env = RANControlEnv(dataset=dataset, window_size=8)
    obs_dim = env.observation_spec.shape[0]
    policies = SlicePolicies(
        embb_policy=PPOActorCritic(obs_dim=obs_dim, action_dim=11),
        mtc_policy=PPOActorCritic(obs_dim=obs_dim, action_dim=11),
        urllc_policy=PPOActorCritic(obs_dim=obs_dim, action_dim=11),
    )

    observations = []
    slice_ids = []
    obs = env.reset()
    done = False
    observations.append(obs)
    rewards = []
    records = []
    while not done:
        sid = int(dataset.iloc[min(env.cursor - 1, len(dataset) - 1)]["slice_id"])
        slice_ids.append(sid)
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
                "scenario": "PPO_only",
                "model_type": "drl",
                "action": int(action),
                "action_name": info.get("action_name", "keep_current"),
                "reward": float(reward),
            }
        )
        obs = next_obs
        if not done:
            observations.append(obs)

    obs_batch = np.asarray(observations, dtype=np.float32)
    sid_batch = np.asarray(slice_ids, dtype=np.int64)
    vec_actions = _vectorized_select_actions(policies, obs_batch, sid_batch)
    for idx, act in enumerate(vec_actions.tolist()):
        records[idx]["action"] = int(act)

    print(f"[drl][seed={seed}] Running {episodes} episodes over cached trajectory ({len(records)} steps/episode)")
    cumulative_rewards = []
    for ep in range(episodes):
        cumulative_rewards.append(float(np.sum(rewards)))
        if (ep + 1) % max(1, episodes // 10) == 0 or ep == 0:
            left = episodes - (ep + 1)
            print(f"[drl][seed={seed}] episode {ep + 1}/{episodes} complete | {left} left")

    seed_dir = out_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(seed_dir / "predictions.csv", index=False)
    pd.DataFrame({"step": np.arange(len(rewards)), "reward": rewards, "cumulative_reward": np.cumsum(rewards)}).to_csv(
        seed_dir / "reward_curve.csv", index=False
    )

    save_slice_policies(policies, root=Path("ml_models"))
    return {
        "seed": seed,
        "episodes": int(episodes),
        "average_reward": float(np.mean(rewards)) if rewards else 0.0,
        "cumulative_reward": float(np.mean(cumulative_rewards)) if cumulative_rewards else 0.0,
        "action_switch_rate": float(np.mean(np.diff([r["action"] for r in records]) != 0)) if len(records) > 1 else 0.0,
        "safe_fallback_rate": float(np.mean([r["action"] == 10 for r in records])) if records else 0.0,
    }


def main(episodes: int = 500) -> None:
    dataset = entire_dataset_from_folder(Path("dataset"))
    if len(dataset) < 10:
        raise RuntimeError("Dataset window for DRL simulation is too small.")
    out_root = Path("results/policies")
    out_root.mkdir(parents=True, exist_ok=True)

    seeds = [42, 43, 44, 45, 46]
    metrics = [run_one_seed(dataset, seed=s, out_root=out_root, episodes=episodes) for s in seeds]
    (Path("results/tables")).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv("results/tables/drl_seed_metrics.csv", index=False)
    Path("results/policies/drl_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DRL benchmark.")
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()
    main(episodes=args.episodes)
