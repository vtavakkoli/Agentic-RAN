from __future__ import annotations
import torch
from models.graph_actor_critic_ran import GraphActorCriticRAN


class MaskedGraphPPORAN(GraphActorCriticRAN):
    def offline_policy_eval(self, actions, rewards):
        n = max(1, len(actions))
        switches = sum(int(actions[i] != actions[i-1]) for i in range(1, len(actions)))
        return {
            "action_switch_rate": switches / n,
            "safe_fallback_rate": sum(int(a == (max(actions) if actions else 0)) for a in actions) / n,
            "average_reward": float(sum(rewards) / n) if rewards else 0.0,
            "cumulative_reward": float(sum(rewards)) if rewards else 0.0,
            "offline_reward_proxy": float(sum(rewards) / n) if rewards else 0.0,
        }
