from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, obs: torch.Tensor):
        h = self.backbone(obs)
        return self.actor(h), self.critic(h).squeeze(-1)


class DQNPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, action_dim))

    def forward(self, obs: torch.Tensor):
        return self.net(obs)


@dataclass
class SlicePolicies:
    embb_policy: PPOActorCritic
    mtc_policy: PPOActorCritic
    urllc_policy: PPOActorCritic


def select_action(policy: PPOActorCritic, obs: np.ndarray) -> int:
    with torch.no_grad():
        logits, _ = policy(torch.tensor(obs[None, :], dtype=torch.float32))
    return int(torch.argmax(logits, dim=-1).item())


def save_slice_policies(policies: SlicePolicies, root: Path) -> None:
    mapping = {
        "embb_policy": policies.embb_policy,
        "mtc_policy": policies.mtc_policy,
        "urllc_policy": policies.urllc_policy,
    }
    for name, model in mapping.items():
        out = root / name
        out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out / "policy.pt")


def load_slice_policy(path: Path, obs_dim: int, action_dim: int) -> PPOActorCritic:
    model = PPOActorCritic(obs_dim=obs_dim, action_dim=action_dim)
    model.load_state_dict(torch.load(path / "policy.pt", map_location="cpu"))
    model.eval()
    return model
