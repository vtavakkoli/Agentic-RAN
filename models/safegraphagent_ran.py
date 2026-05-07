from __future__ import annotations
import torch
from torch import nn
from models.graph_actor_critic_ran import GraphEncoder
from policies.safe_policy_layer import SafePolicyLayer


class SafeGraphAgentRAN(nn.Module):
    full_name = "SafeGraphAgent-RAN: Graph-Augmented Safe Actor-Critic Control with Time-Aware Residual MLP Forecasting"
    def __init__(self, in_dim: int, hidden: int = 64, num_actions: int = 9):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.graph = GraphEncoder(hidden, hidden)
        self.actor = nn.Linear(hidden, num_actions)
        self.critic = nn.Linear(hidden, 1)
        self.safe = SafePolicyLayer()
