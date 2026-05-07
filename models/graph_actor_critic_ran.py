from __future__ import annotations
import torch
from torch import nn


class GraphEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.lin = nn.Linear(in_dim, hidden_dim)

    def forward(self, node_features: torch.Tensor, adjacency_matrix: torch.Tensor | None = None):
        h = self.lin(node_features)
        if adjacency_matrix is not None:
            deg = adjacency_matrix.sum(-1, keepdim=True).clamp_min(1.0)
            h = h + adjacency_matrix @ h / deg
        return torch.relu(h)


class ActorHead(nn.Module):
    def __init__(self, hidden: int, num_actions: int = 9):
        super().__init__(); self.fc = nn.Linear(hidden, num_actions)
    def forward(self, h): return self.fc(h)


class CriticHead(nn.Module):
    def __init__(self, hidden: int):
        super().__init__(); self.fc = nn.Linear(hidden, 1)
    def forward(self, h): return self.fc(h).squeeze(-1)


class GraphActorCriticRAN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, num_actions: int = 9):
        super().__init__()
        self.encoder = GraphEncoder(in_dim, hidden)
        self.actor = ActorHead(hidden, num_actions)
        self.critic = CriticHead(hidden)

    def forward(self, node_features, edge_index=None, adjacency_matrix=None, global_features=None, allowed_action_mask=None):
        h = self.encoder(node_features, adjacency_matrix)
        pooled = h.mean(dim=0)
        logits = self.actor(pooled)
        value = self.critic(pooled)
        if allowed_action_mask is not None:
            logits = logits.masked_fill(~allowed_action_mask.bool(), -1e9)
        probs = torch.softmax(logits, dim=-1)
        action = int(torch.argmax(probs).item())
        confidence = float(torch.max(probs).item())
        return {"action_logits": logits, "value_estimate": value, "decision_confidence": confidence, "selected_action": action}
