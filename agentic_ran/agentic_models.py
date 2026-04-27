from __future__ import annotations

import torch
from torch import nn

from agentic_ran.residual_models import ResidualBlock


class AgenticResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float, num_actions: int = 10):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([ResidualBlock(hidden_size, dropout=dropout) for _ in range(max(3, num_layers))])
        self.reg_head = nn.Linear(hidden_size, 1)
        self.action_head = nn.Linear(hidden_size, num_actions)

    def forward(self, x: torch.Tensor):
        if x.dim() == 3:
            x = x[:, -1, :]
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        pred = self.reg_head(h).squeeze(-1)
        action_logits = self.action_head(h)
        confidence = torch.softmax(action_logits, dim=-1).max(dim=-1).values
        return pred, action_logits, confidence


class AgenticSequenceModel(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float, num_actions: int = 10):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.proj = nn.Linear(input_dim, hidden_size)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=max(2, num_layers))
        self.norm = nn.LayerNorm(hidden_size)
        self.reg_head = nn.Linear(hidden_size, 1)
        self.action_head = nn.Linear(hidden_size, num_actions)

    def forward(self, x: torch.Tensor):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.encoder(self.proj(x))
        h = self.norm(h[:, -1, :])
        pred = self.reg_head(h).squeeze(-1)
        logits = self.action_head(h)
        conf = torch.softmax(logits, dim=-1).max(dim=-1).values
        return pred, logits, conf


class AgenticLiquidModel(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float, num_actions: int = 10):
        super().__init__()
        self.hidden_size = hidden_size
        self.in_proj = nn.Linear(input_dim, hidden_size)
        self.state_proj = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Linear(hidden_size, hidden_size)
        self.reg_head = nn.Linear(hidden_size, 1)
        self.action_head = nn.Linear(hidden_size, num_actions)

    def forward(self, x: torch.Tensor):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        bsz, seq_len, _ = x.shape
        h = torch.zeros((bsz, self.hidden_size), device=x.device, dtype=x.dtype)
        residual = torch.zeros_like(h)
        for t in range(seq_len):
            inp = torch.tanh(self.in_proj(x[:, t, :]))
            g = torch.sigmoid(self.gate(torch.cat([inp, h], dim=-1)))
            candidate = torch.tanh(inp + self.state_proj(h))
            h = (1 - g) * h + g * candidate
            h = self.norm(h + self.residual(residual))
            h = self.dropout(h)
            residual = h
        pred = self.reg_head(h).squeeze(-1)
        logits = self.action_head(h)
        conf = torch.softmax(logits, dim=-1).max(dim=-1).values
        return pred, logits, conf
