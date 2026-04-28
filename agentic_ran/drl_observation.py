from __future__ import annotations

import numpy as np
import torch
from torch import nn


class SliceSequenceEncoder(nn.Module):
    def __init__(self, num_features: int, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape [B, N, F]
        encoded = self.encoder(x)
        return encoded.mean(dim=1)


def build_drl_observation(
    encoder: SliceSequenceEncoder,
    slice_window: np.ndarray,
    current_slice_prb: float,
    current_scheduling_policy: float,
    previous_action: float,
    ratio_granted_req: float,
    predicted_next_throughput: float,
    predicted_peak_risk: float,
    predicted_buffer_pressure: float,
) -> np.ndarray:
    with torch.no_grad():
        x = torch.tensor(slice_window[None, ...], dtype=torch.float32)
        latent = encoder(x).squeeze(0).cpu().numpy().astype(np.float32)
    op_state = np.asarray(
        [
            current_slice_prb,
            current_scheduling_policy,
            previous_action,
            ratio_granted_req,
            predicted_next_throughput,
            predicted_peak_risk,
            predicted_buffer_pressure,
        ],
        dtype=np.float32,
    )
    return np.concatenate([latent, op_state], axis=0).astype(np.float32)
