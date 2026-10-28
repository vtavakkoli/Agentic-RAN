"""Episode-level SLA residual diagnostics for selective offline policies.

The SLA critic predicts a continuous [0,1] loss. Pointwise maxima of
hundreds of dependent observations per episode do not provide an actionable
upper bound on *expected episode-average loss*. This module reports both
estimands separately, and calibrates only episode-mean residual bias for
the offline research gate. No counterfactual action safety is certified.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EpisodeSLACalibration:
    mean_loss_bias_margin: float
    worst_step_residual_margin: float
    episode_count: int
    row_count: int
    confidence: float
    feasible: bool

    def as_dict(self) -> dict:
        return {
            "mean_loss_bias_margin": self.mean_loss_bias_margin,
            "worst_step_residual_margin": self.worst_step_residual_margin,
            "episode_count": self.episode_count,
            "row_count": self.row_count,
            "confidence": self.confidence,
            "feasible": self.feasible,
            "estimand": "logged_action_episode_mean_residual",
            "not_a_bound_on": [
                "single_step_sla_cost", "counterfactual_action_risk",
                "real_network_sla_violation_probability",
            ],
        }


def calibrate_episode_sla(
    residuals: np.ndarray,
    episode_ids: np.ndarray,
    *,
    confidence: float = 0.95,
    min_episodes: int = 19,
) -> EpisodeSLACalibration:
    """Episode-blocked upper residual calibration without row pseudo-replication.

    Calibrate the 95th quantile of episode-*mean* logged-action residuals
    (not their maximum observation). Diagnostic worst-step maxima are kept
    separate and must not silently enter the mean-risk controller.
    Requires distinct independent episodes. If too few, returns infinite
    margin and the candidate must decline actuation.
    """
    values = np.asarray(residuals, dtype=float)
    episodes = np.asarray(episode_ids, dtype=str)
    if values.ndim != 1 or episodes.ndim != 1 or len(values) != len(episodes):
        raise ValueError("residuals and episode identifiers must be aligned vectors")
    if not 0 < confidence < 1 or min_episodes < 2:
        raise ValueError("invalid calibration confidence or minimum episodes")
    finite = np.isfinite(values)
    values, episodes = values[finite], episodes[finite]
    unique, inverse = np.unique(episodes, return_inverse=True)
    n = len(unique)
    if n == 0:
        return EpisodeSLACalibration(float("inf"), float("inf"), 0, 0,
                                     confidence, False)
    episode_sums = np.bincount(inverse, weights=values, minlength=n)
    counts = np.bincount(inverse, minlength=n)
    episode_means = episode_sums / counts
    episode_max = np.full(n, -np.inf)
    np.maximum.at(episode_max, inverse, values)
    if n < max(min_episodes, int(np.ceil(1 / (1 - confidence) - 1))):
        return EpisodeSLACalibration(
            float("inf"), float("inf"), n, len(values), confidence, False,
        )
    level = min(1.0, float(np.ceil((n + 1) * confidence) / n))
    margin = max(0.0, float(np.quantile(episode_means, level, method="higher")))
    worst = max(0.0, float(np.quantile(episode_max, level, method="higher")))
    return EpisodeSLACalibration(margin, worst, n, len(values),
                                 confidence, True)
