"""Regression: episode-blocked risk and feasible action selection.

Does not promise network safety. Tests the specific all-abstain path that
previous generic publication CI tests failed to detect.
"""
from __future__ import annotations

import numpy as np
import pytest

from agentic_ran.research_calibration import calibrate_episode_sla
from agentic_ran.research_support import ConformalActionSupport, rerank


def test_episode_average_does_not_confuse_rare_pointwise_spikes_with_mean_risk():
    # One large instantaneous error per episode should be visible in
    # worst-step diagnostics, but it cannot become a blanket 0.6 margin on
    # every expected episode-average risk prediction.
    episodes = np.repeat([f"episode-{j}" for j in range(40)], 20)
    residuals = np.full(len(episodes), 0.02)
    residuals[::20] = 0.62
    result = calibrate_episode_sla(residuals, episodes)
    assert result.feasible
    assert result.episode_count == 40
    assert 0.0 < result.mean_loss_bias_margin < 0.10
    assert result.worst_step_residual_margin >= 0.60
    assert result.as_dict()["not_a_bound_on"]


def test_insufficient_independent_episodes_fail_closed():
    labels = np.repeat([f"E{i}" for i in range(5)], 100)
    result = calibrate_episode_sla(np.zeros(500), labels)
    assert not result.feasible
    assert np.isinf(result.mean_loss_bias_margin)


def test_calibration_does_not_double_count_intraepisode_rows():
    ids = np.repeat([f"E{i}" for i in range(30)], 10)
    x = np.tile([0.01] * 9 + [0.21], 30)
    doubled = calibrate_episode_sla(np.repeat(x, 3), np.repeat(ids, 3))
    baseline = calibrate_episode_sla(x, ids)
    assert doubled.mean_loss_bias_margin == pytest.approx(baseline.mean_loss_bias_margin)
    assert doubled.episode_count == 30


def test_episode_blocked_conformal_support_can_issue_real_candidate_not_all_abstain():
    rng = np.random.default_rng(39)
    roles = {}
    for role in ("policy", "evaluator"):
        # Independent 80 episodes per role, two observations each. The
        # training/calibration protocol never shares an episode.
        x = rng.normal(0, 0.2, size=(160, 3))
        sl = np.repeat("URLLC", 160)
        action = np.repeat("pf:prb=4", 160)
        eps = np.repeat([f"{role}_ep_{i}" for i in range(80)], 2)
        roles[role] = (x, sl, action, eps)
    support = ConformalActionSupport(
        methods=("gaussian",), min_group=12, max_reference=300,
        seed=39,
    ).fit(roles)
    assert support.calibration_protocol == "held_out_episode_maximum"
    assert all(n >= 19 for n in support.calibration_units.values())
    probe = rng.normal(0, 0.14, (100, 3))
    supported = support.action_support(
        probe, ["URLLC"] * 100, ["pf:prb=4"], "gaussian", alpha=0.20,
    )
    assert supported.sum() >= 20
    assert not support.action_support(
        probe, ["URLLC"] * 100, ["unknown"], "gaussian", alpha=0.2,
    ).any()
    decision = rerank(
        np.full((100, 1), 0.8),
        np.full((100, 1), 0.12),
        np.full((100, 1), 0.10),
        np.ones((100, 1), dtype=bool),
        supported,
        np.zeros(100, dtype=int),
        method="gaussian", alpha=0.20,
        max_sla=0.5, max_uncertainty=0.6,
    )
    assert int((~decision.abstain).sum()) >= 20
