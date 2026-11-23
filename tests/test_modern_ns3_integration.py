"""Native ns-3.49 integration: no mock packets or synthetic environment."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from agentic_ran.research_modern_ns3_adapter import (
    SCENARIO, VERIFIED_NS3_GYM_SHA, VERIFIED_NS3_SHA,
    VerifiedNs3UdpEnv, make_verified_ns3_udp_env, safe_udp_action,
)
from agentic_ran.research_modern_ns3_container_smoke import run


def test_unsafe_udp_count_has_an_explicit_safe_no_send_fallback():
    assert safe_udp_action(7) == (0, True)
    assert safe_udp_action(-1) == (0, True)
    assert safe_udp_action(5) == (5, False)
    with pytest.raises(TypeError, match="PRB"):
        safe_udp_action({"prb": [1, 2]})
    with pytest.raises(ValueError):
        safe_udp_action(4, packet_budget=8)


def test_rejects_silent_prb_and_oran_semantics():
    with pytest.raises(NotImplementedError, match="PRB"):
        VerifiedNs3UdpEnv(scenario="commag-prb")
    with pytest.raises(NotImplementedError, match="PRB"):
        make_verified_ns3_udp_env(SCENARIO, 2026, {"use_case": "commag-prb"})


@pytest.mark.skipif(
    not os.environ.get("VTNS3GYM_BINARY"), reason="Only actual compiled ns-3 can run this integration test"
)
def test_real_native_closed_loop_receipts_and_episode_reproducibility(tmp_path):
    output = tmp_path / "real.json"
    result = run(output, horizon=5)
    assert result["actual_native_episodes"] == 3
    assert result["actual_native_steps"] == 15
    assert result["action_receipts_verified"] is True
    assert result["safety_fallback_checked"] is True
    assert result["commag_prb_controller_evaluated"] is False
    assert result["publication_superiority_ready"] is False
    assert result["ns3_sha"] == VERIFIED_NS3_SHA
    assert result["ns3_gym_sha"] == VERIFIED_NS3_GYM_SHA
    assert len(result["episodes"]) == 3
    assert output.exists()
    for episode in result["episodes"]:
        trace = episode["trace"]
        assert all(t["kpis"]["backend"] == "real-ns3" for t in trace)
        assert all(t["kpis"]["action_receipt_verified"] for t in trace)
        assert all(np.isfinite(t["reward"]) for t in trace)
        assert trace[-1]["kpis"]["rx_packets"] > 0
        assert trace[-1]["kpis"]["tx_packets"] == sum(t["applied_action"] for t in trace)
    assert json.loads(output.read_text())["actual_native_steps"] == 15


@pytest.mark.skipif(
    not os.environ.get("VTNS3GYM_BINARY"), reason="Only actual compiled ns-3 can run this integration test"
)
def test_real_action_mismatch_is_rejected_and_cleanup(tmp_path):
    with VerifiedNs3UdpEnv(horizon=2) as env:
        observation, info = env.reset(seed=42)
        assert info["backend"] == "real-ns3"
        assert env.action_space.n == 6
        assert env.observation_space.contains(observation)
        with pytest.raises(ValueError, match="0..5"):
            env.step(6)
        obs, reward, terminated, truncated, info = env.step(5)
        assert not terminated and not truncated
        assert reward == 5
        assert info["rx_packets"] == 5
        assert info["applied_action"] == 5
        assert info["source"] == "actual-ns3-udp-socket-receiver"
    assert env._env.simulator_pid is None
