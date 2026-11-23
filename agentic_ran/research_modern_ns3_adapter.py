"""Verified, *real ns-3* UDP adapter; explicitly NOT a COMMAG PRB adapter.

The wire/library origin is pinned vtavakkoli/ns3-gym commit
21e68565a6beaf13589fd19778b5332f4d0b29e0 (ns-3.49).
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any

import numpy as np

VERIFIED_NS3_GYM_SHA = "21e68565a6beaf13589fd19778b5332f4d0b29e0"
VERIFIED_NS3_SHA = "418210e8914e9f6bcb5cddbcc94a2cecb7853b68"
SCENARIO = "two-node-udp-packets"


def safe_udp_action(requested: Any, *, packet_budget: int = 5) -> tuple[int, bool]:
    """Explicit safety no-send fallback for an integer UDP packet count.

    Never reinterprets COMMAG PRB/scheduling or handover controls as packets.
    """
    if isinstance(requested, bool) or not isinstance(requested, Integral):
        raise TypeError("Only integer UDP packet counts supported; PRB/scheduler actions are unsupported")
    if packet_budget != 5:
        raise ValueError("The compiled ns-3 example supports packet_budget=5 only")
    if requested < 0 or requested > packet_budget:
        return 0, True
    return int(requested), False


class VerifiedNs3UdpEnv:
    """Validate simulator observations and actual transmitted-action receipts."""

    def __init__(self, *, scenario: str = SCENARIO, horizon: int = 5, timeout_seconds: float = 15.0):
        if scenario != SCENARIO:
            raise NotImplementedError(
                "Only the independently verified ns-3 UDP packet scenario is supported. "
                "NR PRB/scheduler and O-RAN E2 control require separate native implementations."
            )
        from vtns3gym import Ns3GymEnv

        self._env = Ns3GymEnv(horizon=horizon, timeout_seconds=timeout_seconds)
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self._last_transmitted: int | None = None

    def reset(self, *, seed: int | None = None, options: Mapping[str, Any] | None = None):
        observation, info = self._env.reset(seed=seed, options=dict(options) if options else None)
        if info.get("backend") != "real-ns3" or info.get("ns3_release") != "3.49":
            raise RuntimeError("Unverified backend identity: refusing to treat results as ns-3")
        if tuple(observation.shape) != (3,):
            raise RuntimeError("Unexpected native observation shape")
        self._last_transmitted = int(info["tx_packets"])
        return observation, info

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError("Native ns-3 action requires packet count 0..5; no implicit translation")
        observation, reward, terminated, truncated, info = self._env.step(int(action))
        transmitted = int(info["tx_packets"])
        if self._last_transmitted is None:
            raise RuntimeError("Cannot verify applied action without reset receipt")
        delta = transmitted - self._last_transmitted
        if delta != int(action):
            raise RuntimeError(
                f"Native ns-3 action receipt mismatch: requested {action}, applied {delta}"
            )
        self._last_transmitted = transmitted
        if int(info["rx_packets"]) != int(observation[0]):
            raise RuntimeError("Native ns-3 received-packet observation disagrees with KPI")
        if not np.isfinite(float(reward)):
            raise RuntimeError("Nonfinite native ns-3 reward")
        info = {
            **info,
            "applied_action": int(delta),
            "action_receipt_verified": True,
            "source": "actual-ns3-udp-socket-receiver",
            "ns3_sha": VERIFIED_NS3_SHA,
            "ns3_gym_sha": VERIFIED_NS3_GYM_SHA,
            "commag_prb_supported": False,
        }
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self._env.close()

    def __enter__(self) -> "VerifiedNs3UdpEnv":
        return self

    def __exit__(self, *args: Any) -> bool:
        self.close()
        return False


def make_verified_ns3_udp_env(scenario: str, seed: int, config: dict[str, Any]) -> VerifiedNs3UdpEnv:
    """Explicit factory for research benchmark; it cannot be used as a PRB simulator."""
    if config.get("use_case", "udp-packet-count") != "udp-packet-count":
        raise NotImplementedError("No verified COMMAG PRB or O-RAN E2 action backend exists")
    return VerifiedNs3UdpEnv(
        scenario=scenario,
        horizon=int(config.get("max_steps", 5)),
        timeout_seconds=float(config.get("timeout_seconds", 15.0)),
    )


def make_verified_ns3_udp_policy(config: dict[str, Any], action_space: Any):
    """Deterministic observation-dependent *UDP example* policy, not a RAN controller."""
    if getattr(action_space, "n", None) != 6:
        raise ValueError("Expected verified Discrete(6) UDP packet-count action space")

    def policy(observation: Any, previous: Any, step: int, info: dict) -> int:
        raw = np.asarray(observation, dtype=float)
        if raw.shape != (3,) or not np.isfinite(raw).all():
            raise ValueError("Invalid real ns-3 UDP observation")
        if int(raw[0]) > int(raw[1]):
            raise RuntimeError("Native received more packets than transmitted")
        requested = 3 if raw[0] < 6 else 1
        applied, used_fallback = safe_udp_action(requested)
        if used_fallback:
            raise RuntimeError("Example policy unexpectedly required fallback")
        return applied

    return policy
