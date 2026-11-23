"""Mandatory real ns-3.49 network integration; no synthetic simulation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agentic_ran.research_modern_ns3_adapter import (
    SCENARIO, VERIFIED_NS3_GYM_SHA, VERIFIED_NS3_SHA,
    VerifiedNs3UdpEnv, make_verified_ns3_udp_policy, safe_udp_action,
)


def run(output: Path, *, horizon: int = 5) -> dict[str, Any]:
    if not 1 <= horizon <= 50:
        raise ValueError("horizon must be 1..50")
    episodes: list[dict[str, Any]] = []
    for seed in (2026, 2026, 42):
        with VerifiedNs3UdpEnv(horizon=horizon) as env:
            policy = make_verified_ns3_udp_policy({}, env.action_space)
            observation, info = env.reset(seed=seed)
            trace = []
            previous = None
            for step in range(horizon):
                requested = policy(observation, previous, step, info)
                applied, used_fallback = safe_udp_action(requested)
                new_obs, reward, terminated, truncated, info = env.step(applied)
                if used_fallback or not info["action_receipt_verified"]:
                    raise RuntimeError("Unverified real applied-action receipt")
                if not env.observation_space.contains(new_obs):
                    raise RuntimeError("Invalid real ns-3 observation")
                trace.append({
                    "step": step,
                    "requested_action": requested,
                    "applied_action": applied,
                    "observation": new_obs.tolist(),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "kpis": info,
                })
                previous, observation = applied, new_obs
                if terminated or truncated:
                    break
            if len(trace) != horizon or not trace[-1]["terminated"]:
                raise RuntimeError("Native ns-3 failed to execute complete episode")
            if trace[-1]["kpis"]["rx_packets"] <= 0:
                raise RuntimeError("Native ns-3 reported no delivered UDP packets")
            if int(trace[-1]["kpis"]["tx_packets"]) != sum(t["applied_action"] for t in trace):
                raise RuntimeError("Native action count does not match transmitted packet receipts")
            episodes.append({"seed": seed, "trace": trace, "final_kpis": info})
    if episodes[0]["trace"] != episodes[1]["trace"]:
        raise RuntimeError("Repeated fixed-seed native ns-3 episodes are not deterministic")
    if safe_udp_action(999) != (0, True):
        raise RuntimeError("Safe no-send fallback is not enforced")
    report = {
        "backend": "verified-ns3-3.49-udp-real",
        "scenario": SCENARIO,
        "ns3_sha": VERIFIED_NS3_SHA,
        "ns3_gym_sha": VERIFIED_NS3_GYM_SHA,
        "actual_native_episodes": len(episodes),
        "actual_native_steps": sum(len(ep["trace"]) for ep in episodes),
        "action_receipts_verified": True,
        "safety_fallback_checked": True,
        "commag_prb_controller_evaluated": False,
        "publication_superiority_ready": False,
        "episodes": episodes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(
        "/workspace/results/modern-ns3/real_closed_loop.json"
    ))
    parser.add_argument("--horizon", type=int, default=5)
    args = parser.parse_args()
    report = run(args.output, horizon=args.horizon)
    print(json.dumps({
        "backend": report["backend"],
        "actual_native_episodes": report["actual_native_episodes"],
        "actual_native_steps": report["actual_native_steps"],
        "first_episode_final_kpis": report["episodes"][0]["final_kpis"],
        "publication_superiority_ready": report["publication_superiority_ready"],
    }, indent=2))


if __name__ == "__main__":
    main()
