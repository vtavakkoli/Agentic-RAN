"""Fail-closed evidence gate for final Agentic-RAN paper experiments.

Passing code tests is NOT equivalent to demonstrating controller superiority.
This verifies machine-readable artifacts only; manual scientific review remains.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def validate(offline: dict[str, Any], active: dict[str, Any], *,
             min_episodes_per_policy: int = 20) -> dict[str, Any]:
    tests: dict[str, bool] = {
        "COMMAG experiment completed": offline.get("run_status") == "EXPERIMENT-COMPLETE",
        "COMMAG internal evidence gates passed": offline.get("evidence_status") == "READY-FOR-MANUSCRIPT",
        "COMMAG seen and unseen tests exist": (
            int(offline.get("seen_test_rows", 0)) > 0
            and int(offline.get("unseen_test_rows", 0)) > 0
        ),
        "RAN Gym is an external backend": active.get("backend") in ("ns-o-ran-gym", "custom-external-gym"),
        "RAN Gym executed step transitions": active.get("executed_gym_step") is True,
        "RAN Gym is not a toy smoke": active.get("is_toy_smoke") is False,
        "Matched-action execution receipts enforced": active.get("receipt_enforced") is True,
        "Observation/action mapping declared": active.get("observation_action_mapping_documented") is True,
        "External simulator revision recorded": bool(active.get("simulator_revision")),
        "Frozen candidate model SHA-256 recorded": (
            isinstance(active.get("frozen_model_sha256"), str)
            and len(active["frozen_model_sha256"]) == 64
        ),
    }
    scenarios = active.get("heldout_scenarios", [])
    seeds = active.get("heldout_seeds", [])
    policies = active.get("compared_methods", [])
    tests["Heldout matched scenarios/seeds adequate"] = (
        isinstance(scenarios, list) and isinstance(seeds, list)
        and len(scenarios) >= 2 and len(seeds) >= 10
        and len(scenarios) * len(seeds) >= min_episodes_per_policy
        and len(set(seeds)) == len(seeds)
    )
    tests["Agentic candidate and two baselines"] = (
        isinstance(policies, list) and "candidate" in policies and len(policies) >= 3
    )
    comparisons = active.get("comparisons", {})
    tests["All baselines have paired statistical results"] = (
        isinstance(comparisons, dict)
        and isinstance(policies, list)
        and set(comparisons) == set(policies) - {"candidate"}
        and all(
            isinstance(v.get("paired_episodes"), int)
            and v["paired_episodes"] >= min_episodes_per_policy
            and isinstance(v.get("holm_adjusted_p"), (int, float))
            and math.isfinite(v["holm_adjusted_p"])
            and v.get("significant_gain") is True
            and isinstance(v.get("bootstrap_ci95"), list)
            and len(v["bootstrap_ci95"]) == 2
            and v["bootstrap_ci95"][0] > 0
            for v in comparisons.values()
        )
    )
    tests["All comparisons beat baseline on realized reward"] = (
        active.get("all_baselines_significantly_outperformed") is True
    )
    # Prevent a simulator with no measured service metrics from being
    # certified by *reward-only* outcomes. Information must exist per episode.
    tests["Per-episode throughput, latency, SLA evidence present"] = (
        active.get("service_metrics_complete") is True
    )
    tests["SLA non-regression established on paired episodes"] = (
        active.get("all_baselines_sla_nonregression") is True
    )
    return {
        "schema": "agentic-ran-publication-evidence-gate-v1",
        "gate_passed": all(tests.values()),
        "checks": tests,
        "manual_review_still_required": True,
        "claim_restriction": "Pass cannot independently certify simulator fidelity, model novelty, or real-network performance.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", type=Path, required=True)
    parser.add_argument("--active", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        offline = json.loads(args.offline.read_text(encoding="utf-8"))
        active = json.loads(args.active.read_text(encoding="utf-8"))
        report = validate(offline, active)
    except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
        report = {"gate_passed": False, "error": str(exc), "manual_review_still_required": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
