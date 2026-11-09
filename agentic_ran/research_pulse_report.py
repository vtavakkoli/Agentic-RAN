"""Concise, reproducible PULSE-RAN versus offline-RL experiment report.

Reads saved study JSON only; does not retrain or reselect models, and does
not claim the synthetic twin has real wireless-network validity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LABELS = {
    "pulse_ran": "PULSE-RAN",
    "pulse_ran_gated": "PULSE-RAN Gated",
    "twin_greedy": "Twin Greedy",
    "twin_greedy_cvar_gate": "Twin Greedy (CVaR Gate)",
    "twin_cql": "Twin-CQL",
    "distributional_cvar": "Distributional CVaR",
    "twin_cql_cvar_hybrid": "CQL + CVaR Hybrid",
    "heuristic": "PULSE-RAN (legacy)",
}


def format_report(data: dict) -> str:
    if "final_test_metrics" not in data or "scientific_scope" not in data:
        raise ValueError("not a valid twin controller-study JSON")
    lines = [
        "PULSE-RAN | Priority-driven URLLC and Load-Sensitive Engine",
        "SIMULATED INTERVENTIONS ONLY — not COMMAG or a real RAN",
        f"Validation-selected learned family: {data['validation_selected_family']}",
        "All-controller validation ranking: " + ", ".join(
            LABELS.get(k, k) for k in data.get("validation_ranked_all", [])
        ),
    ]
    for split in ("seen", "unseen"):
        table = data["final_test_metrics"][split]
        lines.extend((
            "",
            f"{split.upper()} | realized per-step utility (higher is better)",
            f"{'Controller':<27} {'Utility':>9} {'URLLC SLA':>11} "
            f"{'eMBB SLA':>10} {'Switch':>8} {'Abstain':>9}",
        ))
        for key in sorted(table, key=lambda name: -table[name]["return"]):
            v = table[key]
            lines.append(
                f"{LABELS.get(key, key):<27} {v['return']:>9.4f}"
                f" {v['sla_URLLC']:>10.2%} {v['sla_eMBB']:>10.2%}"
                f" {v['switch_rate']:>8.2%} {v['abstention_rate']:>9.2%}"
            )
        pulse_pairs = data.get("paired_vs_pulse_ran", {}).get(split, {})
        if pulse_pairs:
            lines.extend((
                "",
                "Paired differences: challenger minus PULSE-RAN (negative favors PULSE)",
                f"{'Challenger':<27} {'Delta':>9} {'95% CI':>23} {'Holm p':>9}",
            ))
            for name, stats in sorted(
                pulse_pairs.items(),
                key=lambda kv: kv[1]["mean_episode_delta"],
            ):
                lo, hi = stats["ci95"]
                lines.append(
                    f"{LABELS.get(name, name):<27}"
                    f" {stats['mean_episode_delta']:>+9.4f}"
                    f" [{lo:+.4f}, {hi:+.4f}]"
                    f" {stats['holm_adjusted_p']:>9.4f}"
                )
    lines.extend((
        "",
        "Methodology: all controllers were evaluated on matched episode seeds;",
        "PULSE-RAN is unshielded; PULSE-RAN Gated is a separate diagnostic;",
        "the legacy learned-family selector excludes PULSE-RAN.",
        data["scientific_scope"],
    ))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PULSE-RAN toy experiment")
    parser.add_argument("--input", type=Path, required=True, help="Study JSON path")
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    print(format_report(data))


if __name__ == "__main__":
    main()
