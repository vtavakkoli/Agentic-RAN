"""Fast *synthetic* controlled experiment for incumbent-safe RAN policies.

This creates action-dependent potential outcomes by construction; COMMAG
does NOT supply such intervention evidence. Use only to verify policy ranking,
risk/coverage tradeoffs and paired evaluation code, never as paper results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from agentic_ran.research_safe_improvement import select_safe_improvement
from agentic_ran.research_support import rerank


def run_synthetic_comparison(seed: int = 2026, episodes: int = 40, steps: int = 24):
    if episodes < 5 or steps < 3:
        raise ValueError("need >=5 episodes and >=3 steps")
    rng = np.random.default_rng(seed)
    n = episodes * steps
    quality = rng.normal(0, 1, n)
    load = rng.uniform(0, 1, n)
    # Potential outcomes are known ONLY because this is a toy generative world.
    true_utility = np.column_stack([
        0.75 - 0.025 * quality ** 2 - 0.05 * load,
        0.78 + 0.055 * quality - 0.02 * load,
        0.73 + 0.095 * load - 0.015 * quality,
    ])
    true_sla = np.column_stack([
        0.20 + 0.05 * load,
        0.24 + 0.11 * load + 0.05 * np.maximum(-quality, 0),
        0.18 + 0.32 * load + 0.04 * np.maximum(quality, 0),
    ])
    # Model noise, domain shift and an imperfect critic make choice nontrivial.
    scores = true_utility + rng.normal(0, 0.045, (n, 3))
    estimated_sla = np.clip(true_sla + rng.normal(0, 0.035, (n, 3)), 0, 1)
    uncertainty = 0.035 + rng.uniform(0, 0.065, (n, 3))
    allowed = np.ones((n, 3), dtype=bool)
    support = rng.uniform(size=(n, 3)) > np.array([0.08, 0.20, 0.25])[None, :]
    incumbents = np.zeros(n, dtype=int)
    prior = np.tile([0.67, 0.26, 0.07], (n, 1))
    priorshift = np.clip(quality, -2, 2) * 0.03
    prior[:, 0] -= priorshift
    prior[:, 1] += priorshift
    modes = [
        "conformal_rerank", "advantage_lcb", "constraint_conditioned",
        "behavior_anchored_lcb", "verified_hold",
    ]
    records = []
    for mode in modes:
        if mode == "conformal_rerank":
            decision = rerank(
                scores, estimated_sla, uncertainty, allowed, support,
                incumbents, method="toy", alpha=0.2, max_sla=0.5,
                max_uncertainty=0.5, switch_cost=0.01,
            )
        else:
            decision = select_safe_improvement(
                scores, estimated_sla, uncertainty, allowed, support, incumbents,
                mode=mode, max_sla=0.5, max_uncertainty=0.5,
                advantage_penalty=0.25, risk_weight=0.4,
                behavior_prior=prior, behavior_weight=0.04,
            )
        accepted = ~decision.abstain
        ix = np.arange(n)
        chosen = np.maximum(decision.actions, 0)
        act_utility = true_utility[ix, chosen]
        act_risk = true_sla[ix, chosen]
        # Two *explicitly different* no-action estimands:
        # (i) arbitrary -0.12 penalty and
        # (ii) known incumbent potential outcome in the toy simulator.
        fixed_score = np.where(accepted, act_utility, -0.12)
        hold_score = np.where(accepted, act_utility, true_utility[ix, incumbents])
        for e in range(episodes):
            sl = slice(e * steps, (e + 1) * steps)
            valid = accepted[sl]
            records.append({
                "mode": mode, "episode": e,
                "coverage": float(valid.mean()),
                "switch_rate": float(np.mean(
                    valid & (decision.actions[sl] != incumbents[sl])
                )),
                "fixed_penalty_score": float(fixed_score[sl].mean()),
                "toy_known_hold_score": float(hold_score[sl].mean()),
                "conditional_utility_sum": float(
                    act_utility[sl][valid].sum()
                ),
                "accepted": int(valid.sum()),
                "toy_sla_violations": int(
                    (act_risk[sl][valid] > 0.5).sum()
                ),
            })
    output = {}
    for mode in modes:
        chunk = [row for row in records if row["mode"] == mode]
        accepted_count = sum(row["accepted"] for row in chunk)
        output[mode] = {
            "coverage": float(np.mean([x["coverage"] for x in chunk])),
            "switch_rate": float(np.mean([x["switch_rate"] for x in chunk])),
            "conditional_true_utility": (
                sum(x["conditional_utility_sum"] for x in chunk)
                / max(accepted_count, 1)
            ),
            "fixed_penalty_score": float(
                np.mean([x["fixed_penalty_score"] for x in chunk])
            ),
            "toy_known_hold_score": float(
                np.mean([x["toy_known_hold_score"] for x in chunk])
            ),
            "toy_sla_violation_rate": (
                sum(x["toy_sla_violations"] for x in chunk)
                / max(accepted_count, 1)
            ),
        }
    control = np.array([
        x["toy_known_hold_score"] for x in records
        if x["mode"] == "conformal_rerank"
    ])
    for mode in modes[1:]:
        condition = np.array([
            x["toy_known_hold_score"] for x in records if x["mode"] == mode
        ])
        paired = condition - control
        # Resampling independent toy episodes (not within-episode timesteps).
        draws = rng.integers(0, episodes, size=(1000, episodes))
        boot = np.mean(paired[draws], axis=1)
        output[mode]["paired_known_hold_delta_vs_conformal"] = float(paired.mean())
        output[mode]["paired_known_hold_ci95"] = [
            float(x) for x in np.quantile(boot, [0.025, 0.975])
        ]
    validation_rank = sorted(
        modes, key=lambda name: -output[name]["toy_known_hold_score"],
    )
    return {
        "experiment": "synthetic_potential_outcomes_only_not_COMMAG",
        "seed": seed, "episodes": episodes, "steps_per_episode": steps,
        "ranking_by_toy_known_hold": validation_rank,
        "results": output,
        "limitations": (
            "Synthetic action-dependent labels make counterfactual comparison "
            "possible here; the original static COMMAG log does not. "
            "This is NOT a publication superiority result."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_synthetic_comparison(args.seed, args.episodes, args.steps)
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
