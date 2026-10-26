"""Synthetic comparison of support estimators, safe re-ranking and MPC.

DO NOT interpret simulated oracle utility as counterfactual COMMAG evidence.
Thresholds and method are selected *only* on a validation split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from agentic_ran.research_planning import evaluate_planners
from agentic_ran.research_support import ConformalActionSupport, rerank

ACTION_NAMES = (
    "round_robin:prb=4",
    "waterfilling:prb=8",
    "proportional_fair:prb=12",
)


def generate(rng, n, shift=0.0):
    x = rng.normal(size=(n, 6))
    x[:, 0] += shift
    x[:, 1] = 0.65 * x[:, 0] + 0.6 * x[:, 1]
    slices = rng.choice(["eMBB", "URLLC"], size=n)
    logits = np.column_stack((0.4 - x[:, 0], x[:, 1], x[:, 0] - 0.3))
    prob = np.exp(logits - logits.max(axis=1, keepdims=True))
    prob /= prob.sum(axis=1, keepdims=True)
    actions = (rng.random(n)[:, None] > np.cumsum(prob, axis=1)).sum(axis=1)
    return x, slices, actions


def outcomes(x, slices, seed):
    """Known simulated action-dependent utility and risk plus noisy predictions."""
    risk = np.stack((
        0.12 + 0.03 * np.abs(x[:, 0]),
        0.15 + 0.11 * np.maximum(x[:, 1], 0),
        0.20 + 0.24 * np.maximum(-x[:, 0], 0),
    ), axis=1)
    risk += (np.asarray(slices) == "URLLC")[:, None] * 0.04
    risk = np.clip(risk, 0.01, 0.95)
    utility = np.stack((
        0.63 - 0.04 * x[:, 0]**2,
        0.70 + 0.08 * x[:, 1] - 0.07 * risk[:, 1],
        0.77 + 0.11 * x[:, 0] - 0.10 * risk[:, 2],
    ), axis=1)
    rng = np.random.default_rng(seed)
    scores = utility + rng.normal(0, 0.035, utility.shape) + [0, 0, 0.045]
    risk_upper = np.clip(
        risk + 0.05 + rng.normal(0, 0.012, risk.shape), 0, 1
    )
    uncertainty = np.clip(
        0.09 + 0.11 * np.abs(x[:, 0, None]) * [[0.5, 0.7, 1.0]],
        0, 1,
    )
    return utility, risk, scores, risk_upper, uncertainty


def evaluate_support(seed=2026, train_per_role=3000, validation=850, test=1400):
    rng = np.random.default_rng(seed)
    roles = {}
    for role in ("policy", "evaluator"):
        x, sl, actions = generate(
            rng, train_per_role, shift=0.04 if role == "evaluator" else 0,
        )
        roles[role] = (x, sl, np.asarray(ACTION_NAMES, dtype=object)[actions])
    estimator = ConformalActionSupport(
        seed=seed, max_reference=800,
    ).fit(roles)
    cases = {
        "validation": generate(rng, validation),
        "seen": generate(rng, test),
        "shifted": generate(rng, test, shift=1.1),
    }
    methods = ("knn", "gaussian", "gmm")
    alphas = (0.01, 0.05, 0.10, 0.20)
    evaluated = {}
    selected = {}
    for case, (x, slices, logged) in cases.items():
        utility, risk, scores, upper, uncertainty = outcomes(
            x, slices, seed + {"validation": 1, "seen": 2, "shifted": 3}[case],
        )
        allowed = np.ones(scores.shape, dtype=bool)
        results = {}
        for method in methods:
            p = np.column_stack([
                estimator.pvalues(x, slices, name, method)
                for name in ACTION_NAMES
            ])
            for alpha in alphas:
                choice = rerank(
                    scores, upper, uncertainty, allowed, p > alpha,
                    logged, method=method, alpha=alpha, switch_cost=0.01,
                )
                active = ~choice.abstain
                rows = np.flatnonzero(active)
                vals = np.full(len(x), -0.12)  # explicit abstention opportunity cost
                vals[rows] = utility[rows, choice.actions[rows]]
                selected_risk = risk[rows, choice.actions[rows]] if len(rows) else []
                # Matched eligibility: old method rejects the entire top
                # proposal; re-ranking searches remaining safe candidates.
                legacy_top = np.argmax(scores, axis=1)
                legacy_ok = (
                    (p > alpha) & (upper <= 0.45)
                    & (uncertainty <= 0.5)
                )[np.arange(len(x)), legacy_top]
                legacy_vals = np.full(len(x), -0.12)
                legacy_rows = np.flatnonzero(legacy_ok)
                legacy_vals[legacy_rows] = utility[
                    legacy_rows, legacy_top[legacy_rows]
                ]
                results[f"{method}:{alpha:.2f}"] = {
                    "mean_utility_with_abstention": float(np.mean(vals)),
                    "legacy_deferral_utility": float(np.mean(legacy_vals)),
                    "utility_gain_over_legacy_deferral": float(np.mean(vals - legacy_vals)),
                    "legacy_actuation_coverage": float(legacy_ok.mean()),
                    "actuation_coverage": float(np.mean(active)),
                    "conditional_true_risk": (
                        float(np.mean(selected_risk)) if len(rows) else None
                    ),
                    "rerank_rate": float(np.mean(choice.reranked)),
                    "abstain_rate": float(np.mean(choice.abstain)),
                    "mean_eligible_actions": float(np.mean(choice.eligible_counts)),
                }
        evaluated[case] = results
        if case == "validation":
            # No held-out information influences estimator/alpha selection.
            winner = max(
                results, key=lambda key: (
                    results[key]["mean_utility_with_abstention"]
                    - 0.01 * results[key]["abstain_rate"]
                ),
            )
            selected = {
                "winner": winner,
                "criterion": "validation utility - 0.01*abstention",
            }
    return {
        "experimental_only": True,
        "selection": selected,
        "heldout_seen": evaluated["seen"][selected["winner"]],
        "heldout_shifted": evaluated["shifted"][selected["winner"]],
        "all_results": evaluated,
        "warning": (
            "Synthetic oracle utility/risk. Conformal exchangeability and "
            "action-conditional coverage assumptions can fail under shift."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Experimental RAN methods benchmark")
    parser.add_argument("--output", default="results/research_methods.json")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--episodes", type=int, default=12)
    args = parser.parse_args(argv)
    result = {
        "support": evaluate_support(seed=args.seed),
        "planning": evaluate_planners(seed=args.seed, episodes=args.episodes),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "best_validation_method": result["support"]["selection"]["winner"],
        "seen": result["support"]["heldout_seen"],
        "shifted": result["support"]["heldout_shifted"],
        "planning_rmse": result["planning"]["dynamics_holdout_rmse"],
        "output": str(path),
    }, indent=2))


if __name__ == "__main__":
    main()
