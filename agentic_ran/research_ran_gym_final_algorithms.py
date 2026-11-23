"""Matched executed Gymnasium test of final *compatible* discrete-action algorithms.

All policies operate on Agentic-RAN's five-action, nine-feature, three-slice
action-dependent TOY Gymnasium environment, NOT native ns-O-RAN/5G-LENA.
COMMAG's scheduler/PRB trained policies cannot be used with the stock
ns-O-RAN handover action space. This suite verifies learned-policy
integration, deterministic reset, paired outcomes and no test leakage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from agentic_ran.research_ppo_wsp import (
    _evaluations, _search, train_ppo,
)
from agentic_ran.research_ran_gym import episode, paired_statistics
from agentic_ran.research_ran_gym_adapters import make_toy_env
from agentic_ran.research_twin_environment import SCENARIOS, generate_logged_transitions
from agentic_ran.research_twin_offline import (
    choose_action, fit_calibrated_twin, fit_linear_cql, fit_quantile_td,
)


def run(
    output: Path, *, seed: int = 2026,
    ppo_updates: int = 20, episodes_per_update: int = 8,
    logged_episodes: int = 72, validation_episodes: int = 12,
    test_seeds: int = 20, steps: int = 24, wsp_rounds: int = 2,
    critic_iterations: int = 3,
) -> dict[str, Any]:
    """Freeze training and validation before touching matched held-out Gym seeds."""
    if min(ppo_updates, episodes_per_update, logged_episodes,
           validation_episodes, test_seeds, steps, critic_iterations) < 1:
        raise ValueError("all training/evaluation sizes must be positive")
    if episodes_per_update < 2 or logged_episodes < 12 or validation_episodes < 4:
        raise ValueError("minimum PPO/critic/validation samples not met")
    if test_seeds < 4 or steps < 5 or not 1 <= wsp_rounds <= 5:
        raise ValueError("too few independent test seeds/steps or invalid WSP rounds")
    torch.set_num_threads(1)

    # 1. Freeze original PPO; WSP edits are selected ONLY on validation seeds.
    frozen = train_ppo(
        seed=seed + 100000, updates=ppo_updates,
        episodes_per_update=episodes_per_update, steps=steps,
    )
    validation_seed = seed + 20_000_000
    reference = _evaluations(
        frozen.choose, seed=validation_seed, count=validation_episodes,
        steps=steps, scenarios=("normal", "congested", "bursty"),
    )
    edited, accepted_edits = _search(
        frozen, reference, seed=seed + 310, validation_seed=validation_seed,
        val_episodes=validation_episodes, steps=steps, rounds=wsp_rounds,
    )

    # 2. Train genuine action-dependent offline toy competitors from logged
    # randomized TRAIN interventions. Neither PPO test nor Gym test is read.
    logged = generate_logged_transitions(
        seed=seed + 10_000_000, episodes=logged_episodes, steps=steps,
    )
    twin = fit_calibrated_twin(logged)
    cql = fit_linear_cql(
        logged, alpha=0.05, iterations=critic_iterations, steps_per_iter=35,
    )
    quantile = fit_quantile_td(
        logged, iterations=critic_iterations, steps_per_iter=40,
    )

    def rl(mode: str):
        def decide(state, previous, t, _info):
            prior = 0 if previous is None else int(previous)
            result = choose_action(
                mode, np.asarray(state, dtype=float), prior, t,
                twin=twin, cql=cql, quantile=quantile,
                risk_weight=0.35, blend_weight=0.5,
            )
            # In this toy action space, the declared abstention semantics
            # are to issue the *previous* allocation; Gym receives that
            # explicit supported action, not a made-up action index -1.
            return int(prior if result == -1 else result)
        return decide

    selectors = {
        "ppo_wsp": lambda s, p, t, i: edited.choose(s, p, t),
        "ppo_frozen": lambda s, p, t, i: frozen.choose(s, p, t),
        "pulse_ran": rl("pulse_ran"),
        "twin_cql": rl("twin_cql"),
        "distributional_cvar": rl("distributional_cvar"),
        "twin_cql_cvar_hybrid": rl("twin_cql_cvar_hybrid"),
    }
    parameters = {"max_steps": steps}
    episode_rows, step_rows = [], []
    # Test seeds and scenarios are *never* consumed until all choices freeze.
    heldout = tuple(seed + 30_000_000 + i * 997 for i in range(test_seeds))
    for scenario in SCENARIOS:
        for test_seed in heldout:
            for name, selector in selectors.items():
                def factory(_config, action_space, selector=selector):
                    if int(action_space.n) != 5:
                        raise ValueError("toy final-policy action space is not five-action")
                    return selector

                result, traces = episode(
                    make_toy_env, factory, parameters, label=name,
                    scenario=scenario, seed=test_seed, steps=steps,
                    strict_receipt=True,
                )
                episode_rows.append(result)
                step_rows.extend(traces)
    frames = pd.DataFrame(episode_rows)
    traces = pd.DataFrame(step_rows)
    initial = frames.groupby(["scenario", "seed"])["initial_state_sha256"].nunique()
    if not bool(initial.eq(1).all()):
        raise RuntimeError("Gym matched-seed reset determinism failed")
    if not bool(traces["receipt_verified"].all()):
        raise RuntimeError("Toy Gym action receipt mismatch")
    if not bool(frames["steps"].eq(steps).all()):
        raise RuntimeError("unexpected truncated toy Gym episode")

    candidate = frames[frames["policy"] == "ppo_wsp"]
    comparisons = {}
    for name in selectors:
        if name != "ppo_wsp":
            comparisons[name] = paired_statistics(
                candidate, frames[frames["policy"] == name],
                seed=seed + 7013, draws=2000,
            )
    order = sorted(comparisons, key=lambda k: comparisons[k]["one_sided_sign_flip_p"])
    adjusted = 0.0
    for index, name in enumerate(order):
        comparison = comparisons[name]
        adjusted = max(adjusted, min(
            1.0, comparison["one_sided_sign_flip_p"] * (len(order) - index)
        ))
        comparison["holm_adjusted_p"] = adjusted
        comparison["significant_gain"] = bool(
            comparison["bootstrap_ci95"][0] > 0 and adjusted < 0.05
        )

    report: dict[str, Any] = {
        "schema": "agentic-ran-final-algorithms-toy-gym-v1",
        "backend": "agentic-ran-toy-gymnasium",
        "actual_gym_step_executed": True,
        "same_five_action_space": True,
        "matched_initial_states_verified": True,
        "action_receipts": "self_reported_by_toy_gym_only_not_physical_receipts",
        "training_and_validation_frozen_before_test": True,
        "test_used_for_hyperparameter_selection": False,
        "no_real_prb_scheduler_actuation": True,
        "native_nsoran_gym_controllers_tested": False,
        "native_colosseum_tested": False,
        "publication_superiority_ready": False,
        "publication_experiment_ready": False,
        "publication_restriction": (
            "The experiment uses a constructed toy environment; neither "
            "a real 5G-LENA NR scheduler nor COMMAG's frozen PRB policy is actuated."
        ),
        "heldout_test_seeds": list(heldout),
        "scenarios": list(SCENARIOS),
        "steps_per_episode": steps,
        "independent_test_seeds": test_seeds,
        "executed_episodes": len(episode_rows),
        "train_logged_episodes": logged_episodes,
        "validation_episodes": validation_episodes,
        "wsp_accepted_edits": accepted_edits,
        "methods": {
            name: {
                "mean_total_reward": float(frames[frames["policy"] == name]["total_reward"].mean()),
                "mean_sla_violation_rate": float(
                    frames[frames["policy"] == name]["mean_sla_violation_rate"].mean()
                ),
                "mean_switch_rate": float(frames[frames["policy"] == name]["switch_rate"].mean()),
            } for name in selectors
        },
        "paired_wsp_vs": comparisons,
        "all_methods_physically_comparable": False,
        "missing_for_real_ran_publication": [
            "verified native PRB/scheduler gym action space compatible with frozen COMMAG controller",
            "measured physical BWP capacity and real actuation receipts",
            "paired independent native/Colosseum throughput latency and per-slice SLA measurements",
            "manuscript novelty, multiple-training-seed robustness and scientific reviewer audit",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    frames.to_csv(output / "gym_final_episodes.csv", index=False)
    traces.to_csv(output / "gym_final_steps.csv.gz", index=False, compression="gzip")
    torch.save(frozen.state_dict(), output / "ppo_frozen.pt")
    torch.save(edited.state_dict(), output / "ppo_wsp.pt")
    (output / "gym_final_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Executed final-policy TOY Gymnasium comparison")
    parser.add_argument("--output", type=Path, default=Path("results/ran-gym-final-algorithms"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--ppo-updates", type=int, default=20)
    parser.add_argument("--episodes-per-update", type=int, default=8)
    parser.add_argument("--logged-episodes", type=int, default=72)
    parser.add_argument("--validation-episodes", type=int, default=12)
    parser.add_argument("--test-seeds", type=int, default=20)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--wsp-rounds", type=int, default=2)
    parser.add_argument("--critic-iterations", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(run(
        args.output, seed=args.seed, ppo_updates=args.ppo_updates,
        episodes_per_update=args.episodes_per_update,
        logged_episodes=args.logged_episodes,
        validation_episodes=args.validation_episodes,
        test_seeds=args.test_seeds, steps=args.steps,
        wsp_rounds=args.wsp_rounds, critic_iterations=args.critic_iterations,
    ), indent=2))


if __name__ == "__main__":
    main()
