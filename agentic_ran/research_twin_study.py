"""Paired, action-dependent experimental study: twin CQL vs quantile CVaR.

The experiment explicitly separates training, validation, seen final test, and
scenario-shifted unseen final test. Hyperparameters and the ensemble family
are frozen on validation *before* any test episodes are generated. All scored
outcomes come from the action-dependent toy network transition dynamics,
NOT from independent OPE predictions on static COMMAG data.

No claim of realistic 5G/6G link dynamics or online safety certification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from agentic_ran.research_twin_environment import (
    MultiSliceToyTwin, SCENARIOS, generate_logged_transitions, rollout_episode,
)
from agentic_ran.research_twin_offline import (
    choose_action, fit_calibrated_twin, fit_linear_cql, fit_quantile_td,
)

MODELS = (
    "pulse_ran", "pulse_ran_gated", "twin_greedy", "twin_greedy_cvar_gate",
    "twin_cql", "distributional_cvar", "twin_cql_cvar_hybrid",
)
RISK_REGRESSION_TOLERANCE = 0.015
MIN_VALIDATION_UTILITY_GAIN = 0.003


def _evaluate(
    *, selector, seed: int, episodes: int, steps: int, scenarios: tuple[str, ...],
    label: str,
) -> list[dict]:
    result = []
    for ep in range(episodes):
        scenario = scenarios[ep % len(scenarios)]
        # Same start/noise sequence across all tested policies.
        seed_ep = seed + ep * 997
        output = rollout_episode(
            selector, seed=seed_ep, scenario=scenario, steps=steps,
        )
        result.append({
            "policy": label,
            "episode": ep, "scenario": scenario, "seed": seed_ep,
            **output,
        })
    return result


def _aggregate(rows: list[dict]) -> dict[str, float]:
    return {
        metric: float(np.mean([x[metric] for x in rows]))
        for metric in (
            "return", "sla_eMBB", "sla_mMTC", "sla_URLLC",
            "switch_rate", "abstention_rate",
        )
    }


def _regression_safe(candidate: dict, reference: dict) -> bool:
    return (
        candidate["sla_URLLC"]
        <= reference["sla_URLLC"] + RISK_REGRESSION_TOLERANCE
        and candidate["sla_eMBB"]
        <= reference["sla_eMBB"] + RISK_REGRESSION_TOLERANCE
    )


def _validation_criterion(metrics: dict, baseline: dict) -> float:
    # Values measured after simulated intervention; not a deployable reward.
    urllc_excess = max(
        0.0, metrics["sla_URLLC"]
        - baseline["sla_URLLC"] - RISK_REGRESSION_TOLERANCE
    )
    embb_excess = max(
        0.0, metrics["sla_eMBB"]
        - baseline["sla_eMBB"] - RISK_REGRESSION_TOLERANCE
    )
    return (
        metrics["return"]
        - 1.5 * urllc_excess - 0.5 * embb_excess
        - 0.005 * metrics["switch_rate"]
    )


def _paired_inference(
    candidate: list[dict], reference: list[dict], *, seed: int, nboot: int = 2000,
) -> dict:
    if len(candidate) != len(reference) or len(candidate) < 4:
        raise ValueError("need matched independent test episodes")
    if [(x["seed"], x["scenario"]) for x in candidate] != [
        (x["seed"], x["scenario"]) for x in reference
    ]:
        raise ValueError("policies are not paired by seed and scenario")
    d = np.asarray([
        a["return"] - b["return"] for a, b in zip(candidate, reference)
    ], dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(d), size=(nboot, len(d)))
    boot = np.mean(d[samples], axis=1)
    # Sign-randomization is exploratory when the policy has been chosen
    # on validation data; test episodes remain entirely untouched.
    signs = rng.choice((-1.0, 1.0), size=(4000, len(d)))
    p = (1 + np.sum(
        np.abs((signs * d[None, :]).mean(axis=1))
        >= abs(float(d.mean())) - 1e-12
    )) / (4000 + 1)
    return {
        "mean_episode_delta": float(d.mean()),
        "ci95": [float(x) for x in np.quantile(boot, [.025, .975])],
        "paired_sign_permutation_p": float(p),
        "episode_win_rate": float(np.mean(d > 0)),
        "n_independent_episodes": len(d),
        "mean_reference_return": float(np.mean([x["return"] for x in reference])),
    }


def _holm_adjust(comparisons: dict[str, dict]) -> None:
    ordered = sorted(
        comparisons, key=lambda key: comparisons[key]["paired_sign_permutation_p"],
    )
    largest = 0.0
    m = len(ordered)
    for i, key in enumerate(ordered):
        largest = max(
            largest, (m - i) * comparisons[key]["paired_sign_permutation_p"]
        )
        comparisons[key]["holm_adjusted_p"] = float(min(1.0, largest))


def run_twin_study(
    *, seed: int = 2026, train_episodes: int = 72,
    validation_episodes: int = 16, test_episodes: int = 24,
    steps: int = 20,
) -> tuple[dict, pd.DataFrame]:
    if min(train_episodes, validation_episodes, test_episodes) < 6 or steps < 5:
        raise ValueError("need >=6 episodes per partition and >=5 steps")
    if train_episodes < 12:
        raise ValueError("need >=12 training episodes for disjoint twin calibration")
    # Episode IDs and environment seeds do not overlap between partitions.
    logged = generate_logged_transitions(
        seed=seed + 10000000, episodes=train_episodes, steps=steps,
    )
    action_changes = int(np.sum(
        (logged.action[1:] != logged.action[:-1])
        & (logged.episode[1:] == logged.episode[:-1])
    ))
    if action_changes < 10:
        raise RuntimeError("offline action-dependent logging unexpectedly absent")
    twin = fit_calibrated_twin(logged)
    models = {
        f"twin_cql:alpha={alpha}": fit_linear_cql(
            logged, alpha=alpha, iterations=4, steps_per_iter=55,
        ) for alpha in (0.05, 0.25)
    }
    quantile = fit_quantile_td(logged, iterations=4, steps_per_iter=65)

    def selector(mode: str, *, alpha: float = .05,
                 risk_weight: float = .35, blend_weight: float = .5):
        model = models[f"twin_cql:alpha={alpha}"] if mode in (
            "twin_cql", "twin_cql_cvar_hybrid"
        ) else None
        return lambda state, previous, t: choose_action(
            mode,
            state, previous, t, twin=twin, cql=model, quantile=quantile,
            risk_weight=risk_weight, blend_weight=blend_weight,
        )

    candidates = {
        "pulse_ran": [{"description": "PULSE-RAN priority rule (original unshielded baseline)"}],
        "pulse_ran_gated": [{"description": "PULSE-RAN with shared action-support/queue-risk gate"}],
        "twin_greedy": [{"description": "one_step_greedy"}],
        "twin_greedy_cvar_gate": [
            {"description": "one_step_greedy_with_cvar_queue_risk_mask"}
        ],
        "twin_cql": [
            {"description": f"alpha={a}", "alpha": a}
            for a in (0.05, 0.25)
        ],
        "distributional_cvar": [
            {"description": f"cvar_weight={w}", "risk_weight": w}
            for w in (0.0, 0.35, 0.70)
        ],
        "twin_cql_cvar_hybrid": [
            {"description": f"mix={w}", "risk_weight": .35,
             "blend_weight": w, "alpha": .05}
            for w in (.3, .7)
        ],
    }
    validation_scenarios = ("normal", "congested", "bursty")
    validation_seed = seed + 20000000
    baseline_val = _evaluate(
        selector=selector("twin_greedy"), seed=validation_seed,
        episodes=validation_episodes, steps=steps,
        scenarios=validation_scenarios, label="twin_greedy",
    )
    baseline_metrics = _aggregate(baseline_val)
    conservative_baseline_val = _evaluate(
        selector=selector("twin_greedy_cvar_gate"), seed=validation_seed,
        episodes=validation_episodes, steps=steps,
        scenarios=validation_scenarios, label="twin_greedy_cvar_gate",
    )
    conservative_baseline_metrics = _aggregate(conservative_baseline_val)
    val_grid = []
    frozen = {}
    validation_flags = {}
    for family, settings in candidates.items():
        options = []
        for hyper in settings:
            trial = (
                baseline_val if family == "twin_greedy"
                else conservative_baseline_val if family == "twin_greedy_cvar_gate"
                else _evaluate(
                    selector=selector(
                        family, **{k: v for k, v in hyper.items()
                                  if k in ("alpha", "risk_weight", "blend_weight")}
                    ),
                    seed=validation_seed, episodes=validation_episodes,
                    steps=steps, scenarios=validation_scenarios,
                    label=family,
                )
            )
            metrics = _aggregate(trial)
            matched_reference = (
                conservative_baseline_metrics
                if family in ("distributional_cvar", "twin_cql_cvar_hybrid")
                else baseline_metrics
            )
            acceptable = _regression_safe(metrics, matched_reference)
            record = {
                "family": family, "hyperparameters": hyper,
                "metrics": metrics,
                "estimated_sla_non_regression": acceptable,
                "objective": _validation_criterion(metrics, matched_reference),
                "validation_reference": (
                    "twin_greedy_cvar_gate"
                    if family in ("distributional_cvar", "twin_cql_cvar_hybrid")
                    else "twin_greedy"
                ),
            }
            options.append(record)
            val_grid.append(record)
        acceptable_options = [x for x in options if x["estimated_sla_non_regression"]]
        best = max(
            acceptable_options if acceptable_options else options,
            key=lambda x: x["objective"],
        )
        frozen[family] = best["hyperparameters"]
        validation_flags[family] = {
            "risk_feasible": bool(acceptable_options),
            "validation_metrics": best["metrics"],
            "validation_objective": best["objective"],
            "hyperparameters": best["hyperparameters"],
        }

    # The high-level policy family selector is also locked on validation,
    # and needs a minimum practical gain to promote a learned controller.
    permitted = [
        (key, value) for key, value in validation_flags.items()
        if key in ("twin_cql", "distributional_cvar", "twin_cql_cvar_hybrid")
        and value["risk_feasible"]
        and value["validation_metrics"]["return"]
        >= baseline_metrics["return"] + MIN_VALIDATION_UTILITY_GAIN
    ]
    selected_family = (
        max(permitted, key=lambda x: x[1]["validation_objective"])[0]
        if permitted else "twin_greedy"
    )
    row_outputs = []
    comparisons = {}
    matched_gate_comparisons = {}
    pulse_comparisons = {}
    score_tables = {}
    for split, test_seed, scenarios in (
        ("seen", seed + 30000000, ("normal", "congested", "bursty")),
        ("unseen", seed + 40000000, ("weak_channel",)),
    ):
        test = {}
        for family, hyper in frozen.items():
            test[family] = _evaluate(
                selector=selector(
                    family,
                    **{k: v for k, v in hyper.items()
                       if k in ("alpha", "risk_weight", "blend_weight")}
                ),
                seed=test_seed, episodes=test_episodes, steps=steps,
                scenarios=scenarios, label=family,
            )
            row_outputs.extend([{"split": split, **row} for row in test[family]])
        # The selected model is an alias of one tested family, not a post-hoc
        # oracle that chooses whichever family scores best on held-out data.
        baseline = test["twin_greedy"]
        comparisons[split] = {}
        score_tables[split] = {key: _aggregate(value) for key, value in test.items()}
        same_gate = {}
        for j, family in enumerate(
            ("twin_cql", "distributional_cvar", "twin_cql_cvar_hybrid")
        ):
            reference_name = (
                "twin_greedy_cvar_gate"
                if family in ("distributional_cvar", "twin_cql_cvar_hybrid")
                else "twin_greedy"
            )
            comparisons[split][family] = _paired_inference(
                test[family], baseline,
                seed=seed + 400000 + j * 10 + (0 if split == "seen" else 1),
            )
            comparisons[split][family]["sla_non_regression"] = (
                _regression_safe(
                    score_tables[split][family],
                    score_tables[split]["twin_greedy"],
                )
            )
            same_gate[family] = _paired_inference(
                test[family], test[reference_name],
                seed=seed + 500000 + j * 10 + (0 if split == "seen" else 1),
            )
            same_gate[family]["reference"] = reference_name
            same_gate[family]["sla_non_regression"] = (
                _regression_safe(
                    score_tables[split][family],
                    score_tables[split][reference_name],
                )
            )
        _holm_adjust(comparisons[split])
        _holm_adjust(same_gate)
        matched_gate_comparisons[split] = same_gate
        # PULSE-RAN is compared on the SAME realized episode seeds,
        # not merely against mean outcomes from different trajectories.
        # Negative differences below mean PULSE-RAN has higher reward.
        pulse_comparisons[split] = {
            mode: _paired_inference(
                test[mode], test["pulse_ran"],
                seed=seed + 600000 + 100 * int(split == "unseen") + j,
            )
            for j, mode in enumerate(MODELS) if mode != "pulse_ran"
        }
        _holm_adjust(pulse_comparisons[split])
    rigorous = {}
    for family in ("twin_cql", "distributional_cvar", "twin_cql_cvar_hybrid"):
        seen = comparisons["seen"][family]
        unseen = comparisons["unseen"][family]
        rigorous[family] = {
            "positive_both_splits": (
                seen["mean_episode_delta"] > 0
                and unseen["mean_episode_delta"] > 0
            ),
            "ci_lower_positive_both_splits": (
                seen["ci95"][0] > 0 and unseen["ci95"][0] > 0
            ),
            "holm_significant_both_splits": (
                seen["holm_adjusted_p"] < .05 and unseen["holm_adjusted_p"] < .05
            ),
            "realized_sla_non_regression_both_splits": (
                seen["sla_non_regression"] and unseen["sla_non_regression"]
            ),
        }
        rigorous[family]["satisfies_all_exploratory_checks"] = all(
            rigorous[family].values()
        )
    summary = {
        "experiment": "three_slice_action_dependent_toy_digital_twin",
        "named_priority_controller": {
            "id": "pulse_ran",
            "display_name": "PULSE-RAN",
            "expansion": "Priority-driven URLLC and Load-Sensitive Engine",
            "method_unchanged": True,
            "original_heuristic_alias": "heuristic",
            "gated_diagnostic_id": "pulse_ran_gated",
            "gated_diagnostic_is_original_policy": False,
            "original_baseline_uses_common_learned_support_gate": False,
        },
        "scientific_scope": (
            "Controlled simulated interventions only; not COMMAG, ns-3, "
            "Colosseum, or measured real RAN outcomes."
        ),
        "random_seed": seed,
        "sample_sizes": {
            "offline_training_episodes": train_episodes,
            "offline_training_rows": len(logged.reward),
            "validation_episodes": validation_episodes,
            "test_episodes_per_split": test_episodes,
            "steps_per_episode": steps,
        },
        "action_audit": {
            "observed_within_episode_switches": action_changes,
            "different_train_actions": int(len(np.unique(logged.action))),
            "counterfactual_outcomes": "simulator_intervention_not_observational_commag",
        },
        "learned_dynamics": {
            "calibration_rows": twin.calibration_rows,
            "calibration_episodes": twin.calibration_episodes,
            "queue_positive_residual_p90": twin.queue_upper_margin.tolist(),
            "queue_upper_tail_CVaR_margin": twin.queue_cvar_margin.tolist(),
            "state_conditional_action_support": True,
        },
        "critics": {
            "cql_td_rmse": {
                key: float(model.training_td_rmse)
                for key, model in models.items()
            },
            "quantile_bellman_error_proxy": quantile.training_pinball,
            "implementation": (
                "finite-action linear CQL and quantile-Huber TD variants; "
                "not exact neural-network CQL/CODAC reproduction"
            ),
        },
        "validation_grid": val_grid,
        "frozen_validation_choices": validation_flags,
        "validation_ranked_all": sorted(
            validation_flags,
            key=lambda mode: validation_flags[mode]["validation_metrics"]["return"],
            reverse=True,
        ),
        "validation_selected_family": selected_family,
        "validation_selection_used_final_test": False,
        "final_test_metrics": score_tables,
        "final_test_ranked_by_utility": {
            split: sorted(score_tables[split], key=lambda mode: score_tables[split][mode]["return"], reverse=True)
            for split in ("seen", "unseen")
        },
        "paired_vs_pulse_ran": pulse_comparisons,
        "pulse_comparison_direction": "challenger_minus_pulse_ran; negative favors PULSE-RAN",
        "paired_vs_original_twin_greedy": comparisons,
        "paired_vs_same_risk_gate_greedy": matched_gate_comparisons,
        "superiority_checks": rigorous,
        "publication_ready_as_real_ran_superiority": False,
        "limitations": [
            "Toy physical dynamics are generated, not measured and calibrated from a RAN.",
            "Risk margins are calibrated on independent training episodes, not formal coverage guarantees.",
            "Hold fallback is physically simulated, including any adverse consequences, not declared safe.",
            "Multiple controller comparisons require independently replicated validation.",
            "The original static COMMAG dataset cannot independently identify control effects.",
            "PULSE-RAN (original heuristic) is unshielded; its separate gated diagnostic may have different actions and realized outcomes.",
            "The legacy learned-family selector excludes PULSE-RAN by design; validation_ranked_all includes the named baseline but is not a deployable model promotion policy.",
            "CVaR has a stricter tail-risk gate, so comparisons with both the original and a same-gate greedy baseline are provided.",
            "Holm adjustment covers three contenders within each split; split families remain exploratory.",
        ],
    }
    return summary, pd.DataFrame(row_outputs)


def main() -> None:
    p = argparse.ArgumentParser(description="Twin CQL / distributional-CVaR paired simulator research")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--train-episodes", type=int, default=72)
    p.add_argument("--validation-episodes", type=int, default=16)
    p.add_argument("--test-episodes", type=int, default=24)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--output", type=Path, default=Path("results/twin_cql_cvar_study.json"))
    args = p.parse_args()
    summary, episodes = run_twin_study(
        seed=args.seed, train_episodes=args.train_episodes,
        validation_episodes=args.validation_episodes,
        test_episodes=args.test_episodes, steps=args.steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    episode_path = args.output.with_name(args.output.stem + "_episodes.csv")
    episodes.to_csv(episode_path, index=False)
    print(json.dumps({
        "status": "SIMULATOR_STUDY_COMPLETE",
        "output": str(args.output), "episode_csv": str(episode_path),
        "selected_family": summary["validation_selected_family"],
        "test_metrics": summary["final_test_metrics"],
        "rankings": summary["final_test_ranked_by_utility"],
        "paired_vs_pulse_ran": summary["paired_vs_pulse_ran"],
        "superiority_checks": summary["superiority_checks"],
        "disclaimer": summary["scientific_scope"],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
