"""Multi-training-seed reproducible research campaign on the compatible TOY RAN Gym.

Runs the existing six-controller Gymnasium *executed-action* comparison with
independent training seeds, disjoint held-out episode seeds, paired baselines
and run-level inference. Produces self-contained HTML, aggregate and scenario
tables, step/episode traces, model checksums and a conservative evidence gate.

NOT native ns-O-RAN/5G-LENA: the stock ns-O-RAN Gym traffic-steering actions
cannot actuate the frozen COMMAG PRB/scheduler policies.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from agentic_ran.research_ran_gym_final_algorithms import run as run_training_seed
from agentic_ran.research_twin_environment import SCENARIOS

BASELINES = ("ppo_frozen", "pulse_ran", "twin_cql",
             "distributional_cvar", "twin_cql_cvar_hybrid")
METHODS = ("ppo_wsp", *BASELINES)


def _digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sign_flip(values: np.ndarray, *, seed: int, draws: int = 20000) -> float:
    """One-sided paired train-run sign randomization; exact for <=16 runs."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("need >=2 finite independent training-run effects")
    observed = float(values.mean())
    if len(values) <= 16:
        signs = np.arange(1 << len(values), dtype=np.uint32)
        flips = ((signs[:, None] >> np.arange(len(values))) & 1) * 2 - 1
        samples = (flips * values).mean(axis=1)
        return float(np.mean(samples >= observed - 1e-12))
    rng = np.random.default_rng(seed)
    samples = (rng.choice((-1, 1), (draws, len(values))) * values).mean(axis=1)
    return float((1 + np.count_nonzero(samples >= observed - 1e-12)) / (draws + 1))


def _bootstrap(values: np.ndarray, *, seed: int, draws: int = 10000) -> list[float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    return [float(v) for v in np.quantile(values[indices].mean(axis=1), (.025, .975))]


def summarize(episodes: pd.DataFrame, *, train_seeds: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Equal weight to each independent training run; never bootstrap steps."""
    required = {"train_run", "scenario", "seed", "policy", "total_reward",
                "mean_sla_violation_rate", "switch_rate", "receipt_fraction"}
    if not required.issubset(episodes.columns):
        raise ValueError(f"episode evidence missing {sorted(required - set(episodes.columns))}")
    if episodes[list(required & set(episodes.columns))].isna().any().any():
        raise ValueError("nonfinite/missing episode evidence")
    if set(episodes["policy"]) != set(METHODS) or set(episodes["scenario"]) != set(SCENARIOS):
        raise ValueError("missing required methods or held-out scenarios")
    if episodes["train_run"].nunique() != train_seeds:
        raise ValueError("number of independent training runs mismatches campaign")
    counts = episodes.groupby(["train_run", "scenario", "seed"])["policy"].nunique()
    if not counts.eq(len(METHODS)).all():
        raise ValueError("unmatched held-out episodes")
    if episodes.duplicated(["train_run", "scenario", "seed", "policy"]).any():
        raise ValueError("duplicate policy/seed/scenario rows")
    if not episodes["receipt_fraction"].eq(1).all():
        raise ValueError("missing action receipt in toy experiment")
    run_means = episodes.groupby(["train_run", "policy"], as_index=False).agg(
        mean_total_reward=("total_reward", "mean"),
        mean_sla=("mean_sla_violation_rate", "mean"),
        mean_switch_rate=("switch_rate", "mean"),
    )
    pivot = run_means.pivot(index="train_run", columns="policy", values="mean_total_reward")
    risk = run_means.pivot(index="train_run", columns="policy", values="mean_sla")
    if pivot.isna().any().any() or risk.isna().any().any():
        raise ValueError("unbalanced training-run comparison")
    out = {}
    for index, name in enumerate(BASELINES):
        diffs = (pivot["ppo_wsp"] - pivot[name]).to_numpy(dtype=float)
        risk_diffs = (risk["ppo_wsp"] - risk[name]).to_numpy(dtype=float)
        out[name] = {
            "training_seed_clusters": len(diffs),
            "mean_reward_delta": float(diffs.mean()),
            "train_seed_bootstrap_ci95": _bootstrap(diffs, seed=seed + 301 + index),
            "one_sided_train_seed_sign_flip_p": _sign_flip(diffs, seed=seed + 701 + index),
            "per_train_seed_deltas": [float(x) for x in diffs],
            "train_seed_win_rate": float(np.mean(diffs > 0)),
            "mean_sla_delta": float(risk_diffs.mean()),
            "sla_delta_ci95": _bootstrap(risk_diffs, seed=seed + 1101 + index),
        }
    # Sequential Holm adjusted p-values over five predeclared comparisons.
    ordered = sorted(BASELINES, key=lambda name: out[name]["one_sided_train_seed_sign_flip_p"])
    adjusted = 0.0
    for i, name in enumerate(ordered):
        test = out[name]
        adjusted = max(adjusted, min(
            1.0, (len(BASELINES) - i) * test["one_sided_train_seed_sign_flip_p"]
        ))
        test["holm_adjusted_p"] = adjusted
        test["significant_positive_gain"] = bool(
            test["train_seed_bootstrap_ci95"][0] > 0 and adjusted < .05
        )
        # Conservative noninferiority upper-bound for overall toy SLA proxy.
        test["sla_nonregression"] = bool(test["sla_delta_ci95"][1] <= .01)

    per_scenario = episodes.groupby(
        ["train_run", "scenario", "policy"], as_index=False,
    )["total_reward"].mean()
    summary = per_scenario.groupby(["scenario", "policy"], as_index=False).agg(
        mean_total_reward=("total_reward", "mean"),
        across_run_std=("total_reward", "std"),
        training_runs=("train_run", "nunique"),
    )
    return run_means, summary, out


def _html(report: dict[str, Any], means: pd.DataFrame, scenarios: pd.DataFrame) -> str:
    comparisons = report["comparisons"]
    table = []
    for name, values in comparisons.items():
        lower, upper = values["train_seed_bootstrap_ci95"]
        table.append(
            "<tr><td>" + escape(name) + "</td><td>" +
            f'{values["mean_reward_delta"]:.5f}</td><td>[{lower:.5f}, {upper:.5f}]</td><td>' +
            f'{values["holm_adjusted_p"]:.5f}</td><td>' +
            str(values["significant_positive_gain"]) + "</td></tr>"
        )
    bars = []
    group = means.groupby("policy", sort=False)["mean_total_reward"].mean()
    lo = float(group.min())
    hi = float(group.max())
    span = max(hi - min(0.0, lo), 1e-8)
    for i, (method, value) in enumerate(group.items()):
        width = max(0.0, (float(value) - min(0.0, lo)) / span * 475)
        y = 24 + i * 36
        bars.append(
            f'<text x="5" y="{y + 14}" font-size="12">{escape(method)}</text>'
            f'<rect x="190" y="{y}" width="{width:.1f}" height="18" fill="#175fa8"/>'
            f'<text x="{195 + width:.1f}" y="{y + 14}" font-size="12">{value:.4f}</text>'
        )
    svg = '<svg viewBox="0 0 800 270" role="img" aria-label="Mean toy episode reward by controller">' + "".join(bars) + '</svg>'
    return """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Agentic-RAN multi-training-seed Gymnasium research report</title>
<style>body{font:15px/1.55 system-ui,sans-serif;max-width:1100px;margin:auto;padding:2rem;color:#182231}
h1,h2{color:#143559}table{border-collapse:collapse;width:100%;font-size:13px}
td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}
.warning{border-left:4px solid #ae4b16;padding:14px;background:#fff5e9}
svg{max-width:100%}code{background:#eee;padding:2px 4px}</style></head><body>
<h1>Agentic-RAN: multi-training-seed final-controller comparison</h1>
<p class="warning"><strong>Experimental toy environment, NOT IEEE Access-ready physical RAN evidence.</strong>
Stock ns-O-RAN Gym controls handovers; COMMAG-trained scheduler/PRB control has not been
actuated or verified in NR MAC/PHY. No native throughput, packet latency or BLER was measured.</p>
<h2>Experimental protocol</h2><p>Independent training seeds: """ + str(report["training_runs"]) + """;
held-out test seeds per training run: """ + str(report["test_seeds_per_run"]) + """;
scenarios: """ + escape(", ".join(report["scenarios"])) + """.
All six policies use the same action-dependent Gymnasium five-action toy environment.
PPO-WSP is selected on validation seeds, never on held-out episodes.
Primary inference resamples and sign-randomizes independent <em>training runs</em>,
not correlated steps. Multiple comparisons are Holm-adjusted.</p>
<h2>Mean total reward by policy</h2>""" + svg + """
<h2>Paired PPO-WSP comparisons</h2><table>
<tr><th>Baseline</th><th>Mean delta</th><th>Train-seed bootstrap 95% CI</th>
<th>Holm p</th><th>Positive gain established</th></tr>""" + "".join(table) + """</table>
<h2>Per-scenario reward</h2>""" + scenarios.to_html(index=False, border=0, float_format=lambda x:f"{x:.5f}") + """
<h2>Training-run detail</h2>""" + means.to_html(index=False, border=0, float_format=lambda x:f"{x:.5f}") + """
<h2>Evidence and limitations</h2><p>Test trajectories are action-dependent; episode
return is from the simulator. Native PRB actuation receipts, NR PHY/BWP grids,
per-slice real throughput and latency, true 5QI SLAs and external-domain validation
are unavailable. Software receipts are not hardware attestations.
Inspect run manifest, checksums, CSV traces, and negative comparisons before citation.</p>
</body></html>"""


def campaign(
    output: Path, *, base_seed: int = 2026,
    training_runs: int = 10, test_seeds: int = 40,
    ppo_updates: int = 48, episodes_per_update: int = 12,
    logged_episodes: int = 120, validation_episodes: int = 24,
    steps: int = 32, wsp_rounds: int = 3, critic_iterations: int = 5,
    smoke: bool = False,
) -> dict[str, Any]:
    if training_runs < 2:
        raise ValueError("at least two independent training seeds required")
    if min(test_seeds, steps, validation_episodes, ppo_updates, episodes_per_update,
           logged_episodes, critic_iterations) < 1:
        raise ValueError("all counts must be positive")
    if not smoke and (training_runs < 8 or test_seeds < 20 or steps < 20
                      or validation_episodes < 12 or logged_episodes < 72):
        raise ValueError("research preset requires >=8 train runs, >=20 test seeds, >=20 steps")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("campaign output must be a new empty directory: avoid overwriting evidence")
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema": "agentic-ran-gym-multiseed-v1",
        "base_seed": base_seed, "training_runs": training_runs, "test_seeds": test_seeds,
        "ppo_updates": ppo_updates, "episodes_per_update": episodes_per_update,
        "logged_episodes": logged_episodes, "validation_episodes": validation_episodes,
        "steps": steps, "wsp_rounds": wsp_rounds, "critic_iterations": critic_iterations,
        "preset": "smoke" if smoke else "research", "methods": list(METHODS),
        "scenarios": list(SCENARIOS),
        "inference_unit": "independent_training_run",
    }
    (output / "protocol_frozen.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    episodes = []
    steps_frames = []
    individual = []
    for i in range(training_runs):
        train_seed = base_seed + 1_000_003 * i
        print(f"[campaign] train_run={i} seed={train_seed} TRAIN/VALIDATE/TEST", flush=True)
        run_dir = output / f"training_run_{i:02d}"
        summary = run_training_seed(
            run_dir, seed=train_seed, ppo_updates=ppo_updates,
            episodes_per_update=episodes_per_update,
            logged_episodes=logged_episodes,
            validation_episodes=validation_episodes, test_seeds=test_seeds,
            steps=steps, wsp_rounds=wsp_rounds, critic_iterations=critic_iterations,
        )
        if summary.get("publication_superiority_ready") is not False:
            raise RuntimeError("toy-only provenance unexpectedly changed")
        episode_file = run_dir / "gym_final_episodes.csv"
        step_file = run_dir / "gym_final_steps.csv.gz"
        if not all(x.exists() for x in (episode_file, step_file, run_dir / "ppo_wsp.pt")):
            raise RuntimeError("training run missed required evidence")
        ep = pd.read_csv(episode_file)
        st = pd.read_csv(step_file)
        ep.insert(0, "train_run", i)
        st.insert(0, "train_run", i)
        episodes.append(ep)
        steps_frames.append(st)
        individual.append({
            "train_run": i, "training_seed": train_seed,
            "test_seeds": summary["heldout_test_seeds"],
            "models": {
                key: _digest(run_dir / key) for key in ("ppo_frozen.pt", "ppo_wsp.pt")
            },
            "mean_rewards": {name: metrics["mean_total_reward"] for name, metrics in summary["methods"].items()},
        })
    full_episodes = pd.concat(episodes, ignore_index=True)
    full_steps = pd.concat(steps_frames, ignore_index=True)
    run_means, scenarios, comparisons = summarize(
        full_episodes, train_seeds=training_runs, seed=base_seed
    )
    report = {
        "schema": "agentic-ran-gym-multiseed-evidence-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "backend": "agentic-ran-interventional-toy-gymnasium",
        "training_runs": training_runs,
        "test_seeds_per_run": test_seeds,
        "scenarios": list(SCENARIOS),
        "total_executed_episodes": len(full_episodes),
        "total_executed_steps": len(full_steps),
        "train_test_separation_declared": True,
        "matched_actions_are_toy_software_receipts": True,
        "all_six_algorithms_evaluated_on_identical_action_space": True,
        "heldout_reference": "distinct seeds for every training-run and scenario",
        "comparisons": comparisons,
        "runs": individual,
        "all_comparisons_show_significant_positive_gain": all(
            x["significant_positive_gain"] for x in comparisons.values()
        ),
        "physical_network_sla_evidence": False,
        "frozen_commag_prb_policy_actuated_in_native_gym": False,
        "publication_ready_for_ieee_access": False,
        "study_artifacts_complete": True,
        "manual_novelty_and_methods_review_required": True,
        "claim_limitations": [
            "Toy action-dependent Gym only; not native ns-O-RAN, 5G-LENA or Colosseum",
            "No real BWP/PRB scheduler actuation, real throughput, latency or PHY SLA measures",
            "Multiple training runs do not replace independent physical environment validation",
            "Favorable simulated reward does not imply generalization to a physical network",
        ],
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
    }
    output_paths = {
        "episodes": output / "all_heldout_episodes.csv.gz",
        "steps": output / "all_heldout_steps.csv.gz",
        "run_means": output / "training_run_means.csv",
        "scenarios": output / "scenario_means.csv",
        "summary": output / "campaign_summary.json",
        "html": output / "research_report.html",
    }
    full_episodes.to_csv(output_paths["episodes"], index=False, compression="gzip")
    full_steps.to_csv(output_paths["steps"], index=False, compression="gzip")
    run_means.to_csv(output_paths["run_means"], index=False)
    scenarios.to_csv(output_paths["scenarios"], index=False)
    output_paths["summary"].write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    output_paths["html"].write_text(_html(report, run_means, scenarios), encoding="utf-8")
    hashes = {name: _digest(path) for name, path in output_paths.items()}
    (output / "artifact_sha256.json").write_text(
        json.dumps(hashes, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "training_runs": training_runs, "test_seeds_per_run": test_seeds,
        "executed_episodes": len(full_episodes), "executed_steps": len(full_steps),
        "all_significant_vs_baselines": report["all_comparisons_show_significant_positive_gain"],
        "publication_ready_for_ieee_access": False,
        "output": str(output),
    }, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Full toy-RAN Gym training/evidence campaign; never physical RAN validation")
    parser.add_argument("--output", type=Path, default=Path("results/ran-gym-campaign"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--training-runs", type=int, default=10)
    parser.add_argument("--test-seeds", type=int, default=40)
    parser.add_argument("--ppo-updates", type=int, default=48)
    parser.add_argument("--episodes-per-update", type=int, default=12)
    parser.add_argument("--logged-episodes", type=int, default=120)
    parser.add_argument("--validation-episodes", type=int, default=24)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--wsp-rounds", type=int, default=3)
    parser.add_argument("--critic-iterations", type=int, default=5)
    parser.add_argument("--smoke", action="store_true", help="small CI functional test, not publication evidence")
    args = parser.parse_args()
    campaign(args.output, base_seed=args.seed,
             training_runs=args.training_runs, test_seeds=args.test_seeds,
             ppo_updates=args.ppo_updates, episodes_per_update=args.episodes_per_update,
             logged_episodes=args.logged_episodes, validation_episodes=args.validation_episodes,
             steps=args.steps, wsp_rounds=args.wsp_rounds, critic_iterations=args.critic_iterations,
             smoke=args.smoke)


if __name__ == "__main__":
    main()
