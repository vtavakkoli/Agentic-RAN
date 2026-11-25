"""Fresh-seed PA-SPI research campaign; executed TOY actions only.

The original six-policy campaign remains a historical/exploratory comparison.
This experiment freezes the primary comparison and safety margin before
training. It saves per-run artifacts to avoid aggregating millions of steps
in memory. Native NR and logged COMMAG remain separate evidence layers.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import html
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from agentic_ran.research_pulse_anchored_spi import (
    fit, make_world, TEST_SCENARIOS, TRAIN_SCENARIOS, N,
)
from agentic_ran.research_pulse_ran import select_pulse_ran
from agentic_ran.research_ran_gym_campaign import _bootstrap, _sign_flip
from agentic_ran.research_ppo_wsp import train_ppo, _evaluations, _search
from agentic_ran.research_twin_environment import ACTIONS, SLA_QUEUE_LIMITS, generate_logged_transitions
from agentic_ran.research_twin_offline import (
    choose_action, fit_calibrated_twin, fit_linear_cql, fit_quantile_td,
)

PRIMARY = "pa_spi"
REFERENCE = "pulse_ran"
SLA_TOLERANCE = .01
METHODS = (
    PRIMARY, REFERENCE, "pulse_ran_gated", "ppo_frozen", "ppo_wsp",
    "twin_cql", "distributional_cvar", "twin_cql_cvar_hybrid",
    "balanced", "round_robin", "channel_queue_greedy",
    "pa_no_lcb", "pa_no_support", "pa_no_risk", "pa_no_switch",
    "pa_no_margin", "pa_no_gates",
)
ABLATIONS = tuple(m for m in METHODS if m.startswith("pa_no_"))
METRICS = ("reward", "sla", "sla_embb", "sla_mmtc", "sla_urllc",
           "switch_rate", "max_violation_streak", "override_rate",
           "unsupported_rate", "decision_p95_ms")
BASE_SEED = 510_000_019  # disjoint from the previously inspected 30M+ test seeds


def digest(path):
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def greedy(state, previous, t):
    """Strong nonlearned baseline; explicit myopic queue/channel model."""
    q, d, ch = np.split(state, 3)
    backlog = np.maximum(0., q + d - 3.0 * ACTIONS * ch / (1 + .055 * q))
    cost = (backlog**2) @ np.array([.065, .04, .24])
    cost += (backlog > SLA_QUEUE_LIMITS) @ np.array([.20, .12, .65])
    cost += .055 * (np.arange(N) != previous) + .028 * (ACTIONS**2).sum(axis=1)
    return int(np.argmin(cost))


def evaluate(policy, scenario, seed, steps, *, trace=False):
    """Use real Gymnasium Discrete.contains and explicit executed toy receipt.

    The receipt is from the toy state machine after step; it is not a
    physical scheduler acknowledgment. Never reuse a reward between policies.
    """
    from gymnasium.spaces import Discrete
    env = make_world(seed, scenario)
    initial = sha256(env.state.tobytes()).hexdigest()
    space = Discrete(N)
    rewards, flags, switches, latencies, audits, rows = [], [], [], [], [], []
    streak, max_streak = 0, 0
    for t in range(steps):
        previous = env.previous_action
        start = time.perf_counter_ns()
        selection = policy(env.state.copy(), previous, t)
        latencies.append((time.perf_counter_ns() - start) / 1e6)
        action, audit = selection if isinstance(selection, tuple) else (selection, {})
        if not space.contains(action):
            raise ValueError("policy produced incompatible allocation")
        _, reward, observed = env.step(int(action))
        if env.previous_action != action or not np.isfinite(reward):
            raise RuntimeError("missing toy action receipt or nonfinite reward")
        rewards.append(float(reward))
        flags.append(observed.astype(float))
        switches.append(t > 0 and action != previous)
        streak = streak + 1 if observed.any() else 0
        max_streak = max(max_streak, streak)
        audits.append(audit)
        if trace:
            rows.append({"step": t, "action": int(action), "receipt": True,
                         "reward": float(reward), "sla_embb": bool(observed[0]),
                         "sla_mmtc": bool(observed[1]), "sla_urllc": bool(observed[2]),
                         "action_changed": bool(switches[-1]), "decision_ms": latencies[-1],
                         **audit})
    per_slice = np.asarray(flags).mean(axis=0)
    return {"scenario": scenario, "seed": seed, "initial_hash": initial,
            "steps": steps, "reward": float(sum(rewards)), "sla": float(per_slice.mean()),
            "sla_embb": float(per_slice[0]), "sla_mmtc": float(per_slice[1]),
            "sla_urllc": float(per_slice[2]), "switch_rate": float(np.mean(switches)),
            "max_violation_streak": max_streak,
            "override_rate": float(np.mean([a.get("reason") == "override" for a in audits])),
            "unsupported_rate": float(np.mean([a.get("state_support") is False for a in audits])),
            "decision_p95_ms": float(np.quantile(latencies, .95)),
            "receipt_fraction": 1.0}, rows


def _reference(s, p, t):
    return select_pulse_ran(s)


def _run_one(args):
    output, run, cfg = args
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        return _run_one_inner(output, run, cfg)


def _run_one_inner(output, run, cfg):
    seed = BASE_SEED + run * 1_000_003
    folder = output / f"training_run_{run:02d}"
    folder.mkdir(exist_ok=False)
    steps = cfg["steps"]
    print(f"[PA-SPI] training run {run} seed={seed}", flush=True)
    spi, calibration = fit(seed, cfg["logged_episodes"], cfg["validation_episodes"], steps)
    # A third disjoint validation partition selects the override margin,
    # not the risk tolerance or held-out reward. All choices freeze first.
    v = seed + 20_000_000
    pairs = [(TRAIN_SCENARIOS[i % 3], v + i * 997) for i in range(cfg["validation_episodes"])]
    ref_rows = [evaluate(_reference, s, k, steps)[0] for s, k in pairs]
    safety = ("sla_embb", "sla_mmtc", "sla_urllc")
    reference = {m: float(np.mean([r[m] for r in ref_rows])) for m in ("reward", *safety)}
    best_score, chosen, validation = reference["reward"], 1e9, []
    for margin in (.0, .02, .05, .10, .25):
        spi.margin = margin
        rows = [evaluate(spi.decide, s, k, steps)[0] for s, k in pairs]
        means = {m: float(np.mean([r[m] for r in rows])) for m in reference}
        safe = all(means[m] <= reference[m] + SLA_TOLERANCE for m in safety)
        validation.append({"margin": margin, "means": means, "empirical_sla_ok": safe})
        if safe and means["reward"] > best_score + 1e-8:
            best_score, chosen = means["reward"], margin
    spi.margin = chosen
    # Baselines fit on the same original three scenario family as PA-SPI.
    # Their information budgets differ; report actual budgets, not matched
    # training compute. PA-SPI uses simulator branching (not offline RL).
    frozen = train_ppo(seed=seed + 100_000, updates=cfg["ppo_updates"],
                       episodes_per_update=cfg["episodes_per_update"], steps=steps)
    original = _evaluations(frozen.choose, seed=v, count=cfg["validation_episodes"],
                            steps=steps, scenarios=TRAIN_SCENARIOS)
    edited, edits = _search(frozen, original, seed=seed + 310, validation_seed=v,
                            val_episodes=cfg["validation_episodes"], steps=steps,
                            rounds=cfg["wsp_rounds"])
    logged = generate_logged_transitions(seed=seed + 500_000,
                                         episodes=cfg["logged_episodes"], steps=steps)
    twin = fit_calibrated_twin(logged)
    cql = fit_linear_cql(logged, alpha=.05, iterations=cfg["critic_iterations"], steps_per_iter=35)
    quantile = fit_quantile_td(logged, iterations=cfg["critic_iterations"], steps_per_iter=40)
    def gated(mode):
        def policy(s, p, t):
            a = choose_action(mode, s, p, t, twin=twin, cql=cql, quantile=quantile,
                              risk_weight=.35, blend_weight=.5)
            # Executed hold-last is an intervention, not certified safe.
            return (int(p), {"reason": "hold_last"}) if a == -1 else int(a)
        return policy
    policies = {
        PRIMARY: spi.decide, REFERENCE: _reference, "pulse_ran_gated": gated("pulse_ran_gated"),
        "ppo_frozen": frozen.choose, "ppo_wsp": edited.choose, "twin_cql": gated("twin_cql"),
        "distributional_cvar": gated("distributional_cvar"),
        "twin_cql_cvar_hybrid": gated("twin_cql_cvar_hybrid"),
        "balanced": lambda s, p, t: 0, "round_robin": lambda s, p, t: t % N,
        "channel_queue_greedy": greedy,
    }
    for mode in ABLATIONS:
        # no_margin is the learned point-estimate ranking without margin;
        # the learned representation remains anchored, explicitly labeled.
        key = "pa_no_anchor" if mode == "pa_no_margin" else mode
        policies[mode] = lambda s, p, t, key=key: spi.decide(s, p, t, mode=key)
    # Freeze models and record hashes before opening the fresh test episodes.
    joblib.dump(spi, folder / "pa_spi.joblib")
    torch.save(frozen.state_dict(), folder / "ppo_frozen.pt")
    torch.save(edited.state_dict(), folder / "ppo_wsp.pt")
    write_json(folder / "frozen_selection.json", {
        "seed": seed, "calibration": calibration, "validation_seed": v,
        "validation_reference": reference, "validation": validation,
        "chosen_margin": chosen, "validation_defaulted_to_reference": chosen == 1e9,
        "wsp_accepted_edits": edits,
        "models": {p.name: digest(p) for p in folder.glob("*.joblib")} |
                  {p.name: digest(p) for p in folder.glob("*.pt")},
    })
    rows, audits = [], []
    for scenario in TEST_SCENARIOS:
        for i in range(cfg["test_seeds"]):
            k = seed + 50_000_000 + i * 997
            for name in METHODS:
                row, trace = evaluate(policies[name], scenario, k, steps, trace=True)
                rows.append({"train_run": run, "method": name, **row})
                audits.extend({"method": name, "scenario": scenario, "seed": k, **r} for r in trace)
    pd.DataFrame(rows).to_csv(folder / "episodes.csv.gz", index=False, compression="gzip")
    pd.DataFrame(audits).to_csv(folder / "steps.csv.gz", index=False, compression="gzip")
    write_json(folder / "hashes.json", {p.name: digest(p) for p in folder.iterdir() if p.is_file()})
    return str(folder / "episodes.csv.gz")


def summarize(frame, training_runs, seed=BASE_SEED):
    keys = ["train_run", "scenario", "seed"]
    expected = set(METHODS)
    if set(frame.method) != expected or set(frame.scenario) != set(TEST_SCENARIOS):
        raise ValueError("incomplete methods or scenario coverage")
    if frame.duplicated(keys + ["method"]).any() or frame.train_run.nunique() != training_runs:
        raise ValueError("invalid pairing")
    if not frame.groupby(keys).method.nunique().eq(len(METHODS)).all():
        raise ValueError("unmatched held-out runs")
    if not frame.groupby(keys).initial_hash.nunique().eq(1).all():
        raise ValueError("unmatched initial conditions")
    if not frame.receipt_fraction.eq(1).all() or not np.isfinite(frame[list(METRICS)]).all().all():
        raise ValueError("invalid trace-derived evidence")
    # Every scenario/seed gets equal weight within each independent run.
    counts = frame.groupby(["train_run", "scenario", "method"]).size()
    if counts.nunique() != 1:
        raise ValueError("imbalanced held-out counts")
    means = frame.groupby(["train_run", "method"])[list(METRICS)].mean().reset_index()
    ranks = means.groupby("method")[list(METRICS)].mean().reset_index().sort_values("reward", ascending=False)
    tails = frame.groupby(["train_run", "method"]).reward.agg(
        median="median", p05=lambda x: float(x.quantile(.05)),
        cvar05=lambda x: float(np.sort(x)[:max(1, int(np.ceil(.05*len(x))))].mean()),
    ).reset_index()
    ranks = ranks.merge(tails.groupby("method")[["median", "p05", "cvar05"]].mean().reset_index(), on="method")
    scenarios = frame.groupby(["train_run", "scenario", "method"])[list(METRICS)].mean().reset_index()
    scenarios = scenarios.groupby(["scenario", "method"])[list(METRICS)].mean().reset_index()
    tests = {}
    for j, method in enumerate(m for m in METHODS if m != PRIMARY):
        delta = means.pivot(index="train_run", columns="method", values="reward")
        d = (delta[PRIMARY] - delta[method]).to_numpy()
        safety = {}
        for i, m in enumerate(("sla_embb", "sla_mmtc", "sla_urllc")):
            p = means.pivot(index="train_run", columns="method", values=m)
            safety[m] = _bootstrap((p[PRIMARY]-p[method]).to_numpy(), seed=seed+100+j*10+i)
        tests[method] = {"reward_delta": float(d.mean()), "reward_ci95": _bootstrap(d, seed=seed+j),
                         "one_sided_p": _sign_flip(d, seed=seed+j+500),
                         "run_win_rate": float(np.mean(d > 0)), "per_slice_sla_delta_ci95": safety,
                         "sla_noninferiority": all(ci[1] <= SLA_TOLERANCE for ci in safety.values())}
    adj = 0.
    ordered = sorted(tests, key=lambda m: tests[m]["one_sided_p"])
    for i, name in enumerate(ordered):
        adj = max(adj, min(1., tests[name]["one_sided_p"]*(len(ordered)-i)))
        tests[name]["holm_adjusted_p"] = adj
    return ranks, scenarios, means, tests


def run(output: Path, *, training_runs=20, test_seeds=100, steps=32,
        logged_episodes=120, validation_episodes=24, ppo_updates=48,
        episodes_per_update=12, wsp_rounds=3, critic_iterations=5,
        workers=2, smoke=False):
    if min(training_runs, test_seeds, steps, logged_episodes, validation_episodes,
           ppo_updates, episodes_per_update, wsp_rounds, critic_iterations, workers) < 1:
        raise ValueError("all counts must be positive")
    if training_runs < 2 or test_seeds < 4 or steps < 5 or logged_episodes < 12 or validation_episodes < 4:
        raise ValueError("insufficient smoke sample")
    if not smoke and (training_runs < 20 or test_seeds < 100 or steps < 20 or logged_episodes < 72 or validation_episodes < 12):
        raise ValueError("confirmatory preset requires >=20 runs, >=100 seeds and >=20 steps")
    output.mkdir(parents=True, exist_ok=False)
    cfg = dict(training_runs=training_runs, test_seeds=test_seeds, steps=steps,
               logged_episodes=logged_episodes, validation_episodes=validation_episodes,
               ppo_updates=ppo_updates, episodes_per_update=episodes_per_update,
               wsp_rounds=wsp_rounds, critic_iterations=critic_iterations, workers=workers, smoke=smoke)
    write_json(output / "protocol_frozen.json", {**cfg, "base_seed": BASE_SEED,
               "primary_candidate": PRIMARY, "primary_reference": REFERENCE,
               "scenarios": list(TEST_SCENARIOS), "methods": list(METHODS),
               "sla_margin": SLA_TOLERANCE, "inference_unit": "training_run",
               "training_scenarios": list(TRAIN_SCENARIOS),
               "previous_campaign_is_exploratory": True,
               "train_data_budget_note": "PA-SPI branches every action at sampled training states; baseline training compute is not matched",
               "claim_scope": "toy interventional only; no physical NR, no COMMAG intervention"})
    tasks = [(output, i, cfg) for i in range(training_runs)]
    if workers == 1:
        paths = [_run_one(t) for t in tasks]
    else:
        from multiprocessing import get_context
        with ProcessPoolExecutor(max_workers=min(workers, training_runs), mp_context=get_context("spawn")) as pool:
            paths = list(pool.map(_run_one, tasks))
    frame = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    expected = training_runs * test_seeds * len(TEST_SCENARIOS) * len(METHODS)
    if len(frame) != expected or not frame.steps.eq(steps).all():
        raise ValueError("incomplete experiment")
    ranks, scenarios, means, tests = summarize(frame, training_runs)
    for name, data in (("rankings", ranks), ("scenario_means", scenarios), ("training_run_means", means)):
        data.to_csv(output / f"{name}.csv", index=False)
    primary = tests[REFERENCE]
    result = {"schema": "agentic-ran-pa-spi-v1", "smoke": smoke,
              "training_runs": training_runs, "test_seeds_per_run": test_seeds,
              "executed_episodes": len(frame), "executed_steps": int(frame.steps.sum()),
              "winner_by_mean_toy_return": str(ranks.iloc[0].method),
              "primary_comparison": primary, "all_comparisons": tests,
              "primary_superiority_and_sla_noninferiority": bool(
                  not smoke and primary["reward_ci95"][0] > 0 and
                  primary["one_sided_p"] < .05 and primary["sla_noninferiority"]),
              "publication_ready_for_ieee_access": False,
              "physical_nr_evidence": False, "rankings": ranks.to_dict("records"),
              "limitations": ["empirical calibration not formal safety certification",
                              "branch simulator access differs from offline baseline budgets",
                              "fixed fallback may itself violate SLA",
                              "shifted toy dynamics are not independent physical validation",
                              "per-episode decision p95 averaged across episodes; not RIC latency"]}
    write_json(output / "summary.json", result)
    report = ("<!doctype html><meta charset='utf-8'><title>PA-SPI evidence</title>"
              "<style>body{font:15px system-ui;margin:2em;max-width:1400px}table{border-collapse:collapse}"
              "td,th{padding:7px;border:1px solid #ddd}td{text-align:right}</style>"
              "<h1>PA-SPI fresh-seed research report</h1>"
              "<p>TOY ONLY. No claim of physical NR or IEEE acceptance. "
              "Primary comparison is PA-SPI vs frozen PULSE; negative results retained.</p>"
              f"<h2>Winner by mean return: {html.escape(result['winner_by_mean_toy_return'])}</h2>"
              + ranks.to_html(index=False, float_format=lambda x: f"{x:.5f}")
              + "<h2>Primary and secondary comparisons</h2><pre>"
              + html.escape(json.dumps(tests, indent=2)) + "</pre><h2>Scenario breakdown</h2>"
              + scenarios.to_html(index=False, float_format=lambda x: f"{x:.5f}"))
    (output / "report.html").write_text(report, encoding="utf-8")
    write_json(output / "artifact_sha256.json", {str(p.relative_to(output)): digest(p)
               for p in sorted(output.rglob("*")) if p.is_file()})
    print(json.dumps({k: v for k, v in result.items() if k not in ("rankings", "all_comparisons")}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    for name, default in (("training-runs", 20), ("test-seeds", 100), ("steps", 32),
                          ("logged-episodes", 120), ("validation-episodes", 24),
                          ("ppo-updates", 48), ("episodes-per-update", 12),
                          ("wsp-rounds", 3), ("critic-iterations", 5), ("workers", 2)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--smoke", action="store_true")
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
