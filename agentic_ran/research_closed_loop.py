"""Action-dependent JAGC research: randomized training and paired executed outcomes.

Runs an EXTERNAL, persistent ns-3/5G-LENA or Colosseum controller endpoint.
Static COMMAG CSV rewards are never reused as counterfactual outcomes.
The original COMMAG-trained Agentic-RAN model remains frozen and is visibly
projected to physical PRB capacity for an executable matched baseline.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import re
import shlex

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from agentic_ran.offline_policy import STATE_COLUMNS
from agentic_ran.publication_science import _choose_from_precomputed, _policy_precompute
from agentic_ran.research_ns3_session import ExternalClosedLoopSession
from agentic_ran.research_physical_prb import (
    CellBudget, JointAction, SLICES, SCHEDULERS, discrete_joint_catalog,
    project_reference_action,
)

TRAIN_SCENARIOS = ("normal", "congested", "bursty")
FEATURES = ("dl_buffer_bytes", "dl_cqi", "ul_sinr", "requested_prbs",
            "grant_ratio", "dl_errors_pct", "num_ues")
ACTION_FORMAT = re.compile(r"^(round_robin|waterfilling|proportional_fair):prb=(\d+)$")


def encode(obs: dict, action: JointAction, budget: CellBudget) -> np.ndarray:
    """Pre-action features only; action is expressed in PHY PRB fractions."""
    action.validate(budget)
    values = []
    for sl in SLICES:
        measures = obs["slices"][sl]["metrics"]
        for f in FEATURES:
            if f not in measures:
                raise ValueError(f"missing measured pre-action feature: {sl}/{f}")
            v = float(measures[f])
            if not np.isfinite(v):
                raise ValueError("nonfinite pre-action state")
            values.append(math.log1p(max(v, 0)) if f in ("dl_buffer_bytes", "requested_prbs", "num_ues") else v)
    return np.asarray(values + [x / budget.usable_prbs for x in action.prbs]
                      + [SCHEDULERS.index(x) / 2 for x in action.schedulers], dtype=float)


class FrozenOriginal:
    """Uses the original crossfit policy bundle without retraining.

    The issued per-slice action labels may exceed shared physical capacity.
    The nearest feasible projection is applied EXPLICITLY and recorded; this
    is a capacity-projected baseline rather than as-issued legacy control.
    """

    def __init__(self, path: Path, fold: str):
        if not path.is_file():
            raise FileNotFoundError(f"frozen COMMAG models missing: {path}")
        # joblib uses pickle: load ONLY artifacts generated locally/trusted.
        payload = joblib.load(path)
        if payload.get("crossfit_strategy") != "experiment_role_swap_crossfit":
            raise ValueError("not a frozen Agentic-RAN publication model artifact")
        item = payload["folds"][fold]
        self.bundle = item["policy_bundle"]
        calibration = item["calibrated_controller"]
        self.config = replace(
            self.bundle.config,
            sla_threshold=float(calibration["sla_threshold"]),
            uncertainty_threshold=float(calibration["uncertainty_threshold"]),
            ood_threshold=float(calibration["ood_threshold"]),
            planning_weight=float(calibration["planning_weight"]),
        )
        self.margin = float(calibration["switch_margin"])
        self.fold = fold
        self.path = str(path)

    def choose(self, obs: dict, budget: CellBudget) -> tuple[JointAction, bool]:
        records = []
        for sl in SLICES:
            measures = obs["slices"][sl]["metrics"]
            missing = set(STATE_COLUMNS) - set(measures)
            if missing:
                raise ValueError(f"frozen original requires measured {sl} fields: {sorted(missing)}")
            values = {f: float(measures[f]) for f in STATE_COLUMNS}
            scheduler = int(round(values["scheduler_code"]))
            if scheduler not in range(len(SCHEDULERS)):
                raise ValueError("unrecognized simulator scheduler code")
            records.append({
                **values, **{f"next_{key}": val for key, val in values.items()},
                "slice_type": sl, "timestamp_s": obs["step"],
                "episode_id": f"online/{sl}", "done": False,
                "action": f"{SCHEDULERS[scheduler]}:prb={int(round(values['slice_prb']))}",
            })
        # next_* placeholders copy CURRENT metrics to satisfy offline
        # dataframe schema. No future post-action telemetry is consulted.
        frame = pd.DataFrame(records)
        pre = _policy_precompute(self.bundle, frame)
        issued = _choose_from_precomputed(
            self.bundle, pre, self.config, self.margin,
        )["selected_action"].astype(str).tolist()
        match = [ACTION_FORMAT.fullmatch(s) for s in issued]
        if any(m is None for m in match):
            raise ValueError("frozen original issued an unknown scheduler/PRB label")
        assert all(m is not None for m in match)
        return project_reference_action(
            tuple(int(m.group(2)) for m in match), budget,
            tuple(m.group(1) for m in match),
        )


@dataclass
class Transition:
    episode: int
    seed: int
    scenario: str
    step: int
    policy: str
    action: JointAction
    features: np.ndarray
    realized_reward: float
    urllc_queue_exceeded: bool
    original_projected: bool


class SupportedAdvantage:
    """Bootstrap ensemble fitted on actual randomized post-action rewards.

    A kNN radius is an overlap HEURISTIC, not causal support proof.
    Ensemble dispersion is not a calibrated formal advantage LCB.
    """

    def fit(self, rows: list[Transition], *, seed: int):
        if len({r.action.prbs for r in rows}) < 2 or not any(
            len({r.action.prbs for r in rows if r.episode == e}) > 1
            for e in {r.episode for r in rows}
        ):
            raise ValueError("real within-episode action variation required for advantage learning")
        episodes = np.unique([r.episode for r in rows])
        if len(episodes) < 3:
            raise ValueError("require >=3 independent training episodes")
        x = np.asarray([r.features for r in rows])
        y = np.asarray([r.realized_reward for r in rows])
        self.scaler = StandardScaler().fit(x)
        scaled = self.scaler.transform(x)
        self.neighbors = NearestNeighbors(n_neighbors=min(3, len(rows))).fit(scaled)
        self.radius = max(float(np.quantile(self.neighbors.kneighbors(scaled)[0][:, -1], .95)) * 1.25, .25)
        ids = np.asarray([r.episode for r in rows])
        rng = np.random.default_rng(seed)
        self.models = []
        for i in range(4):
            sampled = rng.choice(episodes, size=len(episodes), replace=True)
            idx = np.concatenate([np.flatnonzero(ids == e) for e in sampled])
            model = ExtraTreesRegressor(
                n_estimators=48, min_samples_leaf=2,
                random_state=seed + 251 * i, n_jobs=1,
            ).fit(scaled[idx], y[idx])
            self.models.append(model)
        return self

    def predict(self, x: np.ndarray):
        scaled = self.scaler.transform(x)
        scores = np.asarray([model.predict(scaled) for model in self.models])
        covered = self.neighbors.kneighbors(scaled)[0][:, -1] <= self.radius
        return scores.mean(axis=0), scores.std(axis=0), covered


@dataclass(frozen=True)
class Gate:
    name: str
    uncertainty_multiplier: float
    margin: float


GATES = (
    Gate("frozen_original", 0, math.inf),
    Gate("supported_advantage_1", 1, .005),
    Gate("supported_advantage_2", 2, .005),
    Gate("supported_advantage_strict", 2, .015),
)


def candidate_action(obs, baseline, catalog, budget, model, gate, residual):
    if gate.name == "frozen_original":
        return baseline, False
    candidates = [baseline] + [a for a in catalog if a != baseline]
    x = np.asarray([encode(obs, a, budget) for a in candidates])
    mean, std, supported = model.predict(x)
    if not supported[0]:
        return baseline, False
    for j in sorted(range(1, len(candidates)), key=lambda i: mean[i], reverse=True):
        if not supported[j]:
            continue
        lower_delta = float(
            mean[j] - mean[0] -
            gate.uncertainty_multiplier * np.hypot(std[j], std[0]) -
            residual
        )
        if lower_delta >= gate.margin:
            return candidates[j], True
    return baseline, False


def episode(
    session: ExternalClosedLoopSession, frozen: FrozenOriginal, catalog: list[JointAction],
    *, seed: int, scenario: str, episode_id: int, steps: int,
    policy: str, model=None, gate=None, residual: float = 0,
) -> tuple[list[Transition], dict]:
    obs = session.reset(seed, scenario)
    rng = np.random.default_rng(seed + 20113)
    recorded = []
    promotions = 0
    projections = 0
    for t in range(steps):
        original, projected = frozen.choose(obs, session.budget)
        projections += int(projected)
        if policy == "exploration":
            selected = catalog[int(rng.integers(len(catalog)))]
        elif policy == "original":
            selected = original
        elif policy == "jagc":
            selected, promoted = candidate_action(
                obs, original, catalog, session.budget, model, gate, residual,
            )
            promotions += int(promoted)
        else:
            raise ValueError("unknown policy")
        before = encode(obs, selected, session.budget)
        next_obs, reward, done = session.step(selected)
        # The threshold is an explicit experimental QUEUE proxy in bytes;
        # it is not equivalent to a certified URLLC 5QI latency requirement.
        violation = float(next_obs["slices"]["URLLC"]["metrics"]["dl_buffer_bytes"]) > 9000
        recorded.append(Transition(
            episode_id, seed, scenario, t, policy, selected, before,
            reward, violation, projected,
        ))
        obs = next_obs
        if done:
            break
    return recorded, {
        "policy": policy, "seed": seed, "scenario": scenario,
        "mean_reward": float(np.mean([r.realized_reward for r in recorded])),
        "urllc_queue_exceedance": float(np.mean([r.urllc_queue_exceeded for r in recorded])),
        "projection_rate": projections / len(recorded),
        "promotion_rate": promotions / len(recorded),
        "within_episode_action_variants": len({r.action.prbs for r in recorded}),
        "steps": len(recorded),
    }


def paired_statistics(rows, seed):
    data = pd.DataFrame(rows).pivot(
        index=["scenario", "seed"], columns="policy", values="mean_reward",
    ).dropna(subset=["original", "jagc"])
    delta = (data["jagc"] - data["original"]).to_numpy(dtype=float)
    if len(delta) < 2:
        raise ValueError("need >=2 paired executed test episodes")
    rng = np.random.default_rng(seed)
    bootstrap = [float(np.mean(rng.choice(delta, size=len(delta), replace=True))) for _ in range(2000)]
    more = sum(
        abs(float(np.mean(delta * rng.choice((-1., 1.), size=len(delta))))) >=
        abs(float(np.mean(delta))) - 1e-12 for _ in range(5000)
    )
    original = float(data["original"].mean())
    jagc = float(data["jagc"].mean())
    return {
        "paired_scenario_seed_episodes": len(delta),
        "original_realized_mean_reward": original,
        "jagc_realized_mean_reward": jagc,
        "mean_paired_delta": float(np.mean(delta)),
        "relative_gain": (jagc-original)/abs(original) if original else None,
        "episode_bootstrap_95ci": [float(x) for x in np.quantile(bootstrap, [.025, .975])],
        "paired_sign_randomization_p": (more + 1) / 5001,
        "estimand": "executed_simulator_action_reward_not_static_COMMAG_DM",
    }


def run_study(
    sim_command: list[str], models: Path, output: Path, *,
    require_real: bool = True, fold: str = "policy_exp1",
    train_episodes: int = 24, validation_episodes: int = 6,
    test_episodes: int = 12, steps: int = 20, seed: int = 20261111,
) -> dict:
    if min(train_episodes, validation_episodes, test_episodes, steps) < 2:
        raise ValueError("minimum two steps and episodes in every split")
    frozen = FrozenOriginal(models, fold)
    with ExternalClosedLoopSession(sim_command, require_real=require_real) as session:
        budget = session.budget
        catalog = discrete_joint_catalog(budget)
        if len(catalog) < 2:
            raise ValueError("insufficient BWP PRBs for action variation")
        counter = 0

        def run(which, episode_seed, scenario, model=None, gate=None, residual=0):
            nonlocal counter
            recorded, score = episode(
                session, frozen, catalog, seed=episode_seed, scenario=scenario,
                episode_id=counter, steps=steps, policy=which,
                model=model, gate=gate, residual=residual,
            )
            counter += 1
            return recorded, score

        train = []
        for i in range(train_episodes):
            rows, _ = run("exploration", seed + i, TRAIN_SCENARIOS[i % len(TRAIN_SCENARIOS)])
            train.extend(rows)
        learner = SupportedAdvantage().fit(train, seed=seed)
        # Entire validation set is disjoint from randomized training.
        calibration = []
        for i in range(validation_episodes):
            rows, _ = run("exploration", seed + 100000 + i, TRAIN_SCENARIOS[i % 3])
            calibration.extend(rows)
        predicted, _, _ = learner.predict(np.asarray([r.features for r in calibration]))
        residual = float(np.quantile(np.abs(
            predicted - np.asarray([r.realized_reward for r in calibration]),
        ), .90))
        baseline_validation = [
            run("original", seed + 100000 + i, TRAIN_SCENARIOS[i % 3])[1]
            for i in range(validation_episodes)
        ]
        base_score = float(np.mean([r["mean_reward"] for r in baseline_validation]))
        base_risk = float(np.mean([r["urllc_queue_exceedance"] for r in baseline_validation]))
        choices = []
        for gate in GATES:
            results = [
                run("jagc", seed + 100000 + i, TRAIN_SCENARIOS[i % 3],
                    model=learner, gate=gate, residual=residual)[1]
                for i in range(validation_episodes)
            ]
            reward = float(np.mean([r["mean_reward"] for r in results]))
            risk = float(np.mean([r["urllc_queue_exceedance"] for r in results]))
            choices.append({
                "gate": gate.name, "mean_realized_reward": reward,
                "reward_delta": reward - base_score, "URLLC_risk_delta": risk-base_risk,
                "promote_on_validation": (
                    gate.name != "frozen_original" and reward-base_score >= .002
                    and risk <= base_risk + .005
                ),
            })
        eligible = [x for x in choices if x["promote_on_validation"]]
        winner = max(eligible, key=lambda x: x["reward_delta"]) if eligible else choices[0]
        selected_gate = next(g for g in GATES if g.name == winner["gate"])
        test = []
        observations = []
        for i in range(test_episodes):
            scenario = "weak_channel" if i % 4 == 0 else TRAIN_SCENARIOS[i % 3]
            episode_seed = seed + 200000 + i
            for policy in ("original", "jagc"):
                rows, result = run(policy, episode_seed, scenario, model=learner,
                                   gate=selected_gate, residual=residual)
                test.append(result)
                observations.extend(rows)
    comparison = paired_statistics(test, seed + 832)
    summary = {
        "method": "Agentic-RAN v3 joint-slice supported-advantage control",
        "simulator_backend": budget.backend, "cell_budget": budget.to_dict(),
        "real_phy_actuation_claim": bool(require_real),
        "frozen_original_models_path": frozen.path, "fold": frozen.fold,
        "baseline_comparator": "capacity-projected frozen original Agentic-RAN",
        "baseline_projection_is_not_original_as_issued": True,
        "training_executed_action_variants": len({x.action.prbs for x in train}),
        "training_episodes_with_action_variation": sum(
            len({x.action.prbs for x in train if x.episode == e}) > 1
            for e in {x.episode for x in train}
        ),
        "training_transitions": len(train),
        "validation_residual_margin90": residual,
        "validation_candidates": choices, "selected_validation_gate": winner["gate"],
        "test_used_for_selection": False, "test_paired": comparison,
        "limitations": [
            "Capacity is externally asserted by the NR/Colosseum backend; no cryptographic attestation",
            "The frozen original is explicitly capacity-projected, not as-issued",
            "Local neighbor support and ensemble dispersion do not certify safe action advantage",
            "The included ns3-event-demo is not 5G-LENA NR MAC/PHY",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(choices).to_csv(output / "jagc_closed_loop_validation.csv", index=False)
    pd.DataFrame(test).to_csv(output / "jagc_closed_loop_test_episodes.csv", index=False)
    pd.DataFrame([{
        "policy": t.policy, "seed": t.seed, "episode": t.episode, "scenario": t.scenario,
        "step": t.step, "embb_prbs": t.action.prbs[0],
        "mmtc_prbs": t.action.prbs[1], "urllc_prbs": t.action.prbs[2],
        "realized_reward": t.realized_reward,
        "urllc_queue_exceedance": t.urllc_queue_exceeded,
        "baseline_projected": t.original_projected,
    } for t in observations]).to_csv(
        output / "jagc_closed_loop_executed_transitions.csv.gz", index=False, compression="gzip",
    )
    (output / "jagc_closed_loop_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Actual-action JAGC v3 closed-loop research runner")
    parser.add_argument("--sim-command", required=True, help="external simulator command, quoted")
    parser.add_argument("--models", type=Path, required=True, help="TRUSTED frozen publication_models.joblib")
    parser.add_argument("--output", type=Path, default=Path("results/jagc-closed-loop"))
    parser.add_argument("--fold", default="policy_exp1")
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--train-episodes", type=int, default=24)
    parser.add_argument("--validation-episodes", type=int, default=6)
    parser.add_argument("--test-episodes", type=int, default=12)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--allow-event-demo", action="store_true",
                        help="ONLY for ns-3 event-demo; NOT actual 5G NR evidence")
    args = parser.parse_args()
    result = run_study(
        shlex.split(args.sim_command), args.models, args.output,
        require_real=not args.allow_event_demo, fold=args.fold,
        seed=args.seed, train_episodes=args.train_episodes,
        validation_episodes=args.validation_episodes,
        test_episodes=args.test_episodes, steps=args.steps,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
