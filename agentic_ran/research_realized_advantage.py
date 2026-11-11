"""Conservative advantage learning from genuinely actuated randomized rollouts.

Never use static COMMAG to identify intervention effects. The training set
requires explicit randomized exploration with varied executed actions and
their acknowledged next states / measured rewards. All policy tuning is on
validation seeds; test seeds are held out, paired, and never used for fitting.
This is a *research* lower-ensemble-quantile advantage heuristic, not a
formal safety certificate or calibrated causal confidence bound.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from agentic_ran.research_external_ran import run_realized_episode
from agentic_ran.research_physical_prb import NRCellCapacity, SliceAllocation, allocate_shares

# Deliberately small, enumerated physical allocations. Must match the
# executable backend and the behavior policy's support, not COMMAG labels.
SHARE_LIBRARY: tuple[tuple[float, float, float], ...] = (
    (.50, .20, .30),
    (.35, .20, .45),
    (.60, .20, .20),
    (.40, .35, .25),
    (.40, .15, .45),
)


def action_library(
    cell: NRCellCapacity, *, minimums: tuple[int, int, int] = (1, 1, 1),
) -> tuple[SliceAllocation, ...]:
    distinct = tuple(dict.fromkeys(
        allocate_shares(cell, share, minimums=minimums) for share in SHARE_LIBRARY
    ))
    if len(distinct) < 2:
        raise ValueError("no action variation under configured physical NR cell")
    return distinct


def _one_action(allocation: SliceAllocation, choices: tuple[SliceAllocation, ...]) -> int:
    try:
        return choices.index(allocation)
    except ValueError as exc:
        raise ValueError("executed allocation is outside predeclared action support") from exc


def logged_from_external(
    command: list[str], cell: NRCellCapacity,
    *, scenarios: tuple[str, ...], seeds: tuple[int, ...],
    steps: int = 24, exploration_seed: int = 104,
    minimums: tuple[int, int, int] = (1, 1, 1),
) -> list[dict[str, Any]]:
    """Randomized *executed* exploration; not simulated COMMAG replay."""
    if len(seeds) < 4 or len(set(seeds)) != len(seeds) or not scenarios:
        raise ValueError("require distinct training seeds and scenarios")
    choices = action_library(cell, minimums=minimums)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for scenario in scenarios:
            rng = np.random.default_rng(exploration_seed + seed * 101)
            def behavior(_state: np.ndarray, _previous: SliceAllocation, _t: int) -> SliceAllocation:
                return choices[int(rng.integers(len(choices)))]
            _, transitions = run_realized_episode(
                command, cell, behavior, scenario=scenario, seed=seed, steps=steps,
            )
            for item in transitions:
                allocation = SliceAllocation(
                    tuple(int(item["action"]["slice_prbs"][key])
                          for key in ("eMBB", "mMTC", "URLLC")),
                    str(item["action"]["scheduler"]),
                )
                rows.append({
                    "state": item["state"], "action_index": _one_action(allocation, choices),
                    "next_state": item["next_state"], "reward": item["reward"],
                    "sla_violations": item["sla_violations"],
                    "seed": seed, "scenario": scenario, "t": item["t"],
                    "propensity": 1.0 / len(choices),
                    "action_verified": True, "behavior_randomized": True,
                })
    return rows


@dataclass
class AdvantageLearner:
    choices: tuple[SliceAllocation, ...]
    models: list[ExtraTreesRegressor]
    support: dict[int, int]
    min_support: int
    confidence_margin: float = 0.0

    def choose(
        self, state: np.ndarray, original: SliceAllocation,
        *, minimum_advantage: float = 0.0,
    ) -> SliceAllocation:
        """Only replace the frozen original if it lies in supported library."""
        s = np.asarray(state, dtype=float)
        if s.shape != (9,) or not np.isfinite(s).all():
            raise ValueError("invalid RAN state")
        if original not in self.choices:
            return original  # not comparable: fail closed
        incumbent = self.choices.index(original)
        candidates = [
            idx for idx in range(len(self.choices))
            if self.support.get(idx, 0) >= self.min_support
        ]
        if incumbent not in candidates:
            return original
        x = np.vstack([
            np.concatenate((s, np.eye(len(self.choices))[i]))
            for i in candidates
        ])
        preds = np.vstack([model.predict(x) for model in self.models])
        # Per-bootstrap *paired* advantage to the incumbent avoids
        # unpaired uncertainty inflation and includes correlation.
        pos = candidates.index(incumbent)
        differences = preds - preds[:, pos, None]
        # Empirical quantile is a robustness heuristic, not a formal LCB.
        lower = np.quantile(differences, 0.05, axis=0)
        indices = np.flatnonzero(lower > max(minimum_advantage, self.confidence_margin))
        if len(indices) == 0:
            return original
        best = indices[int(np.argmax(lower[indices]))]
        return self.choices[candidates[int(best)]]


def fit_advantages(
    records: list[dict[str, Any]], choices: tuple[SliceAllocation, ...],
    *, min_support: int = 12, ensemble: int = 12, seed: int = 2026,
    min_action_switches: int = 8,
) -> AdvantageLearner:
    """Reject nonrandomized, unverifiable, static, and single-action histories."""
    if len(choices) < 2 or min_support < 2 or ensemble < 4 or min_action_switches < 1:
        raise ValueError("invalid advantage-training configuration")
    if not records or any(
        item.get("behavior_randomized") is not True
        or item.get("action_verified") is not True
        or not 0 < float(item.get("propensity", 0)) <= 1
        for item in records
    ):
        raise ValueError(
            "genuine randomized executed actions with known nonzero propensities required; "
            "COMMAG logged transitions are not valid interventional training examples"
        )
    actions = np.asarray([int(row["action_index"]) for row in records], dtype=int)
    if (actions < 0).any() or (actions >= len(choices)).any():
        raise ValueError("unknown training action")
    episode_keys = [(r["scenario"], r["seed"]) for r in records]
    if len(set(episode_keys)) < 4:
        raise ValueError("need at least four independently seeded training episodes")
    # Evidence of actual within-episode action variation is essential.
    ordered = {}
    for row in records:
        ordered.setdefault((row["scenario"], row["seed"]), []).append(
            (int(row["t"]), int(row["action_index"]))
        )
    changes = sum(
        sum(b[1] != a[1] for a, b in zip(sequence, sequence[1:]))
        for sequence in (sorted(v) for v in ordered.values())
    )
    if changes < min_action_switches:
        raise ValueError("insufficient within-episode executed action variation")
    states = np.asarray([row["state"] for row in records], dtype=float)
    next_states = np.asarray([row["next_state"] for row in records], dtype=float)
    rewards = np.asarray([row["reward"] for row in records], dtype=float)
    if (states.shape != next_states.shape or states.shape[1:] != (9,)
            or not np.isfinite(states).all() or not np.isfinite(next_states).all()
            or not np.isfinite(rewards).all()):
        raise ValueError("invalid measured transition values")
    # Fixed identity one-hot actions, no target leakage (future state only y).
    features = np.concatenate((states, np.eye(len(choices))[actions]), axis=1)
    support = {a: int((actions == a).sum()) for a in range(len(choices))}
    if any(value < min_support for value in support.values()):
        raise ValueError("every compared allocation must have randomized support")
    rng = np.random.default_rng(seed)
    episodes = tuple(sorted(ordered))
    ep_index = {key: i for i, key in enumerate(episodes)}
    group = np.asarray([ep_index[key] for key in episode_keys])
    models: list[ExtraTreesRegressor] = []
    for repeat in range(ensemble):
        # Cluster bootstrap: sample entire randomized trajectories rather
        # than treating correlated individual steps as independent.
        sample_groups = rng.integers(0, len(episodes), len(episodes))
        rows = np.concatenate([np.flatnonzero(group == k) for k in sample_groups])
        model = ExtraTreesRegressor(
            n_estimators=48, min_samples_leaf=2,
            random_state=seed + 7 * repeat, n_jobs=1,
        )
        model.fit(features[rows], rewards[rows])
        models.append(model)
    return AdvantageLearner(choices, models, support, min_support)


def compare_matched_external(
    command: list[str], cell: NRCellCapacity,
    frozen_original: Callable[[np.ndarray, SliceAllocation, int], SliceAllocation],
    candidate: Callable[[np.ndarray, SliceAllocation, int], SliceAllocation],
    *, scenarios: tuple[str, ...], seeds: tuple[int, ...], steps: int = 24,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Realized paired outcomes, identical scenarios/seeds, independent runs.

    Caller MUST supply an actual frozen original controller adapter; no
    substitute heuristic is silently misidentified as Agentic-RAN.
    """
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or not scenarios:
        raise ValueError("need independent held-out seeds and scenarios")
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for seed in seeds:
            baseline, _ = run_realized_episode(
                command, cell, frozen_original,
                scenario=scenario, seed=seed, steps=steps,
            )
            proposed, _ = run_realized_episode(
                command, cell, candidate,
                scenario=scenario, seed=seed, steps=steps,
            )
            if baseline["steps"] != proposed["steps"]:
                raise ValueError("unmatched steps across baseline and candidate")
            rows.append({
                "scenario": scenario, "seed": seed,
                "baseline_realized_reward": baseline["mean_realized_reward"],
                "candidate_realized_reward": proposed["mean_realized_reward"],
                "realized_delta": (
                    proposed["mean_realized_reward"]
                    - baseline["mean_realized_reward"]
                ),
                "baseline_sla_URLLC": baseline["sla_URLLC"],
                "candidate_sla_URLLC": proposed["sla_URLLC"],
                "baseline_sla_eMBB": baseline["sla_eMBB"],
                "candidate_sla_eMBB": proposed["sla_eMBB"],
                "baseline_sla_mMTC": baseline["sla_mMTC"],
                "candidate_sla_mMTC": proposed["sla_mMTC"],
                "baseline_throughput_mbps": baseline["mean_throughput_mbps"],
                "candidate_throughput_mbps": proposed["mean_throughput_mbps"],
                "baseline_latency_ms": baseline["mean_latency_ms"],
                "candidate_latency_ms": proposed["mean_latency_ms"],
            })
    delta = np.asarray([x["realized_delta"] for x in rows], dtype=float)
    baseline_mean = float(np.mean([x["baseline_realized_reward"] for x in rows]))
    return rows, {
        "pairs": len(rows),
        "baseline_mean_realized_reward": baseline_mean,
        "candidate_mean_realized_reward": baseline_mean + float(delta.mean()),
        "paired_mean_realized_delta": float(delta.mean()),
        "relative_improvement": (
            float(delta.mean()) / abs(baseline_mean)
            if abs(baseline_mean) > 1e-12 else None
        ),
        "observed_outcomes": "external action-acknowledged simulator or testbed only",
        "external_backend_identity": "not verified automatically",
        "limitations": (
            "same exogenous disturbances are an explicit responsibility "
            "of the externally deployed simulator/testbed reset protocol"
        ),
    }
