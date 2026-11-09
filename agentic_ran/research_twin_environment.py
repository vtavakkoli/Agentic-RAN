"""Action-dependent, reproducible three-slice RAN-like *toy* digital twin.

Scientific scope: this is a simulated intervention environment, not Colosseum,
ns-3/5G-LENA, srsRAN, or independently validated RAN physics.  It is useful
for checking whether learned policies change realized outcomes after actions
and for testing a policy improvement protocol without test-set tuning.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Columns: eMBB / mMTC / URLLC spectrum fractions, summing to one.
ACTIONS = np.asarray([
    [0.50, 0.20, 0.30],
    [0.35, 0.20, 0.45],
    [0.60, 0.20, 0.20],
    [0.40, 0.35, 0.25],
    [0.40, 0.15, 0.45],
], dtype=float)
SLICE_NAMES = ("eMBB", "mMTC", "URLLC")
SLA_QUEUE_LIMITS = np.asarray([3.0, 2.0, 1.10], dtype=float)
SCENARIOS = ("normal", "congested", "bursty", "weak_channel")


@dataclass(frozen=True)
class TwinTransitionBatch:
    state: np.ndarray
    previous_action: np.ndarray
    action: np.ndarray
    next_state: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    t: np.ndarray
    episode: np.ndarray

    def __post_init__(self):
        n = len(self.reward)
        if (
            self.state.shape != (n, 9) or self.next_state.shape != (n, 9)
            or any(np.shape(v) != (n,) for v in (
                self.previous_action, self.action, self.done, self.t, self.episode,
            ))
        ):
            raise ValueError("invalid transition batch shapes")
        if not np.isfinite(self.state).all() or not np.isfinite(self.next_state).all():
            raise ValueError("non-finite transitions")
        if np.any((self.action < 0) | (self.action >= len(ACTIONS))):
            raise ValueError("invalid logged action")
        if len(np.unique(self.episode)) < 2:
            raise ValueError("need multiple independent logged episodes")


class MultiSliceToyTwin:
    """Reproducible exogenous disturbances independent of controller identity."""

    def __init__(self, *, seed: int, scenario: str):
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario: {scenario}")
        self.rng = np.random.default_rng(seed)
        self.scenario = scenario
        self.t = 0
        self.previous_action = 0
        load = {
            "normal": 1.0, "congested": 1.22,
            "bursty": 1.05, "weak_channel": 1.20,
        }[scenario]
        channel_factor = 0.68 if scenario == "weak_channel" else 1.0
        self.demand_reference = load * np.asarray([0.99, 0.35, 0.43])
        self.channel_reference = channel_factor * np.asarray([1.0, 0.87, 0.97])
        self.state = np.concatenate((
            self.rng.uniform(0.0, 0.50, 3),
            self.demand_reference * self.rng.uniform(0.88, 1.13, 3),
            self.channel_reference * self.rng.uniform(0.90, 1.10, 3),
        ))

    @staticmethod
    def reward(next_state: np.ndarray, action: int, previous_action: int) -> float:
        """Observed utility combines queue/service quality and switch overhead."""
        q = np.asarray(next_state, dtype=float)[:3]
        flags = q > SLA_QUEUE_LIMITS
        cost = (
            0.065 * q[0] ** 2 + 0.040 * q[1] ** 2
            + 0.24 * q[2] ** 2 + 0.20 * flags[0]
            + 0.12 * flags[1] + 0.65 * flags[2]
            + 0.028 * np.sum(ACTIONS[action] ** 2)
            + 0.055 * (action != previous_action)
        )
        return float(1.0 - cost)

    def step(self, action: int) -> tuple[np.ndarray, float, np.ndarray]:
        if not (0 <= int(action) < len(ACTIONS)):
            raise ValueError("invalid proposed action")
        prev = self.previous_action
        q, d, ch = np.split(self.state, 3)
        shares = ACTIONS[action]
        served = 3.0 * shares * ch / (1 + 0.055 * q)
        new_q = np.maximum(0.0, q + d - served)
        demand_shock = self.rng.normal(0, 1, 3)
        channel_shock = self.rng.normal(0, 1, 3)
        periodic_burst = (
            0.34 * (np.sin(self.t / 2.8) > 0.8)
            if self.scenario == "bursty" else 0.0
        )
        next_d = np.maximum(
            0.08, 0.76 * d + 0.24 * self.demand_reference
            + 0.045 * demand_shock + periodic_burst * np.asarray([1.0, 0.1, 0.5]),
        )
        next_ch = np.clip(
            0.84 * ch + 0.16 * self.channel_reference
            + 0.035 * channel_shock, 0.32, 1.4,
        )
        following = np.concatenate((new_q, next_d, next_ch))
        rew = self.reward(following, action, prev)
        self.previous_action = int(action)
        self.state = following
        self.t += 1
        return following.copy(), rew, (new_q > SLA_QUEUE_LIMITS)


def generate_logged_transitions(
    *, seed: int, episodes: int = 100, steps: int = 24,
    scenarios: tuple[str, ...] = ("normal", "congested", "bursty"),
) -> TwinTransitionBatch:
    """Offline exploration comes only from TRAIN episodes, before evaluation.

    The behavior policy is stochastic, biased toward a simple queue heuristic
    in 40% of steps, with exploratory random logged actions in the remainder.
    These are truly state-dependent action interventions *in the toy world*.
    """
    if episodes < 6 or steps < 3 or not scenarios:
        raise ValueError("need >=6 episodes, >=3 steps and scenarios")
    for scenario in scenarios:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario: {scenario}")
    states, prevs, actions, next_states = [], [], [], []
    rewards, dones, times, episode_ids = [], [], [], []
    for ep in range(episodes):
        env = MultiSliceToyTwin(seed=seed + ep * 1009, scenario=scenarios[ep % len(scenarios)])
        for t in range(steps):
            before = env.state.copy()
            prev = env.previous_action
            if env.rng.random() < 0.60:
                action = int(env.rng.integers(0, len(ACTIONS)))
            elif before[2] > 0.45:
                action = 1
            elif before[0] > 1.0:
                action = 2
            else:
                action = 0
            nxt, reward, _ = env.step(action)
            states.append(before)
            prevs.append(prev)
            actions.append(action)
            next_states.append(nxt)
            rewards.append(reward)
            dones.append(t == steps - 1)
            times.append(t)
            episode_ids.append(ep)
    return TwinTransitionBatch(
        state=np.asarray(states),
        previous_action=np.asarray(prevs, dtype=int),
        action=np.asarray(actions, dtype=int),
        next_state=np.asarray(next_states),
        reward=np.asarray(rewards),
        done=np.asarray(dones, dtype=bool),
        t=np.asarray(times, dtype=int),
        episode=np.asarray(episode_ids, dtype=int),
    )


def rollout_episode(
    selector, *, seed: int, scenario: str, steps: int,
) -> dict[str, float]:
    """Realize policy actions, including the exact observed hold-last outcome.

    Unlike static COMMAG, here the next state and reward are produced after
    each selected allocation. An abstention physically keeps the last action.
    """
    env = MultiSliceToyTwin(seed=seed, scenario=scenario)
    total = 0.0
    flags = np.zeros(3, dtype=float)
    switches = declined = 0
    for t in range(steps):
        proposal = int(selector(env.state.copy(), env.previous_action, t))
        if proposal < -1 or proposal >= len(ACTIONS):
            raise ValueError("invalid controller output")
        if proposal == -1:
            declined += 1
        action = env.previous_action if proposal == -1 else proposal
        switches += int(action != env.previous_action)
        _, r, violations = env.step(action)
        total += r
        flags += violations.astype(float)
    return {
        "return": float(total / steps),
        "sla_eMBB": float(flags[0] / steps),
        "sla_mMTC": float(flags[1] / steps),
        "sla_URLLC": float(flags[2] / steps),
        "switch_rate": switches / steps,
        "abstention_rate": declined / steps,
    }
