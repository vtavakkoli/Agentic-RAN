"""Interactive action-dependent RAN-like queue simulator and learned-model MPC.

This is a controlled interventional *toy* environment, not calibrated srsRAN,
Colosseum, E2AP, or evidence of live-network control. The same exogenous
disturbances are replayed in paired horizon comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ACTIONS = np.array([0.35, 0.7, 1.0])


def next_state(state, action, t, *, noise=None, disturbance=0.0):
    """Intervention affects queue through allocated spectrum/channel rate."""
    q, demand, channel = np.moveaxis(np.asarray(state, dtype=float), -1, 0)
    a = np.asarray(action, dtype=float)
    served = 1.65 * a * channel / (1.0 + 0.055 * q)
    qnew = np.maximum(0.0, q + demand - served)
    demand_new = 0.75 * demand + 0.25 * (0.95 + 0.24 * np.sin(t / 3.5))
    channel_new = 0.88 * channel + 0.12 * (0.9 + 0.12 * np.cos(t / 5.0))
    if noise is not None:
        demand_new += disturbance * noise[0]
        channel_new += disturbance * noise[1]
    return np.stack(np.broadcast_arrays(
        qnew, np.maximum(0.15, demand_new),
        np.clip(channel_new, 0.25, 1.6),
    ), axis=-1)


def reward(next_s, action, previous_action):
    q = np.asarray(next_s)[..., 0]
    a = np.asarray(action)
    return -(0.15 * q**2 + 0.26 * a**2
             + 0.16 * (np.abs(a - previous_action) > 1e-8))


def features(state, action, t):
    s = np.asarray(state, dtype=float)
    a = np.broadcast_to(np.asarray(action, dtype=float), s.shape[:-1])
    return np.column_stack((
        s.reshape(-1, 3), a.ravel(),
        np.full(a.size, np.sin(t / 3.5)),
        np.full(a.size, np.cos(t / 5.0)),
    ))


@dataclass
class LearnedDynamics:
    predictor: object
    rmse: float

    def predict(self, state, action, t, bias=0.0):
        s = np.asarray(state, dtype=float)
        new = self.predictor.predict(features(s, action, t)).reshape(s.shape)
        new[..., 0] += bias
        new[..., 0] = np.maximum(0.0, new[..., 0])
        new[..., 1] = np.maximum(0.15, new[..., 1])
        new[..., 2] = np.clip(new[..., 2], 0.25, 1.6)
        return new


def fit_dynamics(seed=17, count=2800):
    """Independent offline random-action training with held-out model RMSE."""
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 60, size=count)
    states = np.column_stack((
        rng.uniform(0, 5, count),
        rng.uniform(0.5, 1.6, count),
        rng.uniform(0.55, 1.35, count),
    ))
    action = rng.choice(ACTIONS, size=count)
    x = np.column_stack((
        states, action, np.sin(t / 3.5), np.cos(t / 5.0),
    ))
    y = np.vstack([
        next_state(states[i], action[i], t[i]) for i in range(count)
    ])
    permutation = rng.permutation(count)
    a, b = permutation[:int(count * 0.8)], permutation[int(count * 0.8):]
    model = make_pipeline(
        StandardScaler(), PolynomialFeatures(2, include_bias=False),
        Ridge(alpha=0.5),
    )
    model.fit(x[a], y[a])
    rmse = float(np.sqrt(np.mean((model.predict(x[b]) - y[b])**2)))
    return LearnedDynamics(model, rmse)


def plan(
    state, previous_action, t, horizon, model, *, bias=0.0,
    risk_limit=4.5, discount=0.93, uncertainty_penalty=0.03,
    risk_margin_z=1.0,
):
    """Receding-horizon search; reject predicted-unsafe action sequences.

    Return -1 when *no* sequence satisfies the modeled constraint. No
    automatic fallback. Model misspecification can violate the true limit:
    predicted safety is NOT independently verified network safety.
    """
    if horizon < 1 or horizon > 6:
        raise ValueError("horizon must be in 1..6")
    if risk_margin_z < 0:
        raise ValueError("risk_margin_z must be nonnegative")
    choices = np.asarray(list(product(range(len(ACTIONS)), repeat=horizon)), dtype=int)
    simulated = np.tile(np.asarray(state, dtype=float), (len(choices), 1))
    previous = np.full(len(choices), previous_action, dtype=float)
    scores = np.zeros(len(choices))
    feasible = np.ones(len(choices), dtype=bool)
    for h in range(horizon):
        action = ACTIONS[choices[:, h]]
        simulated = model.predict(simulated, action, t + h, bias=bias)
        # Use a held-out model-error proxy, not the injected bias (which
        # would leak the experimenter's misspecification into the planner).
        model_error = max(float(getattr(model, "rmse", 0.0)), 0.0)
        margin = risk_margin_z * model_error * np.sqrt(h + 1)
        feasible &= simulated[:, 0] <= risk_limit - margin
        scores += discount**h * (
            reward(simulated, action, previous)
            - uncertainty_penalty * model_error**2 * (h + 1)
        )
        previous = action
    scores[~feasible] = -np.inf
    if not np.isfinite(scores).any():
        return -1
    return int(choices[int(np.argmax(scores)), 0])


def evaluate_planners(seed=999, episodes=12, steps=20, disturbance=0.10):
    """Paired episodes across horizons and deliberate model-error levels."""
    model = fit_dynamics()
    results = {}
    paired = {}
    for bias in (0.0, 0.12, 0.30):
        returns_by_horizon = {}
        for horizon in (1, 2, 4):
            cumulative, violations, abstentions, switching = [], [], [], []
            for episode in range(episodes):
                rng = np.random.default_rng(seed + episode)
                noise = rng.standard_normal((steps, 2))
                state = np.array([2.3 + 0.3 * (episode % 3), 0.99, 0.85])
                previous = float(ACTIONS[1])
                total, unsafe, declined, switched = 0.0, 0, 0, 0
                for t in range(steps):
                    choice = plan(state, previous, t, horizon, model, bias=bias)
                    if choice < 0:
                        declined += 1
                        # EXTERNAL hold-last dynamics on controller abstention.
                        # This is not an approved/safe fallback action.
                        action = previous
                    else:
                        action = float(ACTIONS[choice])
                    future = next_state(
                        state, action, t, noise=noise[t], disturbance=disturbance,
                    )
                    total += reward(future, action, previous)
                    unsafe += int(future[0] > 4.5)
                    switched += int(abs(action - previous) > 1e-8)
                    state, previous = future, action
                cumulative.append(total)
                violations.append(unsafe / steps)
                abstentions.append(declined / steps)
                switching.append(switched / steps)
            returns_by_horizon[horizon] = np.asarray(cumulative)
            results[f"bias={bias:.2f}:h={horizon}"] = {
                "mean_return": float(np.mean(cumulative)),
                "return_std": float(np.std(cumulative, ddof=1)),
                "sla_violation": float(np.mean(violations)),
                "abstention": float(np.mean(abstentions)),
                "switching": float(np.mean(switching)),
            }
        for horizon in (2, 4):
            delta = returns_by_horizon[horizon] - returns_by_horizon[1]
            rng = np.random.default_rng(seed + 50000 + horizon)
            sampled = rng.integers(0, episodes, size=(2500, episodes))
            ci = np.quantile(delta[sampled].mean(axis=1), [0.025, 0.975])
            paired[f"bias={bias:.2f}:h={horizon}-h=1"] = {
                "mean_return_improvement": float(delta.mean()),
                "bootstrap_ci95": [float(ci[0]), float(ci[1])],
                "episode_win_rate": float(np.mean(delta > 0)),
                "episodes": episodes,
            }
    return {
        "dynamics_holdout_rmse": model.rmse,
        "scenarios": results,
        "paired_horizon_effects": paired,
        "warning": "Toy action-dependent closed-loop simulator, not a RAN validation.",
    }


class OracleDynamics:
    """True toy transition dynamics, for a simulation upper-reference only.

    This oracle is NEVER a deployable RAN world model.
    """
    rmse = 0.0

    def predict(self, state, action, t, bias=0.0):
        return next_state(state, action, t)


def evaluate_oracle_gap(
    *, seed: int = 2026, episodes: int = 12, steps: int = 20,
    disturbance: float = 0.10, horizons=(1, 2, 4),
):
    """Compare learned vs oracle receding-horizon control on paired noise.

    Includes realized constraint violations and costs. Even oracle MPC is
    noncausal evidence for real RAN: this is a synthetic intervention engine.
    """
    if episodes < 2 or steps < 1:
        raise ValueError("episodes must be >=2 and steps >=1")
    learned = fit_dynamics(seed=seed + 7)
    models = {"learned": learned, "oracle": OracleDynamics()}
    output = {}
    for model_name, model in models.items():
        for horizon in horizons:
            returns = []
            violations = []
            abstentions = []
            switches = []
            for ep in range(episodes):
                rng = np.random.default_rng(seed + ep)
                disturbances = rng.standard_normal((steps, 2))
                state = np.array([2.3 + 0.3 * (ep % 3), 0.99, 0.85])
                previous = float(ACTIONS[1])
                total = bad = abstained = switched = 0
                for t in range(steps):
                    choice = plan(
                        state, previous, t, horizon, model,
                        risk_margin_z=1.0,
                    )
                    if choice < 0:
                        abstained += 1
                        # Environmental hold, not controller-issued safety action.
                        action = previous
                    else:
                        action = float(ACTIONS[choice])
                    future = next_state(
                        state, action, t, noise=disturbances[t],
                        disturbance=disturbance,
                    )
                    total += float(reward(future, action, previous))
                    bad += int(future[0] > 4.5)
                    switched += int(abs(action - previous) > 1e-8)
                    state, previous = future, action
                returns.append(total)
                violations.append(bad / steps)
                abstentions.append(abstained / steps)
                switches.append(switched / steps)
            output[f"{model_name}:h={horizon}"] = {
                "mean_return": float(np.mean(returns)),
                "violation_rate": float(np.mean(violations)),
                "abstention_rate": float(np.mean(abstentions)),
                "switching_rate": float(np.mean(switches)),
                "paired_episode_returns": returns,
            }
    return {
        "learned_transition_holdout_rmse": learned.rmse,
        "results": output,
        "warning": "Oracle and learned planning share a toy simulator; not live RAN evidence.",
    }
