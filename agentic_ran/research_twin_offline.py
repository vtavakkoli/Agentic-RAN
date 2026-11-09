"""Offline CQL and distributional quantile-CVaR research controllers.

Both controllers learn ONLY from action-dependent TRAIN trajectories generated
by research_twin_environment. They share a separately calibrated learned
digital-twin transition model and state-conditional action support. No critic
is trained on validation/test trajectories. No claim of an implementation
identical to deep neural CQL / QR-DQN / CODAC: these are explicit finite-action
linear approximations using Bellman + CQL regularization and quantile-Huber
Bellman learning, respectively. Critical comparisons are closed-loop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from agentic_ran.research_twin_environment import (
    ACTIONS, SLA_QUEUE_LIMITS, TwinTransitionBatch, MultiSliceToyTwin,
)
from agentic_ran.research_pulse_ran import select_pulse_ran

N_ACTIONS = len(ACTIONS)
GAMMA = 0.93
QUANTILES = np.asarray([0.05, 0.15, 0.25, 0.50, 0.75, 0.85, 0.95])


def observation_features(
    states: np.ndarray, previous_actions: np.ndarray, times: np.ndarray,
) -> np.ndarray:
    states = np.asarray(states, dtype=float)
    n = len(states)
    prev = np.asarray(previous_actions, dtype=int).reshape(-1)
    t = np.asarray(times, dtype=float).reshape(-1)
    if states.shape != (n, 9) or len(prev) != n or len(t) != n:
        raise ValueError("invalid observation dimensions")
    if np.any((prev < 0) | (prev >= N_ACTIONS)):
        raise ValueError("invalid previous action")
    prev_one_hot = np.eye(N_ACTIONS)[prev]
    return np.column_stack((states, prev_one_hot, np.sin(t / 4), np.cos(t / 4)))


class FeatureBasis:
    def __init__(self, train: TwinTransitionBatch):
        raw = observation_features(train.state, train.previous_action, train.t)
        self.poly = PolynomialFeatures(2, include_bias=False)
        expanded = self.poly.fit_transform(raw)
        self.scaler = StandardScaler().fit(expanded)

    def transform(self, states, previous_actions, times) -> np.ndarray:
        raw = observation_features(states, previous_actions, times)
        values = self.scaler.transform(self.poly.transform(raw))
        values = np.clip(values, -8.0, 8.0)
        return np.column_stack((np.ones(len(values)), values))


def _adam(weight, gradient, first, second, iteration, learning_rate=0.025):
    first = .9 * first + .1 * gradient
    second = .999 * second + .001 * gradient * gradient
    step = learning_rate * (first / (1 - .9 ** iteration)) / (
        np.sqrt(second / (1 - .999 ** iteration)) + 1e-7
    )
    return weight - step, first, second


@dataclass
class LinearCQLController:
    basis: FeatureBasis
    weights: np.ndarray
    cql_alpha: float
    training_td_rmse: float

    def predict(self, state, previous_action, t) -> np.ndarray:
        x = self.basis.transform(
            np.asarray(state, float).reshape(1, 9),
            np.array([previous_action]), np.array([t]),
        )
        return (x @ self.weights.T)[0]


def fit_linear_cql(
    train: TwinTransitionBatch, *, alpha: float, gamma: float = GAMMA,
    iterations: int = 5, steps_per_iter: int = 75,
) -> LinearCQLController:
    """Finite-action offline CQL objective: Bellman MSE + logsumexp - Q_logged.

    No online test interactions or reward oracle enter this fitting procedure.
    """
    if not 0 <= alpha <= 2 or not 0 < gamma < 1:
        raise ValueError("invalid CQL alpha or discount")
    basis = FeatureBasis(train)
    x = basis.transform(train.state, train.previous_action, train.t)
    xn = basis.transform(train.next_state, train.action, train.t + 1)
    n, d = x.shape
    w = np.zeros((N_ACTIONS, d), dtype=float)
    rows = np.arange(n)
    onehot = np.eye(N_ACTIONS)[train.action]
    last_td = None
    for outer in range(iterations):
        old_q = xn @ w.T
        next_value = np.max(old_q, axis=1)
        target = train.reward + gamma * (~train.done) * next_value
        # Independent target for one complete fitted Q iteration.
        first, second = np.zeros_like(w), np.zeros_like(w)
        for step in range(1, steps_per_iter + 1):
            q = x @ w.T
            q_centered = q - np.max(q, axis=1, keepdims=True)
            softmax = np.exp(np.clip(q_centered, -60, 0))
            softmax /= softmax.sum(axis=1, keepdims=True)
            td = np.sum(q * onehot, axis=1) - target
            partial = td[:, None] * onehot + alpha * (softmax - onehot)
            gradient = partial.T @ x / n + 1e-3 * w
            gradient = np.clip(gradient, -3, 3)
            w, first, second = _adam(w, gradient, first, second, step, 0.03)
        last_td = float(np.sqrt(np.mean(td ** 2)))
    return LinearCQLController(basis, w, alpha, last_td)


@dataclass
class QuantileCVaRController:
    basis: FeatureBasis
    weights: np.ndarray
    quantiles: np.ndarray
    training_pinball: float

    def distribution(self, state, previous_action, t) -> np.ndarray:
        x = self.basis.transform(
            np.asarray(state, float).reshape(1, 9),
            np.array([previous_action]), np.array([t]),
        )
        return np.einsum("nd,akd->nak", x, self.weights)[0]

    def scores(self, state, previous_action, t, risk_weight: float) -> np.ndarray:
        """Lower-tail expected return (CVaR), not upper-tail SLA cost."""
        if not 0 <= risk_weight <= 1:
            raise ValueError("risk weight must be in [0,1]")
        dist = self.distribution(state, previous_action, t)
        dist = np.sort(dist, axis=1)
        mean = dist.mean(axis=1)
        cvar_lower = dist[:, :2].mean(axis=1)
        return (1 - risk_weight) * mean + risk_weight * cvar_lower


def fit_quantile_td(
    train: TwinTransitionBatch, *, gamma: float = GAMMA,
    iterations: int = 5, steps_per_iter: int = 85,
) -> QuantileCVaRController:
    """Offline quantile-Huber fitted Bellman updates for discrete actions.

    Each quantile learns from all target quantiles of the action selected at the
    next logged state. No inference-time knowledge of true future reward.
    """
    if not 0 < gamma < 1:
        raise ValueError("invalid discount")
    basis = FeatureBasis(train)
    x = basis.transform(train.state, train.previous_action, train.t)
    xn = basis.transform(train.next_state, train.action, train.t + 1)
    n, d = x.shape
    k = len(QUANTILES)
    w = np.zeros((N_ACTIONS, k, d), dtype=float)
    rows = np.arange(n)
    result_loss = 0.0
    for outer in range(iterations):
        z_next = np.einsum("nd,akd->nak", xn, w)
        best = np.argmax(z_next.mean(axis=2), axis=1)
        target = (
            train.reward[:, None]
            + gamma * (~train.done)[:, None] * z_next[rows, best]
        )
        first, second = np.zeros_like(w), np.zeros_like(w)
        for step in range(1, steps_per_iter + 1):
            chosen = np.einsum("nd,nkd->nk", x, w[train.action])
            error = target[:, None, :] - chosen[:, :, None]
            # Quantile Huber: derivative of |tau - I(delta<0)|*Huber(delta)
            clipped = np.clip(error, -1, 1)
            weight = np.abs(
                QUANTILES[None, :, None] - (error < 0).astype(float)
            )
            partial = -(weight * clipped).mean(axis=2)
            grad = 8e-4 * w
            for action in range(N_ACTIONS):
                mask = train.action == action
                if np.any(mask):
                    grad[action] += partial[mask].T @ x[mask] / n
            grad = np.clip(grad, -3, 3)
            w, first, second = _adam(w, grad, first, second, step, .035)
        result_loss = float(np.mean(np.abs(error)))
    return QuantileCVaRController(basis, w, QUANTILES.copy(), result_loss)


@dataclass
class CalibratedTwin:
    dynamics: object
    state_support: tuple[NearestNeighbors | None, ...]
    support_thresholds: np.ndarray
    queue_upper_margin: np.ndarray
    queue_cvar_margin: np.ndarray
    calibration_rows: int
    calibration_episodes: int

    def predict_next(self, state, t) -> np.ndarray:
        state = np.asarray(state, dtype=float).reshape(1, 9)
        actions = np.arange(N_ACTIONS)
        x = np.column_stack((
            np.repeat(state, N_ACTIONS, axis=0),
            np.eye(N_ACTIONS), np.full(N_ACTIONS, np.sin(t / 4)),
            np.full(N_ACTIONS, np.cos(t / 4)),
        ))
        predicted = self.dynamics.predict(x)
        predicted[:, :3] = np.maximum(0, predicted[:, :3])
        predicted[:, 3:6] = np.maximum(.04, predicted[:, 3:6])
        predicted[:, 6:] = np.clip(predicted[:, 6:], .25, 1.4)
        return predicted

    def eligible(self, state, t, *, risk_mode="upper") -> tuple[np.ndarray, np.ndarray]:
        predicted = self.predict_next(state, t)
        query = np.asarray(state, dtype=float).reshape(1, 9)
        distances = np.array([
            estimator.kneighbors(query, n_neighbors=1)[0][0, 0]
            if estimator is not None else np.inf
            for estimator in self.state_support
        ])
        supported = distances <= self.support_thresholds
        if risk_mode == "upper":
            margin = self.queue_upper_margin
        elif risk_mode == "cvar":
            margin = self.queue_cvar_margin
        else:
            raise ValueError("unknown queue risk estimand")
        acceptable = np.all(
            predicted[:, :3] + margin[None, :] <= SLA_QUEUE_LIMITS[None, :],
            axis=1,
        )
        return supported & acceptable, predicted


def fit_calibrated_twin(
    train: TwinTransitionBatch,
    *, calibration_fraction: float = .2,
) -> CalibratedTwin:
    """Episode-disjoint fit/calibration for one-step risk, no held-out use."""
    ep = np.unique(train.episode)
    if len(ep) < 12:
        raise ValueError("need at least 12 logged episodes for twin calibration")
    ncal = max(3, int(round(len(ep) * calibration_fraction)))
    held = np.isin(train.episode, ep[-ncal:])
    fit = ~held
    assert not np.intersect1d(
        np.unique(train.episode[fit]), np.unique(train.episode[held])
    ).size
    def inputs(mask):
        return np.column_stack((
            train.state[mask],
            np.eye(N_ACTIONS)[train.action[mask]],
            np.sin(train.t[mask] / 4), np.cos(train.t[mask] / 4),
        ))
    model = make_pipeline(
        StandardScaler(), PolynomialFeatures(2, include_bias=False),
        Ridge(alpha=.65),
    )
    model.fit(inputs(fit), train.next_state[fit])
    residual = train.next_state[held, :3] - model.predict(inputs(held))[:, :3]
    # Positive residual uncertainty measures unanticipated queue growth.
    upper = np.maximum(
        0, np.quantile(residual, .90, axis=0),
    )
    tail = np.maximum(
        upper, np.mean(
            np.sort(residual, axis=0)[max(0, int(.9 * len(residual))):],
            axis=0,
        ),
    )
    neighbors = []
    limits = []
    # State-conditional action overlap: action-specific nearest-neighbor
    # thresholds calibrated *within training episodes*, never final tests.
    for action in range(N_ACTIONS):
        states = train.state[fit & (train.action == action)]
        if len(states) < 10:
            # Sparse actions remain unavailable instead of claiming overlap
            # or aborting every controller on a thin research dataset.
            neighbors.append(None)
            limits.append(-1.0)
            continue
        estimator = NearestNeighbors(n_neighbors=2).fit(states)
        leave_self = estimator.kneighbors(states, n_neighbors=2)[0][:, 1]
        neighbors.append(estimator)
        limits.append(
            max(.15, float(np.quantile(leave_self, .98) * 2.0))
        )
    return CalibratedTwin(
        model, tuple(neighbors), np.asarray(limits),
        upper, tail, int(held.sum()), ncal,
    )


def choose_action(
    mode: str, state: np.ndarray, previous: int, t: int,
    *, twin: CalibratedTwin,
    cql: LinearCQLController | None = None,
    quantile: QuantileCVaRController | None = None,
    risk_weight: float = .3, blend_weight: float = .5,
) -> int:
    """Choose an action; learned controllers use twin support and risk gates.

    PULSE-RAN is the original unshielded priority rule, and its optional
    gated control is a separate diagnostic. The legacy 'behavior' and
    'heuristic' names remain accepted for backward compatibility.
    """
    if mode not in {
        "pulse_ran", "pulse_ran_gated", "behavior", "heuristic",
        "twin_greedy", "twin_greedy_cvar_gate",
        "twin_cql", "distributional_cvar", "twin_cql_cvar_hybrid",
    }:
        raise ValueError("unknown comparison policy")
    if mode in {"pulse_ran", "behavior", "heuristic"}:
        return select_pulse_ran(state)
    safe, predicted = twin.eligible(
        state, t, risk_mode=(
            "cvar" if mode in (
                "twin_greedy_cvar_gate", "distributional_cvar",
                "twin_cql_cvar_hybrid"
            )
            else "upper"
        ),
    )
    if mode == "pulse_ran_gated":
        # Exactly the same action-specific upper queue and support mask.
        # If PULSE's requested action is unsupported, return an explicit
        # abstention: the simulator then executes and measures hold-last.
        # This does NOT certify that hold-last is safe.
        requested = select_pulse_ran(state)
        return requested if safe[requested] else -1
    if not safe.any():
        return -1  # external environment executes explicit hold-last
    if mode in ("twin_greedy", "twin_greedy_cvar_gate"):
        score = np.array([
            MultiSliceToyTwin.reward(predicted[a], a, previous)
            for a in range(N_ACTIONS)
        ])
    elif mode == "twin_cql":
        if cql is None:
            raise ValueError("CQL critic is missing")
        score = cql.predict(state, previous, t)
    elif mode == "distributional_cvar":
        if quantile is None:
            raise ValueError("distributional critic is missing")
        score = quantile.scores(state, previous, t, risk_weight)
    else:
        if quantile is None or cql is None:
            raise ValueError("hybrid requires both trained critics")
        if not 0 <= blend_weight <= 1:
            raise ValueError("hybrid blend weight must be in [0,1]")
        score = (
            (1 - blend_weight) * cql.predict(state, previous, t)
            + blend_weight * quantile.scores(state, previous, t, risk_weight)
        )
    # Identical switching disutility for each learned, gated ranking policy.
    score = score - .055 * (np.arange(N_ACTIONS) != previous)
    score = np.where(safe, score, -np.inf)
    return int(np.argmax(score))
