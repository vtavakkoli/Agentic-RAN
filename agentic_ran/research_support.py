"""Train-only action-conditional support and fail-closed RAN re-ranking.

This is an experimental overlap diagnostic. Split-conformal p-values are
marginal under exchangeability within each slice/action/role, NOT causal
positivity or a guarantee of real-world network safety. Candidate utility,
SLA upper bounds and uncertainty must be supplied by independently validated
critic models; the module does not calibrate those critics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler


@dataclass(frozen=True)
class Selection:
    actions: np.ndarray  # -1 means ABSTAIN, never issue a network action
    abstain: np.ndarray
    reranked: np.ndarray
    eligible_counts: np.ndarray
    method: str
    alpha: float


class ConformalActionSupport:
    """Two experiment-disjoint training roles with per-action split calibration."""

    def __init__(
        self, methods: Sequence[str] = ("knn", "gaussian", "gmm"),
        min_group: int = 32, max_reference: int = 1500,
        calibration_fraction: float = 0.25, seed: int = 2026,
    ):
        self.methods = tuple(methods)
        if set(self.methods) - {"knn", "gaussian", "gmm"}:
            raise ValueError("unknown support estimator")
        if min_group < 12 or max_reference < min_group or not 0 < calibration_fraction < 0.5:
            raise ValueError("invalid calibration/reference configuration")
        self.min_group, self.max_reference = min_group, max_reference
        self.calibration_fraction, self.seed = calibration_fraction, seed
        self.fitted = {}
        self.calibration_units = {}
        self.calibration_protocol = "unfitted"

    def fit(self, roles: dict[str, tuple]):
        """Fit action-conditional conformal nonconformity models.

        Publication roles supply (context, slice, action, episode_id) to
        guarantee disjoint training/calibration episodes. Three-tuples retain
        legacy IID-row behavior for explicitly synthetic experiments.
        Calibrate on one maximum nonconformity score per episode; a nominal
        low alpha is never allowed unless enough independent calibration
        episodes exist to resolve it.
        """
        if set(roles) != {"policy", "evaluator"}:
            raise ValueError("require policy and evaluator training roles")
        rng = np.random.default_rng(self.seed)
        self.fitted = {}
        self.calibration_units = {}
        grouped = all(len(value) == 4 for value in roles.values())
        if any(len(value) not in (3, 4) for value in roles.values()):
            raise ValueError("each role needs context, slice, action, [episode_id]")
        if not grouped and any(len(value) == 4 for value in roles.values()):
            raise ValueError("mixing episode-blocked and IID roles is forbidden")
        self.calibration_protocol = (
            "held_out_episode_maximum" if grouped else "synthetic_iid_row"
        )
        if grouped:
            policy_episodes = set(map(str, roles["policy"][3]))
            evaluator_episodes = set(map(str, roles["evaluator"][3]))
            if policy_episodes & evaluator_episodes:
                raise ValueError("cross-role training episodes overlap")
        for role, data in roles.items():
            x = np.asarray(data[0], dtype=float)
            slices = np.asarray(data[1], dtype=str)
            actions = np.asarray(data[2], dtype=str)
            episodes = np.asarray(data[3], dtype=str) if grouped else None
            if x.ndim != 2 or len(x) != len(slices) or len(x) != len(actions):
                raise ValueError("incompatible role inputs")
            if grouped and len(episodes) != len(x):
                raise ValueError("episode identifiers must match context rows")
            for sl, act in sorted(set(zip(slices, actions))):
                idx = np.flatnonzero((slices == sl) & (actions == act))
                idx = idx[np.isfinite(x[idx]).all(axis=1)]
                if len(idx) < self.min_group:
                    continue
                if grouped:
                    episode_ids = np.unique(episodes[idx])
                    if len(episode_ids) < 2:
                        continue
                    shuffled = rng.permutation(episode_ids)
                    n_cal_episodes = max(1, int(np.ceil(
                        len(shuffled) * self.calibration_fraction
                    )))
                    chosen_cal = set(shuffled[:n_cal_episodes])
                    is_cal = np.isin(episodes[idx], list(chosen_cal))
                    fit_idx = idx[~is_cal]
                    cal_idx = idx[is_cal]
                    # No row from a calibration episode can enter model fit.
                    if len(fit_idx) < 8 or len(cal_idx) < 8:
                        continue
                    fit_idx = rng.permutation(fit_idx)[:self.max_reference]
                    # Calibrate on complete held-out episodes, never on a
                    # randomly thinned set of rows within an episode.
                    cal_episodes = np.unique(episodes[cal_idx])
                    if len(cal_episodes) > self.max_reference:
                        cal_episodes = rng.permutation(cal_episodes)[
                            :self.max_reference
                        ]
                        cal_idx = cal_idx[np.isin(
                            episodes[cal_idx], cal_episodes,
                        )]
                else:
                    idx = rng.permutation(idx)[:self.max_reference]
                    n_cal = max(8, int(len(idx) * self.calibration_fraction))
                    fit_idx, cal_idx = idx[n_cal:], idx[:n_cal]
                    if len(fit_idx) < 8:
                        continue
                train, cal = x[fit_idx], x[cal_idx]
                scaler = RobustScaler().fit(train)
                z_train, z_cal = scaler.transform(train), scaler.transform(cal)
                estimators = {}
                if "knn" in self.methods:
                    estimators["knn"] = NearestNeighbors(
                        n_neighbors=1
                    ).fit(z_train)
                if "gaussian" in self.methods:
                    estimators["gaussian"] = LedoitWolf().fit(z_train)
                if "gmm" in self.methods:
                    components = min(2, max(1, len(train) // 16))
                    estimators["gmm"] = GaussianMixture(
                        n_components=components, covariance_type="diag",
                        reg_covar=1e-3, random_state=self.seed, max_iter=100,
                    ).fit(z_train)
                for method, model in estimators.items():
                    scores = self._nonconformity(model, method, z_cal)
                    if not np.isfinite(scores).all():
                        continue
                    if grouped:
                        # Episode maxima mitigate within-episode dependence,
                        # at the cost of stronger conservatism and lower n.
                        maxima = []
                        for episode in np.unique(episodes[cal_idx]):
                            maxima.append(float(np.max(
                                scores[episodes[cal_idx] == episode]
                            )))
                        scores = np.asarray(maxima, dtype=float)
                    self.fitted[(role, sl, act, method)] = (
                        scaler, model, np.sort(scores),
                    )
                    self.calibration_units[(role, sl, act, method)] = len(scores)
        return self

    @staticmethod
    def _nonconformity(model, method, z):
        if method == "knn":
            return model.kneighbors(z, return_distance=True)[0][:, 0]
        if method == "gaussian":
            delta = z - model.location_
            return np.einsum("ij,jk,ik->i", delta, model.precision_, delta)
        return -model.score_samples(z)

    def pvalues(self, x: np.ndarray, slices: Sequence[str],
                action: str, method: str) -> np.ndarray:
        """Minimum of policy/evaluator role p-values; fail closed on missing."""
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or len(x) != len(slices):
            raise ValueError("invalid context features")
        if method not in self.methods:
            raise ValueError("unknown method")
        result = np.zeros(len(x), dtype=float)
        slices = np.asarray(slices, dtype=str)
        for sl in np.unique(slices):
            idx = np.flatnonzero(slices == sl)
            pair = [
                self.fitted.get((role, sl, str(action), method))
                for role in ("policy", "evaluator")
            ]
            if any(item is None for item in pair):
                continue
            idx = idx[np.isfinite(x[idx]).all(axis=1)]
            if not len(idx):
                continue
            role_p = []
            for scaler, model, cal in pair:
                values = self._nonconformity(model, method, scaler.transform(x[idx]))
                ge = len(cal) - np.searchsorted(cal, values, side="left")
                role_p.append((ge + 1) / (len(cal) + 1))
            result[idx] = np.minimum(role_p[0], role_p[1])
        return result

    def action_support(self, x, slices, action_names, method, alpha=0.05):
        if not 0 <= alpha < 1 or method not in self.methods:
            raise ValueError("invalid method or alpha")
        result = np.column_stack([
            self.pvalues(x, slices, name, method) > alpha
            for name in action_names
        ])
        # If the requested alpha is below the attainable calibration p-value
        # resolution, *every* candidate would pass. Do not report an
        # uninformative support guarantee from an underpowered stratum.
        labels = np.asarray(slices, dtype=str)
        min_cal = (1.0 / alpha - 1.0) if alpha > 0 else float("inf")
        for column, action in enumerate(action_names):
            for sl in np.unique(labels):
                refs = [
                    self.fitted.get((role, sl, str(action), method))
                    for role in ("policy", "evaluator")
                ]
                if any(ref is None or len(ref[2]) < min_cal for ref in refs):
                    result[labels == sl, column] = False
        return result


def rerank(
    scores: np.ndarray, sla_upper: np.ndarray, uncertainty: np.ndarray,
    allowed: np.ndarray, support: np.ndarray, incumbent: Sequence[int],
    *, method: str, alpha: float = 0.05, max_sla: float = 0.45,
    max_uncertainty: float = 0.5, switch_cost: float = 0.0,
) -> Selection:
    """Pick best eligible action. If none exists, ABSTAIN without fallback.

    SLA upper bounds are only as reliable as their *separately calibrated*
    uncertainty model. Validating an incumbent requires the same complete gate.
    """
    score = np.asarray(scores, dtype=float)
    shape = score.shape
    if len(shape) != 2 or any(np.shape(a) != shape for a in (
        sla_upper, uncertainty, allowed, support,
    )):
        raise ValueError("candidate arrays must have identical 2D shape")
    incumbent = np.asarray(incumbent, dtype=int)
    if incumbent.shape != (shape[0],):
        raise ValueError("incumbent shape mismatch")
    valid = (
        np.asarray(allowed, dtype=bool) & np.asarray(support, dtype=bool)
        & np.isfinite(score) & np.isfinite(sla_upper)
        & np.isfinite(uncertainty) & (sla_upper <= max_sla)
        & (uncertainty <= max_uncertainty)
    )
    adjusted = score.copy()
    if switch_cost:
        adjusted -= switch_cost * (np.arange(shape[1])[None, :] != incumbent[:, None])
    adjusted = np.where(valid, adjusted, -np.inf)
    best = np.argmax(adjusted, axis=1)
    counts = valid.sum(axis=1)
    abstain = counts == 0
    chosen = np.where(abstain, -1, best)
    original = np.argmax(np.where(np.asarray(allowed, bool), score, -np.inf), axis=1)
    return Selection(chosen, abstain, (~abstain) & (chosen != original),
                     counts, method, alpha)
