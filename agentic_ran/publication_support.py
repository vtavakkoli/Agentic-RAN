"""Sampled state-conditional action-overlap audit for offline COMMAG OPE.

An action observed somewhere in a slice is *not* necessarily supported near
the state where the candidate policy selects it. This diagnostic compares
selected state/action pairs with each experiment-disjoint training role.

This is an overlap *proxy*, NOT proof of sequential ignorability, causal
identification, or valid intervention outcomes. Measured RAN KPIs may already
reflect the logged configuration.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

# Exclude scheduler_code and slice_prb: including them would trivially reject
# every counterfactual configuration. These remaining KPIs still can be
# consequences of past actions, so the audit is descriptive, not causal.
CONTEXT_COLUMNS = (
    "num_ues",
    "dl_buffer_bytes",
    "ul_buffer_bytes",
    "dl_cqi",
    "ul_sinr",
    "requested_prbs",
)


def _subsample_indices(length: int, limit: int) -> np.ndarray:
    if length <= limit:
        return np.arange(length, dtype=int)
    return np.linspace(0, length - 1, num=limit, dtype=int)


class StateConditionalSupportAudit:
    """Two-role per-slice/action k-NN overlap envelopes.

    Fit radii from within-role second-nearest-neighbor distances (excluding
    the point itself). The held-out sample is never used to tune radii.
    """

    def __init__(
        self,
        policy_fit: pd.DataFrame,
        evaluator_fit: pd.DataFrame,
        *,
        max_reference: int = 2048,
        max_queries: int = 384,
        min_reference: int = 24,
        quantile: float = 0.95,
    ) -> None:
        if not 0 < quantile < 1:
            raise ValueError("quantile must be in (0,1)")
        self.max_queries = max_queries
        self.min_reference = min_reference
        joined = pd.concat([policy_fit, evaluator_fit], ignore_index=True)
        absent = set(CONTEXT_COLUMNS) - set(joined)
        if absent:
            raise ValueError(f"missing context features: {sorted(absent)}")
        self.scaler = RobustScaler(quantile_range=(10, 90)).fit(
            joined[list(CONTEXT_COLUMNS)].to_numpy(float)
        )
        self.references: dict[tuple[str, str, str], tuple[NearestNeighbors, float, int]] = {}
        for role, frame in (("policy", policy_fit), ("evaluator", evaluator_fit)):
            for (slice_type, action), group in frame.groupby(
                ["slice_type", "action"], sort=True
            ):
                if len(group) < min_reference:
                    continue
                group = group.iloc[_subsample_indices(len(group), max_reference)]
                features = self.scaler.transform(
                    group[list(CONTEXT_COLUMNS)].to_numpy(float)
                )
                # A second neighbor excludes self in the training radius.
                nn = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(features)
                within = nn.kneighbors(features, return_distance=True)[0][:, 1]
                radius = max(float(np.quantile(within, quantile)), 1e-4)
                self.references[(role, str(slice_type), str(action))] = (
                    nn, radius, len(features)
                )

    def audit(
        self,
        frame: pd.DataFrame,
        selections: Mapping[str, Sequence[str]],
        *,
        split: str,
        fold: str,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        slices = frame["slice_type"].astype(str).to_numpy()
        for method, action_values in selections.items():
            actions = np.asarray(action_values, dtype=object)
            if len(actions) != len(frame):
                raise ValueError(f"selection length mismatch: {method}")
            for slice_type in sorted(set(slices)):
                local_actions = sorted({str(v) for v in actions[slices == slice_type]})
                for action in local_actions:
                    loc = np.flatnonzero((slices == slice_type) & (actions == action))
                    indices = loc[_subsample_indices(len(loc), self.max_queries)]
                    sample = self.scaler.transform(
                        frame.iloc[indices][list(CONTEXT_COLUMNS)].to_numpy(float)
                    )
                    supports: dict[str, np.ndarray] = {}
                    reference_counts: dict[str, int] = {}
                    for role in ("policy", "evaluator"):
                        reference = self.references.get((role, slice_type, action))
                        if reference is None:
                            supports[role] = np.zeros(len(indices), dtype=bool)
                            reference_counts[role] = 0
                        else:
                            nn, radius, count = reference
                            nearest = nn.kneighbors(sample, n_neighbors=1)[0][:, 0]
                            supports[role] = nearest <= radius
                            reference_counts[role] = count
                    rows.append(
                        {
                            "fold": fold,
                            "split": split,
                            "method": method,
                            "slice_type": slice_type,
                            "action": action,
                            "selected_rows": len(loc),
                            "sampled_rows": len(indices),
                            "policy_reference_rows": reference_counts["policy"],
                            "evaluator_reference_rows": reference_counts["evaluator"],
                            "policy_local_overlap": float(supports["policy"].mean()),
                            "evaluator_local_overlap": float(supports["evaluator"].mean()),
                            "joint_local_overlap": float(
                                (supports["policy"] & supports["evaluator"]).mean()
                            ),
                            "interpretation": "sampled kNN overlap proxy; not causal support",
                        }
                    )
        return rows
