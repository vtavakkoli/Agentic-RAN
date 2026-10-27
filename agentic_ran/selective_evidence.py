"""Paired, cluster-aware offline evidence for selective RAN policies.

The logged-incumbent and abstention analyses here are diagnostics, NOT
identified intervention outcomes. Accepted-action utility is a model-based
estimate. The hold-incumbent estimate is also model-based and must only be
reported on evaluator-supported common action cells.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def cluster_accounting(
    frame: pd.DataFrame,
    accepted: np.ndarray,
    selected_dm: np.ndarray,
    reference_dm: np.ndarray,
    incumbent_dm: np.ndarray,
    incumbent_supported: np.ndarray,
    logged_incumbent: np.ndarray,
    logged_observed: np.ndarray,
    *,
    abstention_penalty: float = -0.12,
) -> list[dict[str, Any]]:
    """Summarize aligned policy decisions at independent experiment-cluster level.

    Output is additive across chunks, so it is safe to evaluate million-row
    COMMAG logs in bounded batches. Missing hold outcomes are *never* replaced
    by optimistic predictions or a zero-valued reward.
    """
    n = len(frame)
    vectors = [accepted, selected_dm, reference_dm, incumbent_dm,
               incumbent_supported, logged_incumbent, logged_observed]
    if any(np.asarray(v).shape != (n,) for v in vectors):
        raise ValueError("all accounting vectors must match frame length")
    accepted = np.asarray(accepted, dtype=bool)
    selected_dm = np.asarray(selected_dm, dtype=float)
    reference_dm = np.asarray(reference_dm, dtype=float)
    incumbent_dm = np.asarray(incumbent_dm, dtype=float)
    incumbent_supported = np.asarray(incumbent_supported, dtype=bool)
    logged_incumbent = np.asarray(logged_incumbent, dtype=bool)
    logged_observed = np.asarray(logged_observed, dtype=float)
    if not np.isfinite(reference_dm).all() or not np.isfinite(selected_dm[accepted]).all():
        raise ValueError("nonfinite reference or accepted-action utility")
    hold_supported = (~accepted) & incumbent_supported & np.isfinite(incumbent_dm)
    observed_hold = (~accepted) & logged_incumbent & np.isfinite(logged_observed)
    with_penalty = np.where(accepted, selected_dm, abstention_penalty)
    with_hold = np.where(accepted, selected_dm, incumbent_dm)
    with_observed = np.where(accepted, selected_dm, logged_observed)
    hold_valid = accepted | hold_supported
    observed_valid = accepted | observed_hold
    groups = ["scenario", "training_config", "experiment"]
    absent = set(groups) - set(frame.columns)
    if absent:
        raise ValueError(f"missing experimental cluster identifiers: {sorted(absent)}")
    output = []
    for key, idx in frame.groupby(groups, sort=True).indices.items():
        ix = np.asarray(idx, dtype=int)
        h = ix[hold_valid[ix]]
        o = ix[observed_valid[ix]]
        output.append({
            "scenario": str(key[0]), "training_config": str(key[1]),
            "experiment": str(key[2]), "rows": len(ix),
            "accepted": int(accepted[ix].sum()),
            "abstained": int((~accepted[ix]).sum()),
            "hold_supported_abstentions": int(hold_supported[ix].sum()),
            "logged_incumbent_abstentions": int(observed_hold[ix].sum()),
            "reference_sum": float(reference_dm[ix].sum()),
            "penalized_sum": float(with_penalty[ix].sum()),
            "hold_pair_count": len(h),
            "hold_pair_difference_sum": float(
                (with_hold[h] - reference_dm[h]).sum()
            ),
            "observed_pair_count": len(o),
            "observed_pair_difference_sum": float(
                (with_observed[o] - reference_dm[o]).sum()
            ),
            "accepted_utility_sum": float(selected_dm[ix[accepted[ix]]].sum()),
        })
    return output


def combine_chunks(chunks: list[dict[str, Any]]) -> pd.DataFrame:
    """Combine chunk summaries without accidentally weighting clusters by chunks."""
    if not chunks:
        return pd.DataFrame()
    identifiers = ["scenario", "training_config", "experiment"]
    numeric = [k for k in chunks[0] if k not in identifiers]
    result = pd.DataFrame(chunks)
    return result.groupby(identifiers, sort=True, as_index=False)[numeric].sum()


def paired_cluster_inference(
    clusters: pd.DataFrame, estimator: str, *, seed: int = 2026,
    bootstrap_samples: int = 3000, permutation_samples: int = 5000,
) -> dict[str, Any]:
    """Cluster bootstrap and paired random-sign permutation for a fixed estimand.

    Inference treats each (scenario, training configuration, experiment) as an
    independent unit only as an approximation. These are offline estimates.
    """
    if bootstrap_samples < 1 or permutation_samples < 1:
        raise ValueError("sample counts must be positive")
    definitions = {
        "fixed_penalty": ("rows", "penalized_sum", "reference_sum"),
        "hold_incumbent_dm": ("hold_pair_count", "hold_pair_difference_sum", None),
        "observed_logged_hold_proxy": (
            "observed_pair_count", "observed_pair_difference_sum", None
        ),
    }
    if estimator not in definitions:
        raise ValueError(f"unknown estimator: {estimator}")
    denom, num, baseline = definitions[estimator]
    available = clusters[clusters[denom].gt(0)]
    if baseline is None:
        difference = available[num].to_numpy(float) / available[denom].to_numpy(float)
    else:
        difference = (available[num] - available[baseline]).to_numpy(float) / available[denom].to_numpy(float)
    difference = difference[np.isfinite(difference)]
    total = int(clusters["rows"].sum())
    covered = int(clusters[denom].sum())
    result = {
        "estimator": estimator,
        "interpretation": {
            "fixed_penalty": "assumed abstention score; sensitivity only",
            "hold_incumbent_dm": "same independent outcome model; action-dependent incumbent utility not observed",
            "observed_logged_hold_proxy": "mix of independent DM estimates and observational logged outcomes; noncausal",
        }[estimator],
        "cluster_count": int(len(difference)),
        "paired_row_coverage": covered / max(total, 1),
        "matched_rows": covered,
        "all_rows": total,
    }
    if len(difference) < 2:
        return {**result, "mean_cluster_delta": None,
                "bootstrap_ci95": None, "paired_sign_p": None}
    rng = np.random.default_rng(seed)
    observed = float(difference.mean())
    boot = np.empty(bootstrap_samples)
    for i in range(0, bootstrap_samples, 250):
        size = min(250, bootstrap_samples - i)
        ix = rng.integers(0, len(difference), size=(size, len(difference)))
        boot[i:i + size] = difference[ix].mean(axis=1)
    if len(difference) <= 16:
        count = 1 << len(difference)
        extreme = 0
        for mask in range(count):
            signs = np.where((mask >> np.arange(len(difference))) & 1, 1., -1.)
            extreme += abs(float(np.mean(signs * difference))) >= abs(observed) - 1e-14
        p = extreme / count
    else:
        extreme = 0
        for i in range(0, permutation_samples, 500):
            size = min(500, permutation_samples - i)
            signs = rng.choice((-1., 1.), size=(size, len(difference)))
            extreme += int(np.sum(abs((signs * difference).mean(axis=1)) >= abs(observed) - 1e-14))
        p = (extreme + 1) / (permutation_samples + 1)
    return {
        **result, "mean_cluster_delta": observed,
        "bootstrap_ci95": [
            float(np.quantile(boot, .025)), float(np.quantile(boot, .975))
        ],
        "paired_sign_p": float(p),
        "cluster_sd": float(np.std(difference, ddof=1)),
    }


def accounting_summary(clusters: pd.DataFrame) -> dict[str, float]:
    if clusters.empty:
        return {}
    total = float(clusters["rows"].sum())
    accepted = float(clusters["accepted"].sum())
    return {
        "rows": int(total),
        "actuation_coverage": accepted / max(total, 1),
        "abstention_rate": 1. - accepted / max(total, 1),
        "fixed_penalty_score": float(clusters["penalized_sum"].sum() / max(total, 1)),
        "reference_dm_mean": float(clusters["reference_sum"].sum() / max(total, 1)),
        "accepted_action_dm_mean": (
            float(clusters["accepted_utility_sum"].sum() / accepted)
            if accepted else math.nan
        ),
        "hold_incumbent_evaluator_support_rate": (
            float(clusters["hold_supported_abstentions"].sum()
                  / max(clusters["abstained"].sum(), 1))
        ),
        "observed_logged_incumbent_rate": (
            float(clusters["logged_incumbent_abstentions"].sum()
                  / max(clusters["abstained"].sum(), 1))
        ),
    }
