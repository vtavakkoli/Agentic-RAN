"""Aligned cluster-level comparisons for optional adaptive hybrid research.

All comparisons are model-based offline estimates. Absolute abstention
scores use the sensitivity value configured by the publication pipeline.
When two policies abstain on exactly the same rows, the common penalty
cancels from their paired difference.
"""
from __future__ import annotations

import pandas as pd

from agentic_ran.selective_evidence import (
    accounting_summary, combine_chunks, paired_cluster_inference,
)


def compare_cluster_policies(
    candidate_rows: list[dict],
    reference_rows: list[dict],
    *,
    seed: int,
    bootstrap_samples: int,
    permutation_samples: int,
) -> dict:
    candidate = combine_chunks(candidate_rows)
    reference = combine_chunks(reference_rows)
    if candidate.empty or reference.empty:
        raise ValueError("both policy cluster tables must be nonempty")
    keys = ["scenario", "training_config", "experiment"]
    merged = candidate.merge(
        reference[keys + ["rows", "accepted", "penalized_sum"]].rename(
            columns={
                "rows": "reference_rows",
                "accepted": "reference_accepted",
                "penalized_sum": "reference_penalized_sum",
            }
        ),
        on=keys, how="outer", validate="one_to_one", indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("candidate and reference are not aligned on experiment clusters")
    if not merged["rows"].eq(merged["reference_rows"]).all():
        raise ValueError("candidate and reference do not cover identical rows")
    coverage_equal_per_cluster = bool(
        merged["accepted"].eq(merged["reference_accepted"]).all()
    )
    aligned = merged[keys + ["rows", "penalized_sum"]].copy()
    aligned["reference_sum"] = merged["reference_penalized_sum"]
    paired = paired_cluster_inference(
        aligned, "fixed_penalty", seed=seed,
        bootstrap_samples=bootstrap_samples,
        permutation_samples=permutation_samples,
    )
    total = float(merged["rows"].sum())
    delta = float(
        (merged["penalized_sum"] - merged["reference_penalized_sum"]).sum()
        / max(total, 1)
    )
    return {
        "candidate": accounting_summary(candidate),
        "reference": accounting_summary(reference),
        "all_row_score_delta": delta,
        "same_cluster_coverage": coverage_equal_per_cluster,
        "abstention_penalty_cancels": coverage_equal_per_cluster,
        "paired_cluster_inference": paired,
        "interpretation": (
            "Matched independent DM estimates; if coverage matches, the "
            "assumed shared abstention penalty cancels, otherwise the "
            "comparison is sensitive to that assumed penalty. "
            "Not an identified causal treatment effect."
        ),
    }
