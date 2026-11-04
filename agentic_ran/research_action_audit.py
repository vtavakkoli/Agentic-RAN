"""Audit whether COMMAG contains actual *within-episode* action variation.

Cross-episode diversity does not identify effects of switching actions.
Adjacent comparisons only use records from the same episode and slice; no
future-state column is treated as an issued network command.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def audit_recorded_actions(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = ("episode_id", "slice_type", "action", "timestamp_s")
    missing = [x for x in required if x not in frame]
    if missing:
        raise ValueError(f"missing audit columns: {missing}")
    group = [x for x in (
        "scenario", "training_config", "experiment", "base_station", "slice_type",
    ) if x in frame]
    fields = list(dict.fromkeys([*group, *required]))
    data = frame.loc[:, fields].copy()
    # Episode identifiers and slices are the boundary of causal chronology.
    data["action"] = data["action"].astype(str)
    data["episode_id"] = data["episode_id"].astype(str)
    data = data.sort_values(
        ["episode_id", "slice_type", "timestamp_s"], kind="mergesort",
    )
    cell = data.groupby(["episode_id", "slice_type"], sort=False, observed=True)
    previous = cell["action"].shift()
    changed = previous.notna() & data["action"].ne(previous)
    # Timestamp ties cannot be treated as sequential interventions.
    prev_time = cell["timestamp_s"].shift()
    valid_adjacent = prev_time.notna() & (data["timestamp_s"] > prev_time)
    data["adjacent_change"] = changed & valid_adjacent
    data["duplicate_or_nonmonotone_timestamp"] = (
        prev_time.notna() & ~valid_adjacent
    )
    if not group:
        group = ["slice_type"]
    grouped = data.groupby(group, observed=True, sort=True)
    result = grouped.agg(
        rows=("action", "size"),
        episodes=("episode_id", "nunique"),
        distinct_actions_across_episodes=("action", "nunique"),
        valid_adjacent_change_rows=("adjacent_change", "sum"),
        timestamp_tie_rows=("duplicate_or_nonmonotone_timestamp", "sum"),
    ).reset_index()
    distinct_within = cell["action"].nunique()
    summary = {
        "rows": len(data),
        "episodes_by_slice": int(len(distinct_within)),
        "episodes_with_multiple_recorded_actions": int((distinct_within > 1).sum()),
        "adjacent_observed_action_changes": int(data["adjacent_change"].sum()),
        "timestamp_tie_rows": int(data["duplicate_or_nonmonotone_timestamp"].sum()),
        "distinct_recorded_action_labels": int(data["action"].nunique()),
        "within_episode_action_effects_identified": False,
        "interpretation": (
            "Cross-episode action variety is NOT counterfactual switch evidence. "
            "Only actual adjacent within-episode action changes could support "
            "a later intervention analysis, subject to confounding and timing."
        ),
    }
    return result, summary
