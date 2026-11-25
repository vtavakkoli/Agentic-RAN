"""Read-only audit of actual COMMAG output; no new counterfactual claims."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def audit(source: Path, out: Path):
    support = pd.read_csv(source / "conditional_support.csv")
    baseline = pd.read_csv(source / "publication_baselines.csv")
    transitions = pd.read_csv(source / "research_action_transition_audit.csv")
    ood = json.loads((source / "ood_detection.json").read_text(encoding="utf-8"))
    latency = json.loads((source / "latency_profile.json").read_text(encoding="utf-8"))
    keys = ["split", "method"]
    rows = []
    for key, group in support.groupby(keys):
        for dimension, subset in [("all", group), *[(str(k), g) for k, g in group.groupby("slice_type")]]:
            selected = subset.selected_rows.to_numpy(dtype=float)
            sampled = subset.sampled_rows.to_numpy(dtype=float)
            overlap = subset.joint_local_overlap.to_numpy(dtype=float)
            valid = np.isfinite(overlap) & (sampled > 0)
            total = float(selected.sum())
            valid_weight = float(selected[valid].sum())
            sample_weight = float(sampled[valid].sum())
            if not total or not valid_weight or not sample_weight:
                continue
            estimate = float(np.average(overlap[valid], weights=selected[valid]))
            # Missing group coverage is unknown, not zero or one. These
            # are bookkeeping bounds assuming within-stratum sample rates,
            # not statistical confidence or causal positivity bounds.
            low = estimate * valid_weight / total
            high = low + (total - valid_weight) / total
            rows.append({"split": key[0], "method": key[1], "slice": dimension,
                         "selected_rows": int(total), "sampled_rows": int(sampled.sum()),
                         "rows_in_sampled_strata": int(valid_weight),
                         "sample_weighted_overlap": float(np.average(overlap[valid], weights=sampled[valid])),
                         "selected_row_weighted_overlap": estimate,
                         "missing_strata_lower": low, "missing_strata_upper": high})
    cov = pd.DataFrame(rows)
    if cov.empty:
        raise ValueError("no conditional overlap evidence")
    out.mkdir(parents=True, exist_ok=True)
    cov.to_csv(out / "conditional_overlap_by_split_slice.csv", index=False)
    support.to_csv(out / "conditional_overlap_by_fold_action.csv", index=False)
    columns = ["split", "model", "estimate_type", "rows", "selected_ope_mean_utility",
               "selected_ope_sla_violation_rate", "evaluator_support_rate", "support_deferral_rate",
               "unsupported_incumbent_fallback_rate"]
    estimates = baseline[[c for c in columns if c in baseline]].copy()
    estimates = estimates.rename(columns={"evaluator_support_rate": "slice_action_set_overlap",
                                          "selected_ope_mean_utility": "reported_estimated_or_logged_utility",
                                          "selected_ope_sla_violation_rate": "reported_estimated_or_logged_sla"})
    estimates["unsupported_action_outcome"] = "unidentified"
    estimates.to_csv(out / "offline_estimate_labels.csv", index=False)
    transition_count = int(transitions.valid_adjacent_change_rows.sum())
    result = {"schema": "commag-conditional-support-audit-v1",
              "dynamic_action_changes_observed": transition_count,
              "conditional_overlap": cov[cov.slice.eq("all")].to_dict("records"),
              "ood": ood, "host_latency": latency,
              "unsupported_selected_action_outcomes": "unidentified",
              "can_identify_dynamic_policy_effects": False,
              "limitations": ["sampled kNN proxy, not propensity or causal identification",
                              "selected-row reweighting is descriptive, not a confidence interval",
                              "offline DM utility is model-estimated; logged observed is separate",
                              "no doubly robust/IPW estimate without justified propensities",
                              "no assumption that hold/defer produces a known safe outcome"]}
    (out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    audit(a.source, a.output)
