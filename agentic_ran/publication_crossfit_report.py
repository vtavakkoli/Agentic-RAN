"""Cross-fit-aware self-contained publication report."""
from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pandas as pd

from agentic_ran.publication_report import render_report


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _crossfit_section(plan: dict[str, Any]) -> str:
    folds = plan.get("folds", [])
    if not folds:
        return "<h2>Experiment cross-fit</h2><div class='panel'><p class='muted'>No fold metadata.</p></div>"
    rows = []
    for fold in folds:
        support = fold.get("common_support_by_slice", {})
        support_text = "; ".join(
            f"{slice_type}: {len(actions)} actions"
            for slice_type, actions in sorted(support.items())
        )
        rows.append(
            "<tr>"
            f"<td>{_esc(fold.get('fold',''))}</td>"
            f"<td>{_esc(fold.get('policy_fit_experiment',''))}</td>"
            f"<td>{_esc(', '.join(map(str, fold.get('ope_fit_experiments',[]))))}</td>"
            f"<td>{float(fold.get('common_policy_action_coverage',0.0)):.3f}</td>"
            f"<td>{int(fold.get('minimum_common_actions_per_slice',0))}</td>"
            f"<td>{_esc(support_text)}</td>"
            "</tr>"
        )
    return (
        "<h2>Experiment cross-fit and positivity support</h2>"
        "<div class='panel'>"
        "<p class='muted'>Each final-test experiment is evaluated exactly once. "
        "The policy/baselines are fitted on that experiment's training runs, while "
        "the independent OPE evaluator is fitted on the opposite experiment(s). "
        "All counterfactual methods are restricted to the intersection of policy-fit "
        "and OPE-fit action support per slice, preventing unsupported-action "
        "extrapolation from entering the primary statistics.</p>"
        "<div class='table-wrap'><table><thead><tr>"
        "<th>fold</th><th>policy experiment</th><th>OPE experiment(s)</th>"
        "<th>common/policy coverage</th><th>min common actions/slice</th>"
        "<th>common support</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div></div>"
    )


def render_crossfit_report(
    output: str | Path,
    *,
    summary: dict[str, Any],
    baselines: pd.DataFrame,
    clustered_statistics: dict[str, Any],
    ood: dict[str, Any],
    ood_detection: dict[str, Any],
    transition_audit: pd.DataFrame,
    shortcut: pd.DataFrame,
    calibration: pd.DataFrame,
    latency: dict[str, Any],
    literature_reference: dict[str, Any],
    crossfit_plan: dict[str, Any],
) -> Path:
    report = render_report(
        output,
        summary=summary,
        baselines=baselines,
        clustered_statistics=clustered_statistics,
        ood=ood,
        ood_detection=ood_detection,
        transition_audit=transition_audit,
        shortcut=shortcut,
        calibration=calibration,
        latency=latency,
        literature_reference=literature_reference,
    )
    text = report.read_text(encoding="utf-8")
    text = text.replace("disjoint configs", "experiment role-swap")
    text = text.replace(
        "Policy fitting, independent OPE fitting, validation calibration, and final testing use explicitly separated evidence roles. "
        "Clustered inference treats each scenario × training configuration × experiment as the primary experimental unit.",
        "Experiment-level role-swapped cross-fitting keeps policy fitting and OPE outcome fitting disjoint while preserving action coverage. "
        "Each final-test experiment is routed exactly once, and clustered inference treats each scenario × training configuration × experiment as the primary experimental unit.",
    )
    text = text.replace(
        "<h2>Readiness gates</h2>",
        _crossfit_section(crossfit_plan) + "<h2>Readiness gates</h2>",
        1,
    )
    claim_scope = summary.get("claim_scope", {})
    if claim_scope:
        items = "".join(
            f"<li><strong>{_esc(name)}</strong>: {_esc(value)}</li>"
            for name, value in claim_scope.items()
        )
        section = (
            "<h2>Allowed manuscript claims</h2><div class='panel'><ul>"
            + items
            + "</ul></div>"
        )
        text = text.replace(
            "<h2>Scientific interpretation</h2>",
            section + "<h2>Scientific interpretation</h2>",
            1,
        )
    report.write_text(text, encoding="utf-8")
    return report
