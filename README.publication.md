# Publication benchmark quick start

The final publication workflow runs on the normal `python:3.13-slim` image and keeps the original COMMAG PPO as a literature reference only.

Run the complete workflow:

```bash
docker compose -f docker-compose.publication.yml up --build --force-recreate publication-test
```

Dependency chain:

```text
prepare-full-commag -> publication-test
```

## Scientific protocol

The prepared COMMAG table retains the fixed condition split:

- `tr0`–`tr11`: training pool on `rome_static_close` and `rome_static_medium`;
- `tr12`–`tr14`: seen-condition validation;
- `tr15`–`tr17`: seen-condition final test;
- `rome_static_far`: unseen RF-distance shift;
- `rome_slow_close`: unseen mobility shift.

### Experiment-level cross-fitted policy/OPE roles

The final evaluator does **not** hold out whole training configurations for OPE. That older design could leave the independent evaluator without support for most selected scheduler+PRB actions.

Instead, the two COMMAG experiments are role-swapped:

```text
Fold A: policy/baselines <- exp1 | independent OPE <- exp2
Fold B: policy/baselines <- exp2 | independent OPE <- exp1
```

Each fold retains all `tr0`–`tr11` training configurations on both evidence roles. Validation and final-test rows are routed by the policy-fit experiment, so every final-test row is evaluated exactly once.

### Common action support / positivity

For each slice, the executable action set is restricted to scheduler+PRB actions observed in **both** the policy-fit and OPE-fit roles. The same common-support mask applies to:

- proposed controller selection;
- FQI;
- linear CQL;
- HGB behavior baseline normalization.

`partition.json` exports the exact per-fold support and readiness requires substantial non-trivial overlap, not merely a single supported action.

Validation freezes SLA/uncertainty/OOD thresholds, planning weight, switching/hysteresis margin, and linear-CQL alpha without using final-test data.

## Primary evidence

Counterfactual outcomes for the proposed method, FQI, linear CQL, and HGB are scored by an **independent HistGradientBoosting outcome evaluator fitted on the opposite experiment role**. The policy's own critics are never the primary final evaluator.

Primary inference is clustered by:

```text
scenario × training configuration × experiment
```

UE/episode-level inference is exported only as a secondary sensitivity analysis. Cluster bootstrap confidence intervals, paired random-sign permutation tests, and Cohen's `dz` are reported.

OOD evaluation includes AUROC, AUPRC, FPR@95%TPR, validation-calibrated threshold performance, per-unseen-scenario discrimination, and threshold sensitivity. A weak OOD result is treated as a claim restriction rather than hidden or promoted as a headline contribution.

A transition audit checks scheduler, PRB and joint-action changes. If no within-episode policy changes are observed, the manuscript framing is **offline RAN configuration/policy selection**, not demonstrated dynamic closed-loop control.

The proposed controller also uses a validation-tuned switching margin. `tradeoff_summary.json` reports utility-versus-churn/SLA trade-offs so a stable-policy contribution can be stated honestly even when scalar utility is not superior.

## Self-contained HTML report

The run creates:

```text
results/publication/report.html
```

The HTML has no external JavaScript, CDN, font, or image dependency. It contains inline SVG figures and tables for:

- experiment role-swap architecture;
- positivity/common-support evidence;
- independent-OPE utility, SLA, energy, and churn comparisons;
- cluster-level effect-size forest plot;
- OOD behavior by scenario;
- validation calibration;
- transition/shortcut audit;
- latency/runtime evidence;
- readiness gates;
- allowed manuscript claims and restrictions.

## Output files

The final evidence directory includes:

- `report.html` — self-contained visual publication report;
- `publication_summary.json` — run/evidence status, readiness gates, warnings, claim scope;
- `publication_baselines.csv` — proposed / FQI / linear-CQL / HGB plus observed logged reference;
- `publication_decisions.csv.gz` — row-level independent-OPE estimates and selected actions;
- `clustered_statistics.json` — **primary** clustered inference;
- `paired_statistics.json` — compatibility alias for primary clustered statistics;
- `episode_statistics_secondary.json` — secondary UE/episode sensitivity analysis;
- `independent_ope_fit.json` — grouped cross-fit MAE/R² for each OPE experiment fold;
- `partition.json` — experiment role-swap plus policy/OPE/common action support;
- `tradeoff_summary.json` — utility delta, churn reduction, SLA delta, clustered effects;
- `validation_calibration.csv` / `validation_selection.json`;
- `cql_validation.csv`;
- `ood_detection.json` / `ood_generalization.json` / `ood_threshold_sensitivity.csv`;
- `transition_audit.csv`;
- `policy_shortcut_test.csv`;
- `latency_profile.json`;
- `literature_reference.json` — published COMMAG PPO findings, reference only;
- `publication_models.joblib` — fold-specific policy/OPE/FQI/CQL models.

## Status semantics

`publication_summary.json` separates:

- `run_status: EXPERIMENT-COMPLETE`
- `evidence_status: READY-FOR-MANUSCRIPT` or `REVIEW-REQUIRED`

Readiness is based on methodology: experiment-disjoint roles, exactly-once test routing, validation isolation, finite evaluator metrics, non-trivial common support, evaluator support, clustered inference, OOD computation, transition audit, and successful report generation.

Readiness does **not** require a favorable p-value or utility win. If the proposed method loses scalar utility, the report automatically restricts utility-superiority claims and points to the measured stability/churn trade-off instead.

## Counterfactual guardrail

Only logged COMMAG outcomes are observed. Alternative selected-action outcomes remain offline direct-method/OPE estimates on observed common support, not causal online intervention effects. Energy is a normalized proxy rather than measured joules. Host/container inference latency is not end-to-end RIC-to-gNB latency.

See `docs/PUBLICATION_METHOD.md`, `docs/PUBLICATION_CHECKLIST.md`, and `docs/PAPER_REFERENCE_BASELINE.md` before using generated evidence in a manuscript.
