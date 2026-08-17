# Publication method

## Full COMMAG slice-traffic coverage

The publication workflow discovers the pinned upstream Git tree and selects every existing `slice_traffic/.../*_metrics.csv` file in the configured scenario/training-configuration/experiment/base-station cells. It does not hard-code a representative UE subset. Preparation fails if an expected scenario/configuration/experiment/BS cell is missing.

The condition split is fixed before model fitting:

- training pool: `tr0`–`tr11` on `rome_static_close` and `rome_static_medium`;
- validation: `tr12`–`tr14` on the same seen conditions;
- seen final test: `tr15`–`tr17` on the same seen conditions;
- unseen RF-distance test: `rome_static_far`;
- unseen mobility test: `rome_slow_close`.

All four base stations and both configured experiments are included. Train/validation/seen-test/unseen-test episode overlap is a hard failure.

## Policy fitting versus independent OPE fitting

The training pool is partitioned again before any final evaluation. A deterministic coverage-maximizing subset of training configurations is reserved exclusively for the independent outcome/OPE evaluator; the remaining configurations fit the proposed controller and the reproduced baselines. The partition is exported to `partition.json` with policy-side and OPE-side action coverage.

The independent evaluator is a separate `HistGradientBoostingRegressor` model per objective (SLA, energy proxy, stability). Its grouped cross-fit quality is measured using `scenario × training configuration × experiment` groups and exported to `independent_ope_fit.json`. Final counterfactual selected-action outcomes are scored by this evaluator rather than by the policy critics that selected the action.

This design avoids the earlier circular evaluation pattern in which the controller could effectively be graded by the same critic family it optimized.

## Validation calibration

Validation is an active part of the protocol. A predeclared grid in `configs/final_publication.yaml` is evaluated with the independent OPE evaluator to freeze:

- SLA safety threshold;
- uncertainty threshold;
- planning weight;
- switching/hysteresis margin.

The OOD threshold is not tuned on unseen conditions. It is calibrated from the seen validation OOD-score distribution to the configured target false-positive rate. Linear-CQL alpha is selected on validation using the independent evaluator and then refit with the selected alpha.

The validation objective rewards independent-OPE utility while penalizing SLA violations, policy churn, and rollback. The chosen setting and every candidate are exported in `validation_selection.json` and `validation_calibration.csv`. Final seen/unseen test data are never used by calibration.

## Reproduced baselines

The executable benchmark compares:

- **Proposed** — safety/OOD-aware multi-objective controller with validation-tuned hysteresis;
- **FQI** — ExtraTrees fitted-Q iteration over the multi-objective utility;
- **Linear CQL** — discrete linear Q-function with Bellman error plus the CQL log-sum-exp conservative penalty;
- **Behavior HGB** — supervised behavior-cloning diagnostic.

The logged policy is exported as an **observed outcome reference**, not as another counterfactual method.

All counterfactual methods are scored using the same independent OPE evaluator. The behavior-HGB result remains primarily a shortcut/behavior diagnostic because current scheduler/PRB state can encode policy persistence.

## Original COMMAG PPO: literature reference only

The original COMMAG PPO/DRL scheduler is not rerun. The benchmark exports a literature-reference record for Bonati et al., “Intelligence and Learning in O-RAN for Data-driven NextG Cellular Networks,” IEEE Communications Magazine, 59(10), 21–27, 2021, DOI `10.1109/MCOM.101.2001120`, arXiv `2012.01263`.

The paper reports eMBB spectral-efficiency gains up to 20% over the best-performing static scheduler and URLLC average-buffer reductions of 37% vs RR, 5% vs WF and 17% vs PF. These values use different metrics, experimental conditions and action semantics. They are stored in `literature_reference.json` and excluded from reproduced baseline tables and statistical tests.

## Primary statistical protocol

The primary inferential unit is:

```text
scenario × training configuration × experiment
```

This avoids treating many UE traces from the same network experiment as independent replications. For each proposed-versus-baseline comparison the workflow reports:

- equal-cluster mean utility difference;
- cluster bootstrap 95% confidence interval;
- paired random-sign permutation p-value (exact when the number of clusters is small, Monte Carlo otherwise);
- Cohen's `dz` paired effect size;
- row-weighted utility difference as a secondary descriptive value.

UE/episode-level paired statistics are exported separately in `episode_statistics_secondary.json` and must not be presented as the primary significance analysis.

Monte-Carlo p-values are formatted according to their resolution instead of reporting a false level of numerical precision.

## OOD/generalization protocol

The final OOD analysis reports:

- AUROC for held-out seen versus unseen conditions;
- AUPRC;
- FPR at 95% TPR;
- seen false-positive and unseen true-positive rates at the validation-calibrated threshold;
- separate discrimination for `rome_static_far` and `rome_slow_close`;
- threshold-sensitivity table;
- mean/p95 OOD score and OOD fallback rate by scenario.

`rome_static_far` represents RF-distance shift and `rome_slow_close` represents mobility shift.

## Transition and shortcut audit

The benchmark explicitly audits, per scenario/configuration/experiment/BS:

- current scheduler diversity;
- current PRB-allocation diversity;
- scheduler-change rows;
- PRB-change rows;
- joint scheduler+PRB action-change rows and rates.

The HistGradientBoosting shortcut diagnostic still removes current scheduler, current PRB, and both fields. If the prepared traces contain no within-episode joint-action changes, policy-change-only accuracy is left unavailable rather than silently interpreted as zero. The report then labels the experiment a **persistence/configuration shortcut sensitivity** analysis.

## Churn / hysteresis

The proposed controller applies a validation-tuned switching margin. A proposed action change is held at the current supported action when its policy-predicted score improvement is smaller than the selected margin. Utility, SLA, churn, and rollback trade-offs used to choose this margin are recorded in the validation table.

## Latency

The report includes critic-prediction, OOD, and selection-logic batch probes plus end-to-end p50/p95/p99 per-decision host/container latency and runtime metadata. These are not claimed as end-to-end RIC-to-gNB latency measurements.

## Readiness semantics

Successful execution and scientific readiness are separated:

- `run_status = EXPERIMENT-COMPLETE` means the pipeline finished;
- `evidence_status = READY-FOR-MANUSCRIPT` is emitted only when explicit methodological gates pass;
- otherwise `evidence_status = REVIEW-REQUIRED` and the failed gates/warnings are rendered in JSON and HTML.

The status is not conditioned on obtaining a statistically significant favorable result.

## Self-contained HTML evidence

`results/publication/report.html` is generated without external JavaScript, CDN assets, remote fonts, or remote images. It contains inline SVG diagrams/charts for the architecture, independent-OPE outcomes, clustered effects, OOD behavior, validation calibration, transition/shortcut evidence, latency, and readiness gates.

## Counterfactual limitation

Only outcomes for logged COMMAG actions are observed. SLA, energy, stability and utility for alternative selected actions remain offline independent direct-method/OPE estimates. They are not causal online intervention effects. Energy is a normalized proxy rather than measured joules. Unseen-condition testing establishes offline distribution-shift behavior; it does not establish live-RAN performance.
