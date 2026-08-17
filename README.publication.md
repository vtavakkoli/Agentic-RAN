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

The prepared COMMAG table retains the existing condition split:

- `tr0`–`tr11`: training pool on `rome_static_close` and `rome_static_medium`;
- `tr12`–`tr14`: seen-condition validation;
- `tr15`–`tr17`: seen-condition final test;
- `rome_static_far`: unseen RF-distance shift;
- `rome_slow_close`: unseen mobility shift.

The final evaluator adds a second separation inside the training pool. A deterministic coverage-maximizing subset of training configurations is reserved exclusively for the independent outcome/OPE evaluator; the remaining training configurations fit the proposed controller and reproduced baselines. The two roles have disjoint episodes.

Validation is now used, rather than merely loaded. It freezes:

- SLA safety threshold;
- uncertainty threshold;
- OOD threshold, calibrated to a target false-positive rate on seen validation data only;
- planning weight;
- switching/hysteresis margin;
- linear-CQL alpha.

Seen and unseen test rows are not used during calibration.

## Primary evidence

Counterfactual outcomes for the proposed method, FQI, linear CQL, and HistGradientBoosting are scored by an **independent HistGradientBoosting outcome evaluator** trained on the reserved OPE subset. The policy's own critics are not used as the primary final evaluator.

Primary inference is clustered by:

```text
scenario × training configuration × experiment
```

UE/episode-level inference is exported only as a secondary sensitivity analysis. Cluster bootstrap confidence intervals, paired random-sign permutation tests, and Cohen's `dz` are reported.

OOD evaluation now includes AUROC, AUPRC, FPR@95%TPR, validation-calibrated threshold performance, per-unseen-scenario discrimination, and threshold sensitivity.

A transition audit checks scheduler changes, PRB changes, and joint-action changes. If the prepared traces contain no within-episode policy changes, the report no longer pretends policy-change-only accuracy exists; the feature-removal experiment is explicitly labeled a persistence/configuration shortcut diagnostic.

The proposed controller also receives a validation-tuned switching margin to reduce unnecessary policy churn.

## Self-contained HTML report

The run creates:

```text
results/publication/report.html
```

The HTML has no external JavaScript, CDN, font, or image dependency. It contains inline SVG figures for:

- the full experimental architecture;
- independent-OPE utility, SLA, energy, and churn comparisons;
- cluster-level effect-size forest plot;
- OOD behavior by scenario;
- validation calibration;
- transition/shortcut audit;
- latency/runtime evidence;
- readiness gates and claim restrictions.

## Output files

The final evidence directory includes:

- `report.html` — self-contained visual publication report;
- `publication_summary.json` — run status, manuscript-readiness gates, warnings, calibrated parameters;
- `publication_baselines.csv` — proposed / FQI / linear-CQL / HGB plus observed logged reference;
- `publication_decisions.csv.gz` — row-level independent-OPE estimates and selected actions;
- `clustered_statistics.json` — **primary** clustered inference;
- `paired_statistics.json` — compatibility alias for the primary clustered statistics;
- `episode_statistics_secondary.json` — secondary UE/episode sensitivity analysis;
- `independent_ope_fit.json` — grouped cross-fit MAE/R² of the independent evaluator;
- `partition.json` — policy-fit vs OPE-fit training configurations and action coverage;
- `validation_calibration.csv` / `validation_selection.json` — controller validation search and selected setting;
- `cql_validation.csv` — validation selection of CQL alpha;
- `ood_detection.json` / `ood_generalization.json` / `ood_threshold_sensitivity.csv`;
- `transition_audit.csv`;
- `policy_shortcut_test.csv`;
- `latency_profile.json`;
- `literature_reference.json` — published COMMAG PPO findings, reference only;
- `publication_models.joblib`.

## Status semantics

`publication_summary.json` no longer sets `PUBLICATION-BENCHMARK-READY` just because the program finished. It separates:

- `run_status: EXPERIMENT-COMPLETE`
- `evidence_status: READY-FOR-MANUSCRIPT` or `REVIEW-REQUIRED`

`evidence_status` is derived from explicit methodological gates such as disjoint policy/OPE fitting roles, independent-evaluator support, clustered inference, validation-only calibration, OOD evaluation, transition audit, and successful report generation.

## Counterfactual guardrail

Only logged COMMAG outcomes are observed. Alternative selected-action outcomes remain offline direct-method/OPE estimates, not causal online intervention effects. Energy is a normalized proxy rather than measured joules. Host/container inference latency is not end-to-end RIC-to-gNB latency.

See `docs/PUBLICATION_METHOD.md`, `docs/PUBLICATION_CHECKLIST.md`, and `docs/PAPER_REFERENCE_BASELINE.md` before using the generated evidence in a paper.
