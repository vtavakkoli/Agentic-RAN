# Publication checklist

The final benchmark is manuscript-ready only when the generated readiness gates and the following evidence are present:

- [ ] full configured COMMAG `slice_traffic` cells are discovered from the pinned upstream tree and every existing UE metrics file in those cells is prepared
- [ ] train/validation/seen-test/unseen-test splits contain rows and have zero episode overlap
- [ ] the training pool is partitioned into disjoint policy-fit and independent-OPE-fit configuration sets
- [ ] policy-fit and OPE-fit action coverage are exported in `partition.json`
- [ ] independent OPE uses grouped cross-fitting and exports finite MAE/R² evidence in `independent_ope_fit.json`
- [ ] proposed, FQI, linear CQL, and HGB counterfactual outcomes are scored by the independent evaluator, not the policy critics
- [ ] logged outcomes are labeled observed and kept distinct from counterfactual estimates
- [ ] `tr12`–`tr14` validation is actually used to freeze controller thresholds/planning/hysteresis before final tests
- [ ] OOD threshold is calibrated on seen validation data only
- [ ] linear-CQL alpha is selected on validation and the full validation search is exported
- [ ] seen/unseen final test rows are not used during calibration
- [ ] primary paired inference clusters by `scenario × training configuration × experiment`
- [ ] cluster bootstrap 95% confidence intervals are reported
- [ ] paired random-sign permutation tests and Cohen's `dz` are reported at the cluster level
- [ ] UE/episode-level statistics, if shown, are labeled secondary sensitivity analyses
- [ ] OOD AUROC, AUPRC, FPR@95%TPR, calibrated-threshold rates, and threshold sensitivity are reported
- [ ] RF-distance and mobility shifts are reported separately
- [ ] transition audit reports scheduler, PRB, and joint-action change counts/rates
- [ ] policy-shortcut diagnostic removes scheduler/PRB features
- [ ] policy-change-only accuracy is shown only when actual within-episode action changes exist; otherwise the diagnostic is labeled persistence/configuration shortcut sensitivity
- [ ] validation-tuned switching/hysteresis evidence and final churn are reported
- [ ] latency report includes p50/p95/p99 plus component probes and runtime metadata
- [ ] latency is described as host/container inference, not end-to-end RIC-to-gNB latency
- [ ] original COMMAG PPO results are cited only as literature-reference values from Bonati et al. (2021)
- [ ] literature-reference PPO values are excluded from reproduced baseline tables and statistical tests
- [ ] energy is described only as a normalized proxy
- [ ] alternative-action outcomes are described only as offline OPE/counterfactual estimates, not causal production gains
- [ ] `run_status` and `evidence_status` are separate; successful execution alone never implies publication readiness
- [ ] `results/publication/report.html` exists and is self-contained with inline SVG figures and no external CDN/JavaScript dependency
- [ ] the Docker workflow runs on `python:3.13-slim`
