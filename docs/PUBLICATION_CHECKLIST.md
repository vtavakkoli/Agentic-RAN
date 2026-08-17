# Publication checklist

The publication benchmark is complete only when all of the following evidence is present:

- [ ] full configured COMMAG `slice_traffic` cells are discovered from the pinned upstream tree and every existing UE metrics file in those cells is prepared
- [ ] all configured train/validation/seen-test/unseen-test splits contain rows and have zero episode overlap
- [ ] proposed method, FQI, linear CQL, and HistGradientBoosting behavior baseline are evaluated reproducibly
- [ ] original COMMAG PPO results are cited only as literature-reference values from Bonati et al. (2021), not as a reproduced baseline
- [ ] literature-reference PPO values are excluded from `publication_baselines.csv` and paired statistical tests
- [ ] unseen RF-distance and mobility OOD score/fallback rates are reported separately
- [ ] policy-shortcut diagnostic is reported with scheduler/PRB features removed
- [ ] behavior accuracy is reported specifically on policy-change timestamps
- [ ] paired episode bootstrap 95% confidence intervals are reported
- [ ] paired random-sign permutation p-values and Cohen's `dz` effect sizes are reported
- [ ] the Docker base image and benchmark seed are recorded
- [ ] the publication Docker workflow builds with `python:3.13-slim`
- [ ] energy is described only as a normalized proxy
- [ ] direct-method outcomes are described only as counterfactual estimates, not causal production gains
