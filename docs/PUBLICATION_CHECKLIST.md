# Publication checklist

The publication benchmark is complete only when all of the following evidence is present:

- [ ] full configured COMMAG `slice_traffic` cells are discovered from the pinned upstream tree and every existing UE metrics file in those cells is prepared
- [ ] all configured train/validation/seen-test/unseen-test splits contain rows and have zero episode overlap
- [ ] proposed method, FQI, linear CQL, HistGradientBoosting behavior baseline, and original COMMAG PPO scheduler baseline are evaluated
- [ ] original PPO results are labeled scheduler-only with exogenous/logged PRB allocation
- [ ] unseen RF-distance and mobility OOD score/fallback rates are reported separately
- [ ] policy-shortcut diagnostic is reported with scheduler/PRB features removed
- [ ] behavior accuracy is reported specifically on policy-change timestamps
- [ ] paired episode bootstrap 95% confidence intervals are reported
- [ ] paired random-sign permutation p-values and Cohen's `dz` effect sizes are reported
- [ ] the exact Docker/image versions and benchmark seed are recorded
- [ ] energy is described only as a normalized proxy
- [ ] direct-method outcomes are described only as counterfactual estimates, not causal production gains
- [ ] no `PUBLICATION-BENCHMARK-READY` claim is made if the original PPO export/evaluation is missing
