# Publication checklist

The publication benchmark is complete only when all of the following evidence is present:

- [ ] multi-scenario COMMAG data prepared from the pinned upstream revision
- [ ] all configured train/validation/seen-test/unseen-test splits contain rows and disjoint episodes
- [ ] proposed method, FQI, conservative FQI/CQL-style baseline, supervised behavior baseline, and exact original PPO export evaluated
- [ ] unseen-scenario OOD score/fallback rates reported
- [ ] policy-shortcut diagnostic reported with scheduler/PRB features removed
- [ ] metrics on policy-change timestamps reported
- [ ] paired episode bootstrap 95% confidence intervals reported
- [ ] paired sign-permutation p-values and effect sizes reported
- [ ] energy described only as a proxy
- [ ] direct-method outcomes described only as counterfactual estimates, not causal production gains
