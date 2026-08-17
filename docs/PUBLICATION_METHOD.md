# Publication method

## Data split

The benchmark uses configuration- and scenario-separated evaluation. Training configurations and scenarios are declared in `configs/full_commag_publication.yaml`; validation configurations are never used as held-out paper test results; seen-test configurations measure interpolation/generalization within observed scenario families; unseen-test scenarios measure distribution shift.

## Baselines

- supervised behavior cloning remains a diagnostic baseline
- FQI provides a reward-maximizing offline RL baseline
- conservative FQI applies a support-frequency penalty as a lightweight CQL-style conservative baseline
- original COMMAG PPO is evaluated only from exact upstream decisions/weights through the compatibility adapter; a silent reimplementation is not called "original PPO"

## Statistical protocol

Primary comparisons are paired by held-out episode. The benchmark reports mean paired utility delta, episode bootstrap 95% confidence interval, paired sign-permutation p-value, and Cohen's dz. Seed refits remain useful for implementation robustness but are not treated as independent network experiments.

## Shortcut test

Because the next action often persists from the current scheduler/RBG configuration, the benchmark retrains the strongest behavior classifier after removing current scheduler, current PRB allocation, and both. It also reports performance only on timestamps where the logged policy actually changes.

## Counterfactual limitation

Policy outcomes for actions not logged in COMMAG are direct-method estimates from learned critics. They are not online intervention measurements. The unseen-scenario experiment evaluates model/generalization and fallback behavior, not causal live-network gains.
