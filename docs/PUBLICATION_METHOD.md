# Publication method

## Full COMMAG slice-traffic coverage

The publication workflow discovers the pinned upstream Git tree and selects every existing `slice_traffic/.../*_metrics.csv` file in the configured scenario/training-configuration/experiment/base-station cells. It does not hard-code a three-UE subset. Preparation fails if an expected scenario/config/experiment/BS cell is missing, so an incomplete download cannot silently become a "full" benchmark.

The default split is deliberately condition separated:

- training configurations `tr0`–`tr11` on `rome_static_close` and `rome_static_medium`;
- validation configurations `tr12`–`tr14` on the same seen conditions;
- seen-condition test configurations `tr15`–`tr17` on the same seen conditions;
- unseen RF-distance test on `rome_static_far`;
- unseen mobility test on `rome_slow_close`.

All four base stations and both configured experiments are included. Train/validation/seen-test/unseen-test episode overlap is a hard failure.

## Baselines

The benchmark evaluates all policies with the same held-out direct-method objective critics so the comparison target is consistent.

- **Behavior HGB** — strongest supervised behavior-cloning diagnostic from the compact study.
- **FQI** — ExtraTrees fitted-Q iteration over the multi-objective utility.
- **Linear CQL** — a discrete linear Q-function trained with Bellman error plus the CQL log-sum-exp conservative penalty. It is labeled `cql_linear` to avoid implying a deep-network reproduction of a particular external CQL implementation.
- **Original COMMAG PPO scheduler** — the exact published upstream SavedModel policies and encoder are loaded in a legacy TensorFlow container. The replay adapter follows the upstream feature transformations but uses deterministic per-second groups so decisions can be joined to held-out transitions reproducibly.

The original COMMAG PPO chooses the scheduler, not the joint scheduler+PRB action used by the proposed controller. For the comparison, the PPO scheduler decision is combined with the logged next-step slice PRB allocation. The report must describe this as a scheduler baseline with exogenous PRB allocation, not as an identical joint-action policy.

## Unseen-condition / OOD experiment

OOD evidence is reported separately for seen and unseen conditions and per unseen scenario. The report exports mean OOD score, p95 OOD score and OOD fallback rate. `rome_static_far` measures RF-distance shift; `rome_slow_close` measures mobility shift relative to the static training scenarios.

## Statistical protocol

Primary policy comparisons are paired by held-out episode rather than treating model seeds as independent network experiments. For each proposed-vs-baseline comparison the workflow exports:

- mean paired episode utility difference;
- episode bootstrap 95% confidence interval;
- paired random-sign permutation p-value;
- Cohen's `dz` paired effect size.

Random-seed refits remain useful as implementation-robustness evidence, but should not be described as independent network trials.

## Policy-shortcut diagnostic

A perfect next-action classifier can be caused by persistence of the current scheduler/PRB state. The publication benchmark therefore retrains HistGradientBoosting using:

1. full state;
2. state without current scheduler;
3. state without current PRB allocation;
4. state without both scheduler and PRB allocation.

It reports macro-F1 and agreement on all rows and separately on timestamps where the logged policy actually changes. This directly tests whether high behavior-cloning accuracy is primarily a persistence shortcut.

## Counterfactual limitation

Only outcomes for actions logged in COMMAG are observed. SLA, energy, stability and utility for alternative selected actions are direct-method critic estimates. They are not causal online intervention effects. Energy remains a normalized proxy rather than a joule measurement. Unseen-condition testing establishes distribution-shift behavior of the offline models and safety gates; it does not establish live-RAN performance.
