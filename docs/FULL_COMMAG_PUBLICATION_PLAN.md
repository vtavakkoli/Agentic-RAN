# Full COMMAG publication benchmark

This branch extends the compact COMMAG study into a publication-oriented benchmark over the complete configured `slice_traffic` evidence from the pinned upstream revision.

## Scope

- discover every existing UE metrics file rather than hard-coding representative IMSIs;
- cover all 18 published training configurations (`tr0`–`tr17`), both experiments, and BS1–BS4;
- use `rome_static_close` and `rome_static_medium` as seen conditions;
- reserve `tr12`–`tr14` for validation and `tr15`–`tr17` for seen-condition testing;
- test RF-distance shift on `rome_static_far` and mobility shift on `rome_slow_close`;
- compare the proposed safety-aware policy against HistGradientBoosting behavior cloning, FQI, linear CQL, and the original published COMMAG PPO scheduler;
- report OOD/fallback behavior per unseen condition;
- run policy-shortcut diagnostics with current scheduler/PRB state removed and on policy-change timestamps only;
- use paired episode bootstrap confidence intervals, sign-permutation tests, and Cohen's `dz`;
- keep observed logged outcomes separate from direct-method counterfactual estimates.

The workflow intentionally keeps raw COMMAG files out of this repository. Source files are cached locally under `data/raw/commag` and are fetched from the pinned upstream commit only when missing.
