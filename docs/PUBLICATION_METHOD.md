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

## Experiment-level cross-fitted policy/OPE roles

The earlier configuration-level policy/OPE split could leave the independent evaluator without support for many scheduler+PRB actions selected by the policy. The final protocol therefore cross-fits by **experiment**, not by training configuration.

For each experiment `E`:

1. the proposed controller and reproduced baselines are fitted on `tr0`–`tr11` rows from experiment `E`;
2. the independent outcome/OPE evaluator is fitted on `tr0`–`tr11` rows from the other experiment(s);
3. validation rows from experiment `E` calibrate thresholds, CQL alpha and hysteresis;
4. seen/unseen final-test rows from experiment `E` are evaluated exactly once with that fold.

With the two COMMAG experiments this produces a role swap:

```text
Fold A: policy/baselines <- exp1   | independent OPE <- exp2
Fold B: policy/baselines <- exp2   | independent OPE <- exp1
```

The role split is therefore independent while both sides retain all configured training configurations.

## Positivity / common action support

Counterfactual direct-method estimates are reported only for scheduler+PRB actions observed on **both** sides of the fold. For every slice, the executable action set is the intersection of policy-fit and OPE-fit support.

The proposed controller, FQI, CQL and behavior baseline are all constrained to this common support before primary evaluation. If the current action is outside common support, the selector falls back to a supported slice-specific action.

The benchmark exports, per fold:

- policy-side action cells;
- OPE-side action cells;
- common action cells;
- fraction of policy action cells retained by common support;
- minimum number of common actions available per slice;
- the exact common action list and fallback per slice.

Readiness requires both:

- at least 80% of policy action cells retained by common support; and
- at least two common actions per slice.

This prevents the evaluator-support gate from being satisfied merely by collapsing the action space to a trivial single action.

## Independent outcome/OPE evaluator

The independent evaluator is a separate `HistGradientBoostingRegressor` per objective (SLA, energy proxy, stability). Within its own fit role, quality is measured by grouped cross-fitting using `scenario × training configuration × experiment` groups.

The policy critics never score their own final selected actions in the primary result table. Final counterfactual outcomes are produced by the opposite-experiment evaluator and only on common support.

`independent_ope_fit.json` stores cross-fit quality for every experiment fold, while `partition.json` stores the role-swap and support audit.

## Validation calibration

Validation remains an active, test-free part of the protocol. For each experiment fold, the predeclared grid in `configs/final_publication.yaml` freezes:

- SLA safety threshold;
- uncertainty threshold;
- planning weight;
- switching/hysteresis margin;
- OOD threshold from seen validation only;
- linear-CQL alpha.

Validation candidates that fail the evaluator-support gate are not eligible. If no candidate satisfies both SLA and support constraints, the run fails rather than silently choosing an unsupported setting.

Final seen/unseen test rows are never used for calibration.

## Reproduced baselines

The executable benchmark compares:

- **Proposed** — safety/OOD-aware multi-objective controller with validation-tuned hysteresis;
- **FQI** — ExtraTrees fitted-Q iteration over the multi-objective utility;
- **Linear CQL** — discrete linear Q-function with Bellman error plus conservative CQL penalty;
- **Behavior HGB** — supervised behavior-cloning diagnostic.

All counterfactual baselines use the same fold-specific common support and the same independent OPE evaluator. The logged policy is exported only as an observed-outcome reference.

## Primary statistical protocol

The primary inferential unit is:

```text
scenario × training configuration × experiment
```

Each final-test row appears in exactly one experiment cross-fit fold. For every proposed-versus-baseline comparison the workflow reports:

- equal-cluster mean utility difference;
- cluster bootstrap 95% confidence interval;
- paired random-sign permutation p-value;
- Cohen's `dz`;
- row-weighted utility difference as a secondary descriptive value.

UE/episode-level statistics remain secondary.

A statistically significant negative utility difference does **not** make the run invalid. Instead, the report prohibits a utility-superiority claim and requires a utility-versus-stability/churn trade-off interpretation.

## OOD/generalization protocol

The OOD analysis reports AUROC, AUPRC, FPR@95%TPR, threshold sensitivity, and scenario-level score/fallback behavior. The OOD threshold is calibrated only from seen validation data within each fold.

OOD is not a manuscript-readiness gate based on achieving a favorable AUROC. If aggregate AUROC is below 0.70, the report explicitly limits OOD to a secondary conservative diagnostic and prohibits a strong OOD-detection claim.

## Transition and shortcut audit

The benchmark audits scheduler, PRB and joint-action changes per scenario/configuration/experiment/base station.

If the prepared COMMAG traces contain no within-episode joint scheduler+PRB changes, the repository does not claim demonstrated dynamic closed-loop RAN control. The primary framing becomes **offline RAN configuration/policy selection**, and the feature-removal analysis is labeled **persistence/configuration shortcut sensitivity**.

## Churn / hysteresis and trade-off reporting

The proposed controller uses a validation-tuned switching margin. The final report exports `tradeoff_summary.json` with, for each baseline:

- independent-OPE utility delta;
- absolute churn reduction;
- SLA-violation delta;
- clustered effect statistics.

This supports a scientifically valid stability/safety trade-off paper even when the proposed method does not maximize scalar utility.

## Latency

The report includes critic-prediction, OOD and selection-logic probes plus per-decision host/container p50/p95/p99 latency. These measurements are not end-to-end RIC-to-gNB latency.

If host/container p95 exceeds 500 ms, the report adds a claim restriction against real-time RIC performance claims.

## Original COMMAG PPO: literature reference only

The original COMMAG PPO/DRL scheduler is not rerun. Bonati et al.'s published findings remain stored in `literature_reference.json` and are excluded from reproduced baseline tables and paired tests because the metrics, conditions and action semantics differ.

## Readiness semantics

Successful execution and scientific readiness remain separate:

- `run_status = EXPERIMENT-COMPLETE` means the pipeline finished;
- `evidence_status = READY-FOR-MANUSCRIPT` is emitted only when the methodological gates pass;
- otherwise `evidence_status = REVIEW-REQUIRED`.

The readiness gates test role separation, exactly-once test routing, validation isolation, evaluator quality, common-support coverage, evaluator support, clustered inference, OOD computation, transition audit, and independent scoring. They do **not** require a favorable or significant performance result.

## Self-contained HTML evidence

`results/publication/report.html` remains self-contained. In addition to the original figures it now explains the experiment role-swap, common action support, allowed manuscript claims, and claim restrictions.

## Counterfactual limitation

Only outcomes for logged COMMAG actions are observed. Alternative selected-action outcomes remain offline independent direct-method/OPE estimates, even after support correction. They are not causal online intervention effects. Energy remains a normalized proxy rather than measured joules. Unseen-condition testing establishes offline distribution-shift behavior; it does not establish live-RAN performance.
