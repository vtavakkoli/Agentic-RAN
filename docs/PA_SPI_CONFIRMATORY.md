# PA-SPI: one-command expanded research campaign

## Launch from Windows PowerShell

After checking out the PA-SPI research branch, preserve any local corporate CA
configuration. Run from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_ieee_access_complete.ps1 -Preset research -ExpandedStudy
```

Defaults: original exploratory 10 x 40 x 4 x 6 campaign plus independent
20 x 100 x 6 x 17 PA-SPI campaign (204,000 new toy episodes / 6,528,000 new
toy steps at 32 steps per episode). Four independent training workers by
default, each with one Torch/BLAS thread. This is computationally substantial.
Use `-ConfirmatoryWorkers 2` if RAM is limited.

The workflow runs original regression/build/campaign, new PA-SPI tests and
training/validation/held-out comparison, full COMMAG prepare/evaluate, a
conditional support/abstention/OOD/latency audit, PULSE logged shadow,
native ns-3.49 UDP test, and final combined evidence report. The existing
optional `-ExternalPhysical` gate still requires a separately validated
PRB/MAC/PHY adapter and config; it is not silently emulated.

Functional smoke (not research evidence):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_ieee_access_complete.ps1 -Preset smoke -ExpandedStudy -SkipOffline -SkipNative
```

## New experiment contract

PA-SPI labels all feasible first actions via paired TRAIN-only synthetic
rollouts, following original PULSE for subsequent steps. Five ridge critics
resample complete episodes. Calibration uses separate episodes and empirical
residual quantiles for relative utility and per-slice risk; local support is a
nearest training-state distance since every action was branched. There is no
theorem asserting conditional safety, and an unsupported PULSE fallback is
explicitly logged, not labeled safe.

Validation chooses a positive improvement margin without changing original
PULSE or the one-percentage-point per-slice safety margin. The test seed range
is separate from the previously examined campaign. The primary comparison is
PA-SPI versus PULSE; secondary comparisons use Holm adjustment, train-run
cluster bootstrapping and sign flips. A failed hypothesis is a valid outcome.
All methods see the same realized toy dynamics, but training data and compute
budgets differ: PA-SPI has explicit branching simulator access, not offline
only access. Two shifted toy environments use different arrival/channel and
reward dynamics to stress generalization; neither is a calibrated NR model.
The no-margin ablation still uses the anchored representation and must not
be represented as a separately trained unanchored critic. Offline coverage is
a post-evaluation diagnostic, not a new causal estimator.

## Evidence files

- `confirmatory/protocol_frozen.json`: primary hypothesis, counts, seeds and margins
- `confirmatory/training_run_*/frozen_selection.json`: calibration and validation choices
- `confirmatory/training_run_*/episodes.csv.gz` and `steps.csv.gz`: executed toy outcomes and traces
- `confirmatory/rankings.csv`, `scenario_means.csv`, `training_run_means.csv`, `summary.json`, `report.html`
- `confirmatory/artifact_sha256.json`: checksums including frozen models and traces
- `commag-audit/conditional_overlap_by_split_slice.csv`: sample-weighted and selected-row-weighted overlap
- `commag-audit/offline_estimate_labels.csv`: estimated vs logged outcomes, unsupported outcomes unidentified
- `final-report/ieee_access_report.html`: links and verified summary of both separate toy campaigns and offline/native results

## Remaining research limitations

The new campaign is not yet a physical 5G-LENA experiment. The existing
native UDP integration cannot execute joint PRB/scheduler actions; that
requires a validated NR adapter with measured capacities and action receipts.
COMMAG counterfactual effects remain model-dependent; no propensities or
within-episode action transitions are fabricated. The audit relabels and
reweights overlap evidence but does not retrain the COMMAG OOD detector or
change its estimator. Independent novelty review, a power analysis and full
fresh-seed execution are still necessary. Run output is unique and never
overwritten; full cross-stage resume is not implemented. Do not tune on the
new test results and call the same seeds independent confirmation again.
