# COMMAG multi-objective offline policy study

This study layer extends the existing pinned COMMAG/Fitted-Q benchmark into a paper-oriented, experiment-separated evaluation of conservative RAN policy selection.

## What it adds

The study trains four independent learned critics over state-action pairs:

- **SLA critic** — predicts a normalized slice-aware SLA risk. eMBB emphasizes downlink throughput and errors; mMTC emphasizes low-rate delivery, grant success and errors; URLLC emphasizes grant success, errors and buffer pressure.
- **Energy critic** — predicts a normalized **energy proxy** derived from power multiplier, PRB grant/load and slice pressure. It is not a joule measurement.
- **Stability critic** — predicts policy-change and abrupt-KPI-transition cost.
- **Uncertainty critic** — predicts grouped out-of-fold residual magnitude from the three objective critics.

An Isolation Forest over robust-scaled RAN state features supplies an independent OOD score. On high OOD risk, or when all candidate actions fail the safety gate, the selector uses a conservative sticky fallback: keep the current supported action when possible, otherwise use the lowest-risk supported action learned from training traces.

A Fitted-Q model supplies a bounded planning term. The final selector minimizes SLA, energy, stability and uncertainty cost while using Fitted-Q only as a small look-ahead preference.

## Experimental separation

The existing COMMAG preparation remains authoritative:

- `exp1` transitions are used for fitting critics, OOD calibration, residual adaptation, baselines and Fitted-Q planning.
- `exp2` transitions remain held out.
- Episode overlap is rejected by a hard validity gate.
- Held-out actions without training support are rejected.

The default compact profile keeps the pinned upstream COMMAG revision and does not redistribute raw COMMAG files.

## Baselines

The held-out report compares the multi-objective selector with:

1. Logistic Regression
2. Random Forest
3. ExtraTrees
4. HistGradientBoosting

The four supervised baselines predict the logged next policy from state features. Their selected actions are then scored with the same objective critics for a consistent direct-method comparison. Logged-action macro-F1 and agreement are reported separately from estimated policy utility.

## Ablations

Every evaluation produces the requested matched ablations:

- `without_safety_gate`
- `without_ood_gate`
- `without_planning`
- `without_real_data_adaptation`

`without_real_data_adaptation` disables slice/action residual calibration learned exclusively from COMMAG training episodes. It does not leak held-out `exp2` outcomes into training.

## Robustness protocol

The default evaluation trains and evaluates **30 independent random seeds**. Each seed re-fits objective critics, the uncertainty critic, OOD model, Fitted-Q planner and all four supervised baselines on the same experiment-separated training split. The report exports mean, standard deviation and a normal-approximation 95% confidence-interval half-width for the main metrics.

The report also exports per-held-out-episode metrics to make trace heterogeneity visible.

## Metrics

The main report includes:

- host/container decision latency (p50/p95/max)
- batch decision throughput
- logged observed SLA-violation rate
- direct-method estimated selected-policy SLA-violation rate
- logged and selected energy proxy
- stability cost
- policy churn
- rollback/fallback rate
- OOD fallback rate
- safety fallback and override rate
- uncertainty and OOD scores
- logged-action agreement
- direct-method estimated utility and uplift

### Counterfactual interpretation

Only outcomes associated with the logged action are observed in fixed COMMAG traces. Outcomes for a policy action that was not logged are model estimates. Therefore:

- selected-policy SLA, energy, stability and utility are **direct-method estimates**;
- estimated uplift is **not** an observed intervention effect;
- the offline report does **not** establish causal production improvement;
- active/canary control still requires a validated interactive RIC/gNB/Colosseum/ns-3/srsRAN environment, operator approval and rollback exercises.

## Docker workflow

Create writable host directories once:

```bash
mkdir -p data/raw/commag data/prepared/commag artifacts/offline-policy \
  results/prepare-data results/train results/test
```

The Compose pipeline uses four explicit one-shot stages:

```text
prepare-data -> prepare-report -> train -> test
```

Run the complete dependency chain:

```bash
LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) \
  docker compose -f docker-compose.offline-study.yml up --build test
```

Do not use `--abort-on-container-exit`: the dependency services are intentionally expected to exit successfully before the next stage starts.

On Windows PowerShell:

```powershell
docker compose -f docker-compose.offline-study.yml up --build test
```

For a clean retry after changing the Compose definition:

```bash
docker compose -f docker-compose.offline-study.yml down --remove-orphans
docker compose -f docker-compose.offline-study.yml up --build --force-recreate test
```

Or run all stages explicitly without dependency re-execution:

```bash
LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) \
  docker compose -f docker-compose.offline-study.yml run --rm --no-deps prepare-data

LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) \
  docker compose -f docker-compose.offline-study.yml run --rm --no-deps prepare-report

LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) \
  docker compose -f docker-compose.offline-study.yml run --rm --no-deps train

LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) \
  AGENTIC_RAN_STUDY_SEEDS=30 \
  docker compose -f docker-compose.offline-study.yml run --rm --no-deps test
```

COMMAG source downloads are cached at `data/raw/commag`: an existing non-empty file is reused. The preparation, training and evaluation stages still rerun so reports and model artifacts reflect the current code and configuration.

## Outputs for a paper

### `results/prepare-data/`

- `report.html` — dataset validity/provenance report
- `summary.json` — machine-readable preparation summary
- `action_support.csv` — slice/action support table

### `results/train/`

- `report.html` — critic/OOD/fallback training report
- `metrics.json` — training metadata and critic fit metrics
- `critic_fit.csv` — independent critic fit table
- `action_support.csv` — training support table

### `results/test/`

- `report.html` — comprehensive held-out paper-style report
- `metrics.json` — complete held-out summary and caveats
- `decisions.csv.gz` — row-level held-out policy decisions
- `baseline_comparison.csv` — final-seed comparison
- `baselines_per_seed.csv` / `baseline_summary.csv` — 30-seed baseline evidence
- `ablations.csv` — final-seed ablations
- `ablations_per_seed.csv` / `ablation_summary.csv` — 30-seed matched ablations
- `per_seed.csv` / `seed_summary.csv` — robustness evidence
- `per_episode.csv` — held-out episode heterogeneity

The CSV and JSON files are intentionally kept separate from the HTML so tables, confidence intervals and statistical tests can be regenerated for a paper without scraping the report.

## Reproducibility knobs

The default configuration is `configs/offline_policy.yaml`. For a lighter developer smoke test, reduce `estimators`, `robustness_seeds` and `latency_samples`; paper runs should retain at least 20–30 seeds and document any changed thresholds or objective weights.
