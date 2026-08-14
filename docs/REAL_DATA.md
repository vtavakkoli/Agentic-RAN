# Real-data benchmark workflow

Agentic-RAN can complement its deterministic bootstrap dataset with public 5G measurement data. The goal is external realism and shadow evaluation, not to claim that public measurements contain operator control ground truth.

For scheduler/PRB actions, sequential transitions and offline Fitted-Q evaluation, use the separate pinned COMMAG profile described in `docs/COMMAG.md`.

## Default sources

The source catalog is `configs/data_sources.yaml`. It currently includes the Telenor/COMMECT 5G private-network forestry measurements and the Glasgow 5G Dataset 2025. Raw files are fetched from the original Zenodo records, checksum-verified, and kept under `data/raw/`.

## Prepared outputs

`agentic-ran prepare-real-data` writes:

- `data/prepared/real_measurements.csv.gz`: normalized measured KPIs.
- `data/prepared/real_policy_eval.csv.gz`: complete Agentic-RAN feature rows with metadata showing which inputs were measured and which were derived.
- `data/prepared/provenance.json`: DOI, source URL, license field, checksums, row counts and measured-field counts.

The compressed CSV.GZ format keeps the prepared dataset small, streamable and dependency-light.

## Measured versus derived features

The public sources measure useful quantities such as RSRP, throughput, ping/latency and jitter, but do not expose every internal RAN counter required by the policy engine. Missing values such as PRB utilization, active-user count, packet loss, energy load, handover-failure rate or SINR are completed by transparent deterministic proxies in `agentic_ran/real_features.py`.

Every prepared row records `measured_fields`, `observed_model_features`, `derived_model_features`, and `realism_score`. This prevents an imputed value from being presented as a field measurement.

The `policy_label` column on real data is produced by the repository's transparent expert policy rule. It is an evaluation reference, not an operator action and not causal production ground truth.

## Model selection

`agentic-ran select-model` compares four CPU-friendly candidates:

1. histogram gradient boosting;
2. Extra Trees;
3. Random Forest;
4. multinomial logistic regression.

Ranking combines synthetic hold-out macro-F1, real expert-reference macro-F1, prediction stability under KPI perturbation, probability calibration and inference latency. The selected model is retrained with synthetic data plus a capped amount of real expert-referenced data so the public-data proxy cannot dominate training.

## Reproducible Docker workflow

Docker Compose prefers the repository's `compose.yaml`.

```bash
docker compose up prepare-data
docker compose up model-selection
docker compose up report
docker compose --profile test up --abort-on-container-exit test
```

The end products are written to `results/`, especially `results/report.html` and `results/report.json`.

## Readiness interpretation

The report can classify the repository as `NEEDS CALIBRATION`, `RESEARCH-READY`, or `SHADOW-READY`. Even `SHADOW-READY` is not a production certification. Active or canary RAN control still requires a calibrated RIC/gNB lab, authenticated transport, operator approval, rollback drills, long-duration shadow validation and independent safety/security review.
