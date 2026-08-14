# Changelog

All notable changes to Agentic-RAN are documented here.

## 2.2.0 — 2026-08-14

### Added

- pinned Colosseum O-RAN COMMAG compact data profile without cloning the full upstream repository;
- 250 ms to one-second KPI aggregation and compressed sequential transition generation;
- experiment-separated offline Fitted-Q training and held-out value-model evaluation;
- scheduler/PRB action support guards, provenance, hashes and a self-contained COMMAG HTML report;
- `commag-prepare → commag-train → commag-test` Docker Compose workflow;
- explicit third-party data licensing and COMMAG citation documentation.

### Fixed

- hard-SLA fallback decisions are no longer force-labelled safe or approved for execution;
- `CITATION.cff` now represents Vahid Tavakkoli and Kabeh Mohsenzadegan as separate authors.

## 2.1.0 — 2026-08-13

### Added

- checksum-verified public 5G data catalog and downloader;
- real KPI normalization with compressed `data/prepared/*.csv.gz` outputs and provenance;
- explicit measured-versus-derived feature tracking and realism score;
- HistGradientBoosting, ExtraTrees, RandomForest and LogisticRegression production-candidate comparison;
- ranking across hold-out quality, real expert-reference agreement, perturbation robustness, calibration and latency;
- capped real expert-referenced augmentation for winner training;
- self-contained `results/report.html` and `results/report.json` readiness evidence;
- preferred Compose pipeline for `prepare-data → model-selection → report → test`;
- CI execution of the complete compact real-data workflow;
- optional catalog entry for the large TU Wien Vienna 4G/5G Drive-Test Dataset.

### Changed

- package and citation version updated to 2.1.0;
- README now distinguishes deterministic bootstrap evidence, public real-measurement shadow evidence and live RIC/gNB validation;
- real-data policy labels are explicitly expert-derived references, not operator ground truth;
- raw third-party datasets are downloaded locally rather than redistributed in Git.

## 2.0.0 — 2026-08-13

### Added

- short-horizon world-model planning and trajectory traces;
- pluggable surrogate, trace-replay, ns-3, and srsRAN twin adapters;
- uncertainty aggregation and out-of-distribution gating;
- independent SLA, energy, stability, uncertainty, and safety critics;
- temporal policy-churn protection;
- SLA intents and Pareto-front reporting;
- multi-cell interference-aware coordination;
- persistence, EWMA, trend, and ensemble forecasting;
- offline Fitted Q Iteration and constrained RL baseline;
- recommendation, shadow, simulated, canary, and active execution modes;
- E2SM-KPM normalized provider boundary;
- E2SM-RC Style-2 generic bridge actuator;
- srsRAN JSON WebSocket telemetry provider;
- Near-RT xApp and Non-RT rApp orchestration primitives;
- A1 policy transport adapter;
- rollback manager;
- tamper-evident hash-chain audit log;
- model registry, drift monitor, and promotion gate;
- fault-injection resilience benchmark;
- professional v2 API/dashboard;
- local development E2/twin bridge;
- srsRAN integration fragment and deployment documentation;
- expanded tests for v2 safety/control modules.

### Changed

- project positioning from a standalone policy selector to a safety-governed O-RAN research control plane;
- execution is explicitly gated and defaults to recommendation-only;
- README and architecture documentation now distinguish protocol conformance from bridge integration.

## 1.0.0

- initial lightweight policy-selection engine;
- Docker workflow, API, web demo, metrics, tests, and benchmark reporting.
