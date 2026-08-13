# Changelog

All notable changes to Agentic-RAN are documented here.

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
