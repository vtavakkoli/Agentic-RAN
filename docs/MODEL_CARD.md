# Model Card: Agentic-RAN Policy Proposer

## Purpose

The model proposes candidate network policies from a compact RAN KPI snapshot. It is not the final controller: deterministic simulation, safety guardrails, and utility ranking decide the output.

## Model

- Histogram gradient-boosting classifier
- Standardized numeric features
- One-hot encoded slice type
- Balanced class weights
- CPU-only inference

## Training data

The included bootstrap dataset is synthetic and deterministic. Labels come from transparent expert rules covering congestion, latency pressure, weak coverage, dense mMTC access, eMBB demand, low-load energy saving, and balanced operation.

## Intended use

- software tests;
- policy-engine demonstrations;
- offline architecture experiments;
- integration prototypes with a simulator or lab RIC.

## Out-of-scope use

- autonomous production-network actuation without operator validation;
- subscriber-level decisions;
- safety or performance claims for a commercial RAN;
- use of bootstrap metrics as real operator ground truth.

## Evaluation

Training writes `results/training_metrics.json`. End-to-end benchmarking writes `results/benchmark.json` and `results/benchmark.html`. Regenerate all metrics on the target platform and approved data.

## Risk controls

- Pydantic range and schema validation
- independent hard guardrails
- mandatory balanced fallback
- explicit rejection reasons
- complete candidate trace
- bounded action parameters

## Known limitations

The policy impact model is deliberately simple. It does not model inter-cell interference, mobility trajectories, multi-agent conflicts, hardware constraints, delayed effects, or non-stationary network behavior.
