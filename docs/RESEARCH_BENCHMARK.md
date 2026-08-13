# Research Benchmark Protocol

## Primary question

Can a safety-governed agentic controller improve heterogeneous RAN utility while maintaining explicit SLA and safety constraints under disturbances, uncertainty, and distribution shift?

## Controllers to compare

Minimum recommended set: static balanced policy; heuristic policy controller; learned proposer without world-model planning; Fitted-Q offline RL; constrained Fitted-Q; MPC/external world-model baseline when available; and full Agentic-RAN.

For publication-quality work, add PPO/SAC/constrained PPO using the same observation/action envelope and safety-filter comparison protocol.

## Ablations

Compare the full model against variants without the world model, OOD, uncertainty, SLA critic, temporal guard, rollback, multi-cell coordination, and Pareto/intent weighting.

## Scenarios

Run normal operation and all built-in fault scenarios: traffic spike, cell outage, backhaul degradation, poor radio, handover storm, packet-loss burst, stale telemetry, corrupted KPI, model drift, and neighbor interference.

## Metrics

Report throughput, demand satisfaction, latency, packet loss, handover failures, unsafe proposal rate, critic rejection rate, autonomous approval rate, SLA violations, rollback rate, policy churn, recovery time, OOD behavior, energy/throughput tradeoff, decision latency, and transport latency.

## Statistical reporting

Use repeated seeds and report mean and standard deviation, bootstrap 95% confidence intervals, paired comparisons where scenarios are shared, failure counts, and full configuration/model/policy hashes. Do not claim universal superiority from one traffic trace or simulator configuration.
