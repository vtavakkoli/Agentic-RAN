# Benchmarks

Agentic-RAN v2 treats safety and closed-loop resilience as first-class metrics. Publication-quality experiments should export per-step rows containing seed, scenario, cell/slice, controller, state before/after, action, utility, SLA/safety status, uncertainty, OOD score, rollback, decision/transport latency, and model/policy hashes.

Use the same trace, seed, world model, intent, and action envelope for every controller comparison. See `docs/RESEARCH_BENCHMARK.md` for the recommended baselines and ablations.
