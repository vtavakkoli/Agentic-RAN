# Architecture

## Control objective

Agentic-RAN selects one bounded network policy from a curated catalog for each RAN KPI observation. The design separates probabilistic recommendation from deterministic control authority.

## Decision stages

1. **Observe:** validate a typed KPI snapshot.
2. **Propose:** the compact multinomial model returns policy probabilities.
3. **Expand:** add mandatory heuristic and balanced candidates so low model confidence cannot hide critical actions.
4. **Simulate:** estimate policy effects on latency, loss, throughput, energy, handover failures, SINR, and PRB pressure.
5. **Guard:** enforce hard safety rules independently of model probability.
6. **Critique:** record why candidates were rejected.
7. **Select:** rank safe candidates by slice-aware utility plus bounded proposal confidence.
8. **Explain:** return the chosen action, predicted KPIs, candidate table, and trace.

## Trust boundaries

- **Untrusted/advisory:** learned proposal probabilities.
- **Controlled:** policy catalog and impact parameters.
- **Authoritative:** validation, safety rules, and final selector.
- **External:** live network actuation. This repository does not directly change a RAN.

## Slice-aware utility

The evaluator uses different KPI priorities:

- URLLC emphasizes latency and packet loss.
- eMBB emphasizes demand satisfaction and throughput.
- mMTC emphasizes access reliability, stability, and energy efficiency.

The score is intentionally bounded to 0–100 and is used only to rank candidates for the current observation.

## Deployment

Docker Compose implements a deterministic dependency chain:

```text
dataset (download/fallback) -> trainer -> api
                                  └-----> test (optional profile)
```

The API container runs as a non-root user with a read-only filesystem, dropped Linux capabilities, and no-new-privileges. Runtime artifacts are stored in named volumes.

## Production extension points

- Replace the bootstrap data with approved KPI streams.
- Replace the surrogate impact function with a calibrated simulator or digital twin.
- Add O-RAN A1/E2 adapters behind an explicit approval gate.
- Persist immutable decision/audit records.
- Add drift monitoring and model promotion controls.
- Add multi-cell coordination and conflict resolution.
