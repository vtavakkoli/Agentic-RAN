# Agentic-RAN v2 Architecture

## Objective

Agentic-RAN selects and, only when explicitly authorized, executes bounded RAN policies under heterogeneous SLA, energy, and safety constraints. The design separates prediction from authority so that learned models can fail without automatically gaining unsafe control capability.

## Control cycle

1. **Observe** — receive a typed normalized KPI snapshot from synthetic, replay, srsRAN JSON, Prometheus, or decoded E2SM-KPM telemetry.
2. **Assess input quality** — evaluate telemetry completeness, age, and OOD score.
3. **Forecast** — optionally estimate near-future demand/PRB/latency as advisory context.
4. **Propose** — retrieve top-k policies from the learned proposer and add mandatory safe/heuristic candidates.
5. **Plan** — simulate each candidate for a configurable horizon.
6. **Critique** — independent critics assess safety, SLA, energy, temporal stability, uncertainty, and optionally inter-cell interference.
7. **Guard** — reject candidates that violate authoritative constraints.
8. **Pareto rank** — identify non-dominated safe choices across latency/loss/throughput/energy/stability.
9. **Select** — maximize slice/intent-aware utility plus bounded proposal confidence.
10. **Authorize** — execution envelope checks confidence, OOD, uncertainty, mode, canary scope, allowed policy, and operator-approval requirements.
11. **Act** — recommend, shadow, simulate, or send a generic bounded control request to an external E2 bridge.
12. **Verify** — observe the next network state and evaluate rollback conditions.
13. **Audit** — append the entire step to a tamper-evident hash chain.

## Trust boundaries

```text
UNTRUSTED / ADVISORY
  learned proposer
  RL baseline
  forecaster
  optional LLM explainer
  external world-model predictions

CONTROLLED
  policy catalog
  SLA intents
  topology
  model registry metadata

AUTHORITATIVE
  schema validation
  action bounds
  independent critics
  OOD / uncertainty gate
  execution envelope
  canary allow-list
  rollback guard

EXTERNAL OPERATIONAL AUTHORITY
  standards-capable E2 bridge / xApp binding
  Near-RT RIC deployment policy
  operator approval process
```

## Telemetry architecture

All sources map to one stable `NetworkObservation` schema.

```text
SyntheticTelemetryProvider ─┐
CSVReplayTelemetryProvider ─┤
SrsRANWebSocketProvider ────┤
E2KPMProvider ──────────────┼──► NetworkObservation ─► AgenticPolicyEngine
PrometheusTelemetryProvider ┘
```

The srsRAN mapper is alias-tolerant because JSON layout and available metrics can evolve. Defaults are explicit, and `telemetry_completeness` lowers autonomous confidence when important measurements are missing.

## World-model architecture

`RANWorldModel` is the counterfactual boundary.

Implementations:

- `SurrogateWorldModel` — transparent offline-safe default;
- `TraceReplayWorldModel` — policy-specific deltas learned/estimated from recorded experiments;
- `NS3WorldModelAdapter` — JSON bridge to ns-3 or an ns-3-based simulator;
- `SrsRANDigitalTwinAdapter` — JSON bridge to a calibrated srsRAN lab/twin service;
- any future learned dynamics model implementing the same `predict` contract.

The world model returns a trajectory, an uncertainty score, and a model name. The final selector never treats a world-model prediction as ground truth.

## Safety model

Static guards enforce bounded transmit power and PRB changes, URLLC protection, weak-coverage protection, congestion safeguards, SINR floors, PRB overload limits, and packet-loss degradation limits. Temporal guards enforce minimum dwell time and switch-rate limits, while rollback evaluates measured post-action degradation. High or critical combined uncertainty blocks autonomous execution even when a recommendation remains available.

## O-RAN boundary

Agentic-RAN owns **decision semantics and safety authority**, not low-level E2 protocol encoding.

```text
Agentic-RAN action
       │
       ▼
E2RCActuator
       │ generic typed Style-2 bridge JSON
       ▼
standards-capable bridge / xApp binding
       │ ASN.1 + E2AP/E2SM-RC + SCTP
       ▼
Near-RT RIC / E2 node
```

This separation keeps the repository portable between O-RAN SC, FlexRIC, and other E2 stacks while preventing accidental claims that an HTTP JSON object is itself E2AP.

## xApp / rApp split

The Near-RT xApp receives normalized telemetry, performs short-horizon planning, runs safety critics, applies execution gating, and records rollback/audit state. The Non-RT rApp handles long-term telemetry, drift, model promotion, intent management, A1 publication, and cross-cell policy analysis.

## Multi-cell coordination

The coordinator applies neighbor-sensitive costs for positive transmit-power changes near low-SINR neighbors, aggressive throughput policy next to critically loaded neighbors, and power reduction during mobility instability. Replace the transparent default with calibrated interference prediction before claiming network-optimal control.

## Audit architecture

Each record contains `timestamp`, `event_type`, `previous_hash`, `payload`, and `record_hash`. The hash chain detects local alteration or deletion/reordering but is not a substitute for external immutable storage or trusted timestamping.
