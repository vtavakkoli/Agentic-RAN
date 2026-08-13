# O-RAN Integration Guide

## Design principle

Agentic-RAN does not implement E2AP ASN.1/SCTP in Python. It integrates through a transport boundary so an O-RAN SC, FlexRIC, or equivalent xApp/RIC stack can own protocol conformance while Agentic-RAN owns policy reasoning and safety.

## srsRAN telemetry path

srsRAN Project can expose JSON metrics over WebSocket. Configure JSON metrics and remote control in the gNB, then run:

```bash
agentic-ran xapp-shadow --url ws://127.0.0.1:8001 --steps 20
```

The provider sends:

```json
{"cmd": "metrics_subscribe"}
```

and normalizes available values into `NetworkObservation`.

A sample configuration fragment is included in `integrations/srsran/gnb-agentic-ran.yml`.

## E2SM-KPM path

Use a standards-capable RIC/xApp decoder and feed decoded indication dictionaries into `E2KPMProvider`. Missing metrics must lower `telemetry_completeness`; do not silently treat defaults as measured ground truth.

## E2SM-RC path

`E2RCActuator` emits a generic bridge request containing `service_model`, `control_style_type`, cell/slice identity, decision identity, selected policy, and bounded action parameters. The external bridge must validate target capabilities, map supported parameters, encode the target service model, use E2AP/SCTP through the chosen RIC framework, and return an explicit result.

## Compatibility note

O-RAN and srsRAN support evolves by version. The external bridge is therefore capability-aware. Do not assume scheduler names, PRB deltas, transmit-power changes, or every policy field map one-to-one to an E2SM-RC parameter in a particular release.

## Recommended integration sequence

```text
Level 0  synthetic telemetry
Level 1  recorded srsRAN replay
Level 2  live srsRAN JSON metrics, shadow decisions
Level 3  decoded E2SM-KPM, shadow decisions
Level 4  E2SM-RC development bridge / digital twin
Level 5  isolated RIC/gNB lab canary
Level 6  multi-cell coordinated lab control
Level 7  Non-RT rApp + A1 intent/model workflow
```
