# Colosseum O-RAN COMMAG benchmark

## Purpose

This profile adds real O-RAN slice telemetry, scheduler configurations and dynamic radio-resource allocation to Agentic-RAN. It is designed as a reproducible offline-RL benchmark and does not claim that a policy has been validated on a live production network.

Upstream source: [wineslab/colosseum-oran-commag-dataset](https://github.com/wineslab/colosseum-oran-commag-dataset), pinned to commit `7331cd725fe42b5b9930fdc8acee3006cea00bd7`.

## Compact profile

The upstream repository is approximately 1.2 GB. The default Agentic-RAN core does not clone it. It downloads selected immutable CSV blobs covering:

- `slice_traffic/rome_slow_close`;
- training configurations `tr0`, `tr1` and `tr2`;
- experiment 1 for training and experiment 2 for testing;
- base stations 1 and 4;
- one representative UE trace for eMBB, mMTC and URLLC per base station.

The selection provides round-robin, water-filling and proportional-fair scheduling, plus several initial and dynamic PRB allocations. Set `AGENTIC_RAN_COMMAG_BASE_STATIONS=1` for a smaller CI/local smoke run.

## Preparation

```bash
agentic-ran prepare-commag
```

The preparer:

1. downloads files from the full pinned commit SHA;
2. records every source path, byte count and SHA-256 hash;
3. parses the upstream slice-metrics schema;
4. aggregates 250 ms observations into one-second measurements;
5. aligns the following scheduler/PRB configuration with the following KPI reward;
6. constructs sequential `state, action, reward, next_state, done` records;
7. assigns `exp1` episodes to `train` and `exp2` episodes to `test`;
8. writes deterministic Gzip-compressed CSV and a provenance manifest.

The action format is `<scheduler>:prb=<allocation>`. State variables include slice type, current scheduler/allocation, users, bitrate, buffers, MCS, errors, CQI, SINR and requested/granted PRBs.

## Reward

The normalized reward follows the upstream experiment objectives:

- eMBB: downlink throughput relative to the documented 1 Mbps/UE offered load;
- mMTC: downlink throughput relative to the documented 30 packets/s × 125-byte load;
- URLLC: granted/requested PRB ratio;
- all slices: a bounded downlink-error penalty.

The reward definition is transparent and should be replaced or sensitivity-tested for a specific research question.

## Offline training and testing

```bash
agentic-ran train-commag
agentic-ran validate-commag
```

Fitted Q Iteration uses Extra Trees as the nonlinear value approximator. Candidate actions are restricted to the scheduler/PRB combinations observed for the same slice in the training split. The held-out report includes:

- immediate reward-model MAE and R²;
- Q versus logged discounted-return MAE and R²;
- Bellman residual MAE;
- logged-action agreement;
- logged mean reward;
- a clearly labelled direct-method selected-policy reward estimate.

`BENCHMARK-READY` means that the dataset, split, support and numeric validity gates passed. It does not mean production-ready, causally superior or safe for live execution.

## Why PPO is not trained here

PPO is on-policy. Sampling fixed rows from an observational CSV does not create a valid PPO environment because the selected action does not control the next state. The upstream repository includes old TensorFlow PPO inference artifacts, but those are not retrained or copied into Agentic-RAN.

PPO should be added only with an interactive Colosseum, ns-3, srsRAN or calibrated Gymnasium environment in which an action changes subsequent telemetry. The same held-out scenarios, reward definition and safety filter can then be used for a fair Fitted-Q versus PPO comparison.

## Docker

```bash
LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) \
  docker compose --profile commag up --build commag-test
```

This executes `commag-prepare → commag-train → commag-test` and writes `results/commag_report.html`. Mapping the host UID/GID keeps the containers non-root while allowing them to write the bind-mounted `data/`, `artifacts/` and `results/` directories on Linux.

## License and citation

The upstream COMMAG repository declares GPL-3.0. Agentic-RAN does not redistribute its raw or prepared data. Runtime downloads and generated local outputs retain the upstream provenance and license metadata.

Publications using the profile should cite:

> L. Bonati, S. D'Oro, M. Polese, S. Basagni, and T. Melodia, “Intelligence and Learning in O-RAN for Data-driven NextG Cellular Networks,” IEEE Communications Magazine, 59(10), 21–27, 2021.
