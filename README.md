# Agentic O-RAN NAS Traffic Prediction Simulation

This repository provides a complete Dockerized simulation framework for proactive traffic prediction in an O-RAN-inspired setup:

- **Non-RT RIC / rApp (NAS orchestration):** encoded by scenario definitions in `model_zoo.py`.
- **Near-RT RIC / xApp (agentic inference):** implemented in `xapp_agent.py` with a ReAct loop (Thought, Action, Monitor).

## Implemented scenarios (8 containers)

1. `lightweight-32` (LSTM 1x32, 6 features)
2. `lightweight-64` (LSTM 1x64, 6 features)
3. `balanced-small` (LSTM 64x32, 8 features)
4. `balanced-medium` (LSTM 100x50, 8 features)
5. `deep-performance` (LSTM 128x100x64, 10 features)
6. `ultra-performance` (LSTM 512x256x128, 16 features)
7. `attention-baseline` (Transformer-like baseline, 8 features)
8. `liquid-baseline` (Liquid-style continuous-time RNN baseline, 6 features)

## File overview

- `generate_data.py`: Generates ~5000-step synthetic 16-feature multivariate traffic data.
- `model_zoo.py`: Dynamic model factory and complexity utilities.
- `xapp_agent.py`: Per-container train/inference + ReAct loop + metrics export.
- `evaluate_nas.py`: Aggregates and ranks scenarios with normalized efficiency.
- `Dockerfile`: Runtime image with required ML dependencies.
- `docker-compose.yml`: 8 concurrent services, one per scenario.

## Run instructions

### 1) Generate shared synthetic dataset

```bash
python generate_data.py --steps 5000 --output shared_data/traffic_data.csv
```

### 2) Build and run all model containers concurrently

```bash
docker compose up --build --abort-on-container-exit
```

This creates/updates:

- `shared_data/results.csv` (all scenario metrics)

### 3) Evaluate NAS efficiency after container completion

```bash
python evaluate_nas.py --input shared_data/results.csv --output shared_data/nas_efficiency.csv
```

Output:

- `shared_data/nas_efficiency.csv` with:
  - `c_norm = C_model / max(C_model)`
  - `p_norm` derived from normalized `R^2`
  - `efficiency_E = c_norm / p_norm`

## Notes on complexity formulas

- **LSTM formula (exactly implemented):**
  \[
  C_{LSTM} = 4 \times (d_x d_h + d_h^2 + d_h)
  \]
- For stacked LSTMs, this is summed per layer using the previous layer hidden size as next layer input.
- Attention and Liquid models use generalized operation proxies in `model_zoo.py` for cross-paradigm comparison.
