# Agentic O-RAN NAS Traffic Prediction Simulation

A structurally extensible, Dockerized framework that simulates:
- **Non-RT RIC / rApp NAS orchestration** (scenario definitions + complexity-aware model selection inputs)
- **Near-RT RIC / xApp agentic inference** (training/inference + ReAct control loop)

## Project structure (extensible)

```text
.
├── oran_sim/
│   ├── config.py        # scenario catalog + feature order
│   ├── data.py          # synthetic dataset generation + preprocessing
│   ├── models.py        # LSTM/Attention/Liquid implementations + complexity math
│   ├── training.py      # reusable training routine
│   ├── agent.py         # ReAct loop (Thought/Action/Monitor)
│   └── reporting.py     # markdown tables + chart generation
├── generate_data.py     # entrypoint: create shared_data/traffic_data.csv
├── xapp_agent.py        # entrypoint: run one scenario in one container
├── evaluate_nas.py      # entrypoint: aggregate + rank + report
├── docker-compose.yml   # 8 concurrent services (one per scenario)
├── Dockerfile
└── requirements.txt
```

## Scenarios (8)
1. `lightweight-32`
2. `lightweight-64`
3. `balanced-small`
4. `balanced-medium`
5. `deep-performance`
6. `ultra-performance`
7. `attention-baseline`
8. `liquid-baseline`

## Math implemented

- **Exact LSTM layer complexity**:
  \[
  C_{LSTM}=4\times(d_xd_h+d_h^2+d_h)
  \]
- Stacked LSTM total = sum of layer-wise complexities.
- Attention/Liquid use generalized complexity proxies for cross-architecture comparison.
- NAS efficiency:
  \[
  C_{norm}=\frac{C_{model}}{\max(C_{model})}, \quad E=\frac{C_{norm}}{P_{norm}}
  \]
  where `P_norm` is derived from normalized `R^2`.

## Run

### 1) Generate data or read from data-kpm
```bash
python generate_data.py --steps 5000 --output shared_data/traffic_data.csv
```
```bash
python generate_data.py --step 10000 --input shared_data/dataset-kpm --output shared_data/traffic_data.csv
```

### 2) Launch all 8 scenario containers concurrently
```bash
docker compose up --build
```

Each service reads `shared_data/traffic_data.csv` and appends its metrics to `shared_data/results.csv`.

### 3) Build NAS report (tables + charts)
```bash
python evaluate_nas.py \
  --input shared_data/results.csv \
  --output shared_data/nas_efficiency.csv \
  --report shared_data/nas_report.md \
  --chart_dir shared_data/charts
```

Generated outputs:
- `shared_data/nas_efficiency.csv`
- `shared_data/nas_report.md` (comparison tables)
- `shared_data/charts/r2_vs_complexity.png`
- `shared_data/charts/efficiency_ranking.png`
