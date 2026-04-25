# Agentic RAN Scenario Benchmarking

## Project overview
This repository now provides a **scenario-driven benchmark framework** for agentic deep neural network experimentation with a reproducible Docker workflow. It runs multiple model profiles, writes per-scenario outputs, and builds a final benchmark dashboard (`results/report.html`) summarizing which architecture performs best.

## Requirements
- Python **3.12**
- PyTorch (CPU-compatible build by default in Docker)
- Docker + Docker Compose

Python dependencies are declared in `requirements.txt`.

## Repository structure
- `agentic_ran/`: reusable core modules
  - `data_loading.py`: dataset loading (from `shared_data/` CSV files or synthetic fallback)
  - `preprocessing.py`: feature engineering, scaling, sequence building, train/val/test split (default 60/10/30)
  - `models.py`: model factory and architectures (MLP, Attention, Liquid, xLSTM)
  - `training.py`: shared training loop and logging
  - `evaluation.py`: metrics computation and scoring
  - `reporting.py`: predictions/metadata/plots export helpers
  - `scenarios.py`: scenario catalog and hyperparameters
- `scripts/run_scenario.py`: runs exactly one scenario
- `scripts/aggregate_report.py`: aggregates all scenario outputs and writes `results/report.html`
- `docker-compose.yml`: multi-service scenario execution + aggregator
- `Dockerfile`: Python 3.12 runtime

## Docker setup
Build image:
```bash
docker compose build
```
If you previously built an older image and see an error like `No module named scripts.prepare_splits`, rerun with `--build` or rebuild first.

The image is tagged as `agentic-ran:latest` and mounts:
- `./shared_data -> /app/shared_data`
- `./results -> /app/results`

Set training epochs globally via environment variable:
```bash
EPOCHS=5 docker compose up lightweight-32
```

## Docker Compose usage
The compose file includes the following services:
- prepare-data (converts `slice_mixed/` + `slice_traffic/` into `shared_data/splits/{train,val,test}.csv` with a 60/10/30 split)
- lightweight-32
- lightweight-64
- balanced-small
- balanced-medium
- deep-performance
- ultra-performance
- attention-baseline
- liquid-baseline
- xlstm-baseline
- aggregator
- run-all (single command for prepare-data + all scenarios + aggregator)

Each scenario service executes:
```bash
python -m scripts.run_scenario --scenario <scenario-name>
```
after `prepare-data` has generated split CSV files.

The `aggregator` service runs:
```bash
python -m scripts.aggregate_report
```
after scenario services complete.

## Scenario descriptions
- **lightweight-32** / **lightweight-64**: compact MLP models for fast baseline checks.
- **balanced-small** / **balanced-medium**: deeper MLP profiles balancing quality and speed.
- **deep-performance** / **ultra-performance**: larger MLP stacks for high-capacity performance testing.
- **attention-baseline**: Transformer-encoder style sequence regressor.
- **liquid-baseline**: lightweight liquid/dynamical recurrent baseline.
- **xlstm-baseline**: bidirectional LSTM sequence baseline.

## Run one scenario
```bash
docker compose up prepare-data lightweight-32
```
or locally:
```bash
python -m scripts.run_scenario --scenario lightweight-32
```


## Create a lightweight dataset and train
If `shared_data/` has no CSV files, training falls back to synthetic data automatically.
To explicitly create a small dataset and train with it:

```bash
python -c "from pathlib import Path; import numpy as np, pandas as pd; rng=np.random.default_rng(42); n=3000; x=rng.normal(size=(n,10)); y=(x @ np.array([0.8,-1.1,0.4,0.6,-0.2,1.3,-0.5,0.7,0.9,-0.3])) + 0.6*np.sin(x[:,0]*x[:,1]) + 0.15*rng.normal(size=n); df=pd.DataFrame(x, columns=[f'feature_{i}' for i in range(10)]); df['target']=y; Path('shared_data').mkdir(parents=True, exist_ok=True); df.to_csv('shared_data/generated_training_dataset.csv', index=False)"
python -m scripts.run_scenario --scenario lightweight-32
```

## Shortcut commands
```bash
# 1) prepare train/val/test split files from slice_mixed + slice_traffic
docker compose up --build prepare-data

# 2) run full pipeline (prepare-data + all scenarios + aggregator)
docker compose up --build run-all
```

## Run all scenarios + aggregator
```bash
docker compose up --build prepare-data lightweight-32 lightweight-64 balanced-small balanced-medium deep-performance ultra-performance attention-baseline liquid-baseline xlstm-baseline aggregator
```

## Output layout and artifacts
Each scenario writes to its dedicated folder, e.g.:
- `results/lightweight-32/`
- `results/attention-baseline/`

Typical outputs per scenario:
- `metrics.json`
- `predictions.csv`
- `training_log.csv`
- `model_metadata.json`
- `data_summary.json`
- `status.json`
- `plots/predictions_vs_truth.png`
- `plots/training_curve.png`

Aggregator output:
- `results/report.html`

## How aggregation works
`aggregate_report.py` scans all scenario folders, checks run status and expected artifacts, builds a comparison table, computes a leaderboard from the composite benchmark score, identifies the best scenario, and renders a clean HTML dashboard with scientific summary + conclusion.

## How to interpret the final report
- **Higher is better**: `R2`, `composite_score`.
- **Lower is better**: `RMSE`, `MAE`, `MAPE`, `sMAPE`, `wMAPE`.
- Use the comparison table for traceability (model type, profile, dataset size, feature count).
- Use the leaderboard for quick ranking.
- Read the final conclusion for architecture-level recommendation under the current data/training budget.
