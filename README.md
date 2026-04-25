# Agentic RAN Scenario Benchmarking

## Project overview
This repository provides a **scenario-driven benchmark framework** for O-RAN data-driven forecasting experiments with reproducible Docker and local Python workflows.

The proposed method family is **Liquid Dynamics** (represented by `liquid-baseline`) and is benchmarked against:
- lightweight MLP (`lightweight-32`, `lightweight-64`)
- balanced MLP (`balanced-small`, `balanced-medium`)
- deep MLP (`deep-performance`)
- ultra-performance MLP (`ultra-performance`)
- attention-based sequence modeling (`attention-baseline`)
- xLSTM (`xlstm-baseline`)

## Dataset and attribution
### Documented input folder
Use `dataset/` as the canonical input location for raw CSV data preparation.

Expected layout:
- `dataset/slice_mixed/**/*.csv`
- `dataset/slice_traffic/**/*.csv`

`prepare_splits.py` also accepts `dataset/` directly and recursively scans CSVs.

### Tested reference dataset
Data preparation is tested with the **Colosseum O-RAN COMMAG Dataset** associated with:
> L. Bonati, S. D'Oro, M. Polese, S. Basagni, T. Melodia, “Intelligence and Learning in O-RAN for Data-driven NextG Cellular Networks,” IEEE Communications Magazine, vol. 59, no. 10, pp. 21–27, October 2021.

Please cite that paper if you use the dataset in a publication.

## Target column and feature handling
- The prepared benchmark dataset always uses an explicit target column named **`target`**.
- During preparation, you can explicitly set the raw target column with:
  ```bash
  python -m scripts.prepare_splits --target-col "tx_brate downlink [Mbps]"
  ```
- If `--target-col` is not set, preparation uses known candidates (`target`, `tx_brate downlink [Mbps]`, `dl_brate`, `rx_brate uplink [Mbps]`, `ul_brate`) and finally falls back to the last numeric column.
- **Actual source feature names are preserved** (no remapping to `feature_0`, `feature_1`, ...).

## Requirements
- Python **3.12**
- PyTorch (CPU-compatible build by default in Docker)
- Docker + Docker Compose

Dependencies are declared in `requirements.txt`.

## Repository structure
- `agentic_ran/`
  - `data_loading.py`: dataset loading and fallback behavior
  - `preprocessing.py`: feature extraction, scaling, sequence building, splitting
  - `models.py`: model factory and architectures
  - `training.py`: training loop
  - `evaluation.py`: metric computation and composite scoring
  - `reporting.py`: outputs and plots
  - `scenarios.py`: scenario catalog and hyperparameters
- `scripts/prepare_splits.py`: raw-data preparation and train/val/test split generation
- `scripts/run_scenario.py`: run one scenario
- `scripts/run_all.py`: end-to-end prepare + run + aggregate
- `scripts/aggregate_report.py`: final benchmark report generation (`results/report.html`)

## Experiments workflow
### 1) Prepare data
```bash
docker compose up --build prepare-data
```
Equivalent local command:
```bash
python -m scripts.prepare_splits \
  --input-dir dataset/slice_mixed \
  --input-dir dataset/slice_traffic \
  --output-dir shared_data/splits
```

### 2) Run a single scenario
```bash
docker compose up lightweight-32
```
or
```bash
python -m scripts.run_scenario --scenario lightweight-32
```

### 3) Run complete benchmark
```bash
docker compose up --build run-all
```

## Reproducibility guidance
- Keep a fixed random seed for data splitting (`split_and_save` uses seed 42 by default).
- Reuse the same files in `shared_data/splits/` across scenario runs.
- Pin epochs with `EPOCHS=<N>` when comparing architectures.
- Preserve run artifacts under `results/<scenario>/` (metrics, metadata, predictions, training logs, plots).
- Track `shared_data/splits/summary.json` to capture file provenance, source target columns, and selected feature names.

## Final report interpretation
The report in `results/report.html` includes cumulative and per-metric views.

- **Higher is better**: `R2`, `composite_score`
- **Lower is better**: `RMSE`, `MAE`, `MAPE`, `sMAPE`, `wMAPE`

The current benchmark report ranks **`liquid-baseline`** as best under cumulative composite score.
However, you should also inspect individual metrics separately because other baselines may win on specific metrics (e.g., stronger `R2` or lower `RMSE`).

## License
This project is licensed under the **MIT License**. See `LICENSE`.
