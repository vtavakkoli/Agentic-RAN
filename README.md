# Agentic-RAN: End-to-End KPM + Synthetic Time-Series Training

This repository now supports:
- deterministic synthetic dataset generation,
- robust KPM ingestion from `dataset-kpm/...`,
- end-to-end train/predict/report pipelines,
- and 8 concurrent Docker scenario runs writing to `results/<scenario>/`.

## Standard output contracts
- `shared_data/traffic_data.csv` (synthetic dataset)
- `results/model/` (single-run model artifacts)
- `results/predictions/` (single-run predictions)
- `results/final/report.html` (single-run report)
- `results/<scenario_name>/...` (compose scenario outputs)

## Deterministic seed
Single source of truth: `oran_sim/seed.py` (`SEED=42`).

## Smoke-test flow (no KPM required)
### Bash (Linux/macOS)
```bash
python generate_data.py --steps 5000 --output shared_data/traffic_data.csv
python -m scripts.train --data_root shared_data/traffic_data.csv --out_dir results/model --seq_len 32 --feature_count 12 --model ridge
python -m scripts.predict --model_dir results/model --input shared_data/traffic_data.csv --output results/predictions/preds.csv
python -m scripts.report --metrics results/model/metrics.json --preds results/predictions/preds.csv --out results/final/report.html
docker compose up --build
```

### PowerShell (Windows)
```powershell
python generate_data.py --steps 5000 --output shared_data/traffic_data.csv
python -m scripts.train --data_root shared_data/traffic_data.csv --out_dir results/model --seq_len 32 --feature_count 12 --model ridge
python -m scripts.predict --model_dir results/model --input shared_data/traffic_data.csv --output results/predictions/preds.csv
python -m scripts.report --metrics results/model/metrics.json --preds results/predictions/preds.csv --out results/final/report.html
docker compose up --build
```

## KPM data usage
```bash
python -m scripts.train --data_root dataset-kpm --out_dir results/model --seq_len 32 --feature_count 12 --model hgb
python -m scripts.predict --model_dir results/model --input dataset-kpm --output results/predictions/preds.csv
```

## Docker scenarios
`docker compose up --build` launches exactly 8 services concurrently:
- `scenario_1` ... `scenario_8`

Each scenario logs started/finished markers to stdout and writes artifacts under:
- `results/scenario_i/model/`
- `results/scenario_i/predictions/`
- `results/scenario_i/final/report.html`

## Tests
```bash
python -m unittest discover -s tests -v
```
