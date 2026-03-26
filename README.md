# Agentic O-RAN KPM End-to-End Pipeline

This repository now supports end-to-end KPM processing with one consistent workflow:

1. Generate a unified dataset (exactly `N` rows) from `RESERVATION-*` folders.
2. Deterministically split train/val/test (`60%/30%/10%`).
3. Train a CPU baseline model.
4. Predict on test (or any reservation csv).
5. Run all 8 scenarios concurrently via Docker Compose.
6. Produce final HTML report at `results/final/report.html`.

## Required commands

### Dataset generation
```bash
python generate_data.py --steps 5000 --input shared_data/dataset-kpm --output shared_data/traffic_data.csv
```

Madrid LTE Zone I (frequencies `f796`, `f1815`, `f2650`) is also supported:
```bash
python generate_data.py --steps 5000 --input dataset/madrid-lte-dataset/zoneI --output shared_data/traffic_data.csv
```
This produces aligned per-second features such as:
`downlink_f796`, `uplink_f796`, `users_f796`, `downlink_f1815`, `uplink_f1815`, `users_f1815`, `downlink_f2650`, `uplink_f2650`, `users_f2650`.

### Train
```bash
python -m scripts.train --csv shared_data/traffic_data.csv --out_dir results/model --seed 42
```

### Predict
```bash
python -m scripts.predict --model_dir results/model --csv shared_data/traffic_data_test.csv --output results/predictions/preds.csv
```

### Final report
```bash
python -m scripts.report --preds results/predictions/preds.csv --metrics results/model/metrics.json --out results/final/report.html --config results/model/config.json
```

### Run all 8 scenarios concurrently
```bash
docker compose up --build
```

Each scenario emits stdout logs for:
- scenario started
- data generation started/done
- training started/done
- prediction started/done
- report generated (status file)

The aggregator service collects all scenario statuses and writes final report:
- `results/final/report.html`
- `results/final/scenario_status.csv`

## Output tree

```text
shared_data/
  traffic_data.csv
  traffic_data_train.csv
  traffic_data_val.csv
  traffic_data_test.csv

results/
  data/traffic_data_summary.json
  model/
    model.joblib
    config.json
    features.json
    metrics.json
  predictions/preds.csv
  scenarios/<scenario>/
    status.json
    model/metrics.json
    preds.csv
  final/
    report.html
    scenario_status.csv
```


## Run tests
```bash
pip install -r requirements.txt
pytest -q
```
