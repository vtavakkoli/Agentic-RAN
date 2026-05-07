# Agentic-RAN Pipeline

A streamlined Docker-first pipeline with four commands:

## Commands
```bash
docker compose up --build generate_data   # generate dataset
docker compose up --build train           # train models using generated dataset
docker compose up --build test            # test models and create comprehensive report
docker compose up --build run-all         # run generate_data, train, and test
```

## Outputs
- Dataset splits: `shared_data/splits/`
- Training artifacts and summary:
  - `results/train/train.html`
  - `results/train/train_summary.json`
  - model outputs in `results/`
  - trained weights in `ml_models/`
- Test artifacts and comprehensive report:
  - `results/test/test.html`

`run-all` executes the full end-to-end flow and produces both training and testing reports.
