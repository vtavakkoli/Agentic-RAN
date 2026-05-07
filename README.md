# Agentic-RAN Benchmark

This repository benchmarks slice-aware RAN forecasting and safe agentic control.

## Main benchmark scope
The default benchmark focuses on:
- Time-aware tabular/residual forecasting models
- Strong gradient boosting baseline
- Graph-aware actor-critic and masked PPO control baselines
- **SafeGraphAgent-RAN** proposed method

Appendix temporal models remain available but are excluded from the default main scope.

## Commands
### Docker
```bash
docker compose up --build prepare-data
docker compose up --build benchmark-main
docker compose up --build benchmark-appendix
docker compose up --build benchmark-all
docker compose up --build report
```

### Local CLI
```bash
python -m src.benchmark --benchmark-scope main
python -m src.benchmark --benchmark-scope appendix
python -m src.benchmark --benchmark-scope all
python -m src.benchmark --benchmark-scope foundation
python -m src.report
```

## Outputs
- `results/main_benchmark.csv`
- `results/appendix_benchmark.csv`
- `results/model_ranking.csv`
- `results/control_ranking.csv`
- `results/safegraphagent_ran_metrics.csv`
- `results/report.html`

## Scientific note
Pseudo-label action metrics are not enough to prove real control quality. Use offline reward proxies, safety fallback behavior, and constraint adherence for control assessment.
