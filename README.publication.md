# Publication benchmark quick start

```bash
docker compose -f docker-compose.publication.yml up --build publication-test
```

Outputs are written to `results/publication/` and raw COMMAG files are cached under `data/raw/commag/`.

See `docs/PUBLICATION_METHOD.md`, `docs/PUBLICATION_CHECKLIST.md`, and `docs/ORIGINAL_PPO_BASELINE.md` before using the results in a paper.
