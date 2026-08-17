# Publication benchmark quick start

Run the complete publication workflow:

```bash
docker compose -f docker-compose.publication.yml up --build publication-test
```

The dependency chain is:

```text
prepare-full-commag -> original-ppo -> publication-test
```

`prepare-full-commag` discovers every existing UE metric file in the configured pinned COMMAG `slice_traffic` cells. Raw source files are cached under `data/raw/commag/`, so subsequent runs reuse non-empty downloads. The prepared multi-scenario transition table is written under `data/prepared/commag-publication/`.

The publication outputs are written to `results/publication/`, including:

- `publication_baselines.csv` — proposed / behavior-HGB / FQI / linear-CQL / original-PPO comparison;
- `publication_decisions.csv.gz` — row-level held-out decisions and direct-method outcomes;
- `ood_generalization.json` — seen/unseen and per-scenario OOD/fallback evidence;
- `policy_shortcut_test.csv` — behavior-cloning shortcut diagnostics;
- `paired_statistics.json` — episode-paired bootstrap CI, permutation p-value and Cohen's `dz`;
- `publication_summary.json` — benchmark verdict and methodological caveats.

The full workflow is intentionally much larger than the earlier compact COMMAG benchmark. Disk/network use depends on the files present in the pinned upstream cells; cached raw files are not downloaded again.

See `docs/PUBLICATION_METHOD.md`, `docs/PUBLICATION_CHECKLIST.md`, and `docs/ORIGINAL_PPO_BASELINE.md` before using the results in a paper.
