# Original COMMAG PPO baseline

The upstream COMMAG repository publishes the three slice-specific SavedModel policies (`embb_policy`, `mtc_policy`, `urllc_policy`) and `encoder.h5`. The publication workflow loads those exact artifacts in `Dockerfile.ppo-legacy` and exports scheduler decisions before the main benchmark runs.

## Reproducible replay adapter

`scripts/export_original_ppo.py` follows the release preprocessing relevant to inference:

- discard rows with zero requested PRBs;
- scale downlink buffer by `100000`;
- derive granted/requested PRB ratio and clip it to `[0,1]`;
- build ten observations per slice for the encoder;
- append the current slice PRB value to the encoded observation;
- invoke the corresponding published TensorFlow SavedModel policy.

For reproducible joining to the transition table, the adapter forms deterministic per-second groups instead of using the release script's random sampling loop. Therefore the repository uses the **exact published model weights and input transformation**, but does not claim bit-for-bit reproduction of the original demonstration loop.

## Action-space caveat

The original PPO outputs a scheduler code. The Agentic-RAN publication policy uses a joint `scheduler:prb=<n>` action. To compare them without inventing a PPO PRB controller, evaluation combines the PPO scheduler output with the logged next-step slice PRB allocation. Results are consequently labeled `original_commag_ppo_scheduler` and must be interpreted as a scheduler-only baseline with exogenous PRB allocation.

## Docker chain

The normal publication command performs the PPO export automatically:

```bash
docker compose -f docker-compose.publication.yml up --build publication-test
```

The dependency chain is:

```text
prepare-full-commag -> original-ppo -> publication-test
```

The exported scheduler decisions are stored at `data/prepared/commag-publication/original_ppo_actions.csv.gz` and then scored with the same direct-method critics and paired episode statistics as the other baselines.
