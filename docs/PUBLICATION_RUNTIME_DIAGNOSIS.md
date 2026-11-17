# Publication performance and failure diagnosis (Windows / Docker Desktop)

## Confirmed from the November 13 run

- The standard pytest run finished (147 passed, 1 skipped).
- Full publication evaluation failed because the optional physical JAGC study expected `/workspace/configs/jagc_phy_bwp.json`, which was absent. This must **not** be replaced with made-up PRB capacities. The new early check rejects a missing manifest before training.
- PULSE-COMMAG shadow evaluation has one Python process, pandas groupby/pivot, and gzip CSV reading/writing. It does not use all logical CPUs by design. The new output identifies its phases and row counts.
- Data download uses a ThreadPoolExecutor capped at **8 workers**, not 16 CPU compute processes. The default Compose preparation setting is now 8, but this does not parallelize cross-fit training or pandas.

## Useful commands

```powershell
# See whether work is progressing, and whether Docker allocated enough CPUs:
docker stats --no-stream
docker compose -f docker-compose.publication.yml ps -a
docker compose -f docker-compose.publication.yml logs --timestamps --tail 100 pulse-commag-shadow

# Verify the new logging in a foreground run:
docker compose -f docker-compose.publication.yml up --build --exit-code-from pulse-commag-shadow pulse-commag-shadow
```

Set Docker Desktop CPU allocation to the desired number of CPUs (or check the effective WSL2 limits); a 16-logical-CPU Windows host does not necessarily mean Docker has 16 CPUs. Check `docker info` and `docker stats`. A low Task Manager reading can also mean storage/network waits, lock contention, or limited parallelism rather than a stalled program.

For the regular publication benchmark without a physically measured BWP manifest:

```powershell
$env:AGENTIC_RAN_EVAL_JOINT_SLICE_ADVANTAGE = "0"
docker compose -f docker-compose.publication.yml up --build --exit-code-from publication-test publication-test
```

If running the **optional** physical JAGC study, supply the externally measured, provenance-backed manifest at the mounted path and enable the flag. Do not use placeholder 52/4 values as proof of COMMAG hardware.

## Remaining performance engineering

The above patch is about faster failure, more download concurrency and observability. It does **not** implement safe multi-process model training or demonstrate a speed-up. The publication cross-fit loop remains serial; to parallelize it, factor each fold into independent processes, limit BLAS/OpenMP threads per process, and validate deterministic outputs against a serial reference. Parallel fold result merging must preserve exactly-once final-test routing and no leakage.
