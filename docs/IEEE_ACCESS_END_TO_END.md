# IEEE Access: reproducible Agentic-RAN study (train → validate → test → report)

> **Evidence-first research workflow, not an IEEE acceptance guarantee.**
> The native ns-3.49 backend currently controls UDP packet counts. The
> COMMAG controller acts on scheduler/PRB allocations and is **not mapped**
> to this UDP example. The workflow is designed to expose that limitation,
> not to hide it or manufacture 5G NR performance.

## Prerequisites

- `Agentic-RAN` working copy; Docker Engine/Desktop with Linux containers;
  PowerShell 5.1+ (`powershell.exe` on Windows or `pwsh` elsewhere).
- Internet access and disk/memory for PyTorch, the pinned ns-3.49 C++23 build,
  and the full public COMMAG data. Review source license/data rights before downloading.
- For private/corporate CA/proxy setups, preserve certificates in **local**
  Docker stages; never commit private CA keys or internal endpoints.
- The code and data sources **must be frozen before test**. Use clean Git
  revisions and keep the run manifest and output hashes with any manuscript.
- On Linux, set `LOCAL_UID=$(id -u)` and `LOCAL_GID=$(id -g)` if your Docker
  bind-mounted research output permissions differ; Compose defaults to UID/GID 1000.

## One command: complete research workflow (Windows PowerShell)

```powershell
cd C:\development\research\Agentic-RAN
# Make sure the repository contains the new script and report module.
git pull --ff-only origin main

powershell -ExecutionPolicy Bypass -File .\scripts\run_ieee_access_complete.ps1 `
  -Preset research -TrainingRuns 10 -TestSeeds 40 -PPOUpdates 48 `
  -LoggedEpisodes 120 -ValidationEpisodes 24 -Steps 32 -BuildJobs 4
```

**Note:** This is a resource-intensive, real execution workflow. It downloads
public COMMAG traces, fits independent offline estimators, trains learned toy
controllers (including PPO), runs held-out paired tests, compiles actual ns-3
C++, and creates a structured **final evidence report**. The code does not
promise that models improve reward, or that the final manuscript meets the
journal's acceptance criteria.

**Fast check (not for paper claims):**

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_ieee_access_complete.ps1 `
  -Preset smoke -SkipOffline -SkipNative
```

**What runs, in strict dependency order:**

1. Verify Docker and Compose configuration; record Git SHA and a unique run ID.
2. Build CPU/PyTorch study images and run standard research regression tests.
3. Train six policies on controlled action-dependent Gymnasium toy environments;
   select hyperparameters on validation only; evaluate separately on independent
   held-out test seeds/scenarios; store action/episode traces and signed baseline
   differences with corrected statistics.
4. Download/prepare full COMMAG records, train offline policy and independent
   evaluator on separate roles, execute seen/unseen held-out checks, and store
   estimates and observed outcomes with provenance. Execute PULSE logged shadow.
5. Build the pinned native ns-3.49 + Python 3.13 image and execute repeated
   UDP simulator episodes; verify realized action receipts and network KPIs.
6. Generate final machine-readable research gates, HTML and Markdown report,
   two CSV manuscript tables, SVG figure, and SHA-256 manifest.

The script uses **new per-run output paths**, so one run cannot silently
reuse the previous run's model report or native evidence. It fails on a broken
Docker command but still attempts to generate an **incomplete-evidence report**.

## Output structure

```
results/ieee-access-YYYYMMDD-HHmmss-xxxxxx/
├── run_manifest.json                         # Git SHA, parameters, stage outcomes
├── campaign/
│   ├── protocol_frozen.json                  # preregistered training/held-out split
│   ├── campaign_summary.json
│   ├── training_run_means.csv
│   ├── scenario_means.csv
│   ├── all_heldout_episodes.csv.gz
│   ├── all_heldout_steps.csv.gz
│   ├── artifact_sha256.json                  # original campaign file hashes
│   ├── research_report.html
│   └── training_run_*/                       # frozen PPO and WSP checkpoints
├── offline/
│   ├── publication_summary.json
│   ├── publication_models.joblib
│   ├── publication_baselines.csv
│   ├── publication_decisions.csv.gz
│   └── ...                                   # full COMMAG OPE and ablation evidence
├── pulse/
│   ├── pulse_commag_shadow_summary.json
│   └── pulse_commag_shadow_recommendations.csv.gz
├── native/
│   └── real_closed_loop.json                 # real ns-3 UDP packet observations
├── physical/                                # only with validated -ExternalPhysical
│   ├── ran_gym_summary.json
│   └── final_evidence_gate.json
└── final-report/
    ├── ieee_access_report.html              # attractive shareable research report
    ├── ieee_access_report.md                # manuscript-oriented methods and caveats
    ├── ieee_access_evidence.json            # machine-readable gating and status
    ├── table_toy_comparisons.csv            # train-run cluster CI, Holm-corrected p
    ├── table_native_udp_episodes.csv        # real native network counts/KPIs
    ├── fig_toy_reward_deltas.svg            # signed toy-only effects
    └── sha256_manifest.json                 # tamper-evident output checksums
```

Read `final-report/ieee_access_report.html` first, then audit the JSON,
raw traces, signed tables and source manifest before selecting claims.
A generated `READY-FOR-MANUSCRIPT` value in the *offline COMMAG subsystem*
**does not** mean that physical scheduler results or the whole paper are ready.

## Test and report commands without the PowerShell wrapper

First build and run the functional toy suite:

```bash
mkdir -p results artifacts data/raw/commag

docker compose -f docker-compose.ran-gym.yml build \
  ran-gym-unit ran-gym-campaign-test ran-gym-final-test ran-gym-campaign

docker compose -f docker-compose.ran-gym.yml run --rm ran-gym-unit
docker compose -f docker-compose.ran-gym.yml run --rm ran-gym-campaign-test
docker compose -f docker-compose.ran-gym.yml run --rm ran-gym-final-test
```

An **independent research run** (choose a new unique name):

```bash
export RUN_ID="ieee-access-manual-001"
export RAN_GYM_CAMPAIGN_NAME="$RUN_ID/campaign"
export RAN_GYM_TRAINING_RUNS=10
export RAN_GYM_TEST_SEEDS=40
export RAN_GYM_PPO_UPDATES=48
export RAN_GYM_LOGGED_EPISODES=120
export RAN_GYM_VALIDATION_EPISODES=24
export RAN_GYM_STEPS=32
export RAN_GYM_PPO_EPISODES_PER_UPDATE=12
export RAN_GYM_WSP_ROUNDS=3
export RAN_GYM_CRITIC_ITERATIONS=5

docker compose -f docker-compose.ran-gym.yml run --rm ran-gym-campaign
```

The **full offline** evaluation can be isolated to the same run via:

```bash
export AGENTIC_RAN_PUBLICATION_OUTPUT="/workspace/results/$RUN_ID/offline"
export AGENTIC_RAN_PULSE_OUTPUT="/workspace/results/$RUN_ID/pulse"
docker compose -f docker-compose.publication.yml up --build \
  --force-recreate --exit-code-from publication-test publication-test
docker compose -f docker-compose.publication.yml run --rm --no-deps pulse-commag-shadow
```

The separate **real native ns-3 UDP** demonstration:

```bash
export MODERN_NS3_BUILD_JOBS=4
export AGENTIC_RAN_NATIVE_REPORT_PATH="/workspace/results/$RUN_ID/native/real_closed_loop.json"
mkdir -p "results/$RUN_ID/native"
docker compose -f docker-compose.ran-gym.yml --profile native build ran-gym-modern-ns3-smoke
docker compose -f docker-compose.ran-gym.yml --profile native run --rm ran-gym-modern-ns3-smoke
```

For an automatically generated combined report, prefer the PowerShell wrapper,
which creates the matching `run_manifest.json`. Alternatively create a manifest
of your actual stages and run:

```bash
docker compose -f docker-compose.ran-gym.yml build ran-gym-unit
docker compose -f docker-compose.ran-gym.yml run --rm --entrypoint python \
  ran-gym-unit -m agentic_ran.research_ieee_access_report \
  --run-root "/workspace/results/$RUN_ID" \
  --manifest "/workspace/results/$RUN_ID/run_manifest.json" \
  --output "/workspace/results/$RUN_ID/final-report"
```

The audit produces output even when required evidence is missing; add
`--require-complete` to exit non-zero on missing stages/invalid evidence.

## Scientific acceptance matrix for an IEEE Access research article

| Evidence source | Valid scientific use | Forbidden inference |
|---|---|---|
| COMMAG logged telemetry | measured KPI distributions, model-fit metrics and honest OPE estimates | causal effects of unexecuted alternative scheduling actions |
| Action-dependent toy Gym | controlled reward and SLA *proxy* comparisons among 6 trained policies; multi-seed paired statistics | physical NR PHY, BLER, PRB scheduling or measured end-user latency |
| Compiled ns-3.49 UDP | verified C++/Python interoperability, delivery counters, action receipts and deterministic resets | COMMAG policy applied to true RAN PRB/scheduler controls |
| Real NR/5G-LENA/Colosseum backend (user-supplied) | physical scheduler actuation only **after** authentic state/action/PHY mapping, receipts and matched trials pass | automatic real-network readiness, latency safety or production certification |

**Required next research work for a strong novelty and performance submission:**

1. Implement a **compatible NR scenario** mapping eMBB/mMTC/URLLC slice
   actions to actual BWP-constrained PRB/scheduler operations. Do **not**
   reinterpret UDP action counts as PRBs.
2. Add competing baselines on a common action space and equal compute budget:
   frozen original, FQI, calibrated greedy/behavior, risk-constrained and
   ablations removing OOD gates, support gating, world model, hysteresis,
   uncertainty, and WSP changes.
3. Freeze algorithm selection before test; compare multiple independent
   training seeds and held-out cell/traffic/channel scenarios, preserving
   matched trial seeds. State the independent statistical unit.
4. Report **effect sizes** and distribution (not just a p-value), clustered
   bootstrap CIs, Holm-adjusted paired tests, violation rates, fallback and
   rollback, compute cost, and negative and shifted-domain outcomes.
5. Measure native per-slice throughput, delay percentiles, BLER/drop/queue,
   action latency, PRB budget, fairness, energy (only if actually measured),
   and safety constraints using genuine traffic/channel traces.
6. Do not relabel direct-method OPE predictions as realized intervention
   rewards, nor toy SLA proxies as real 5QI SLA measurements.

## Suggested manuscript outline

A suitable *research question* is whether safety/uncertainty/support gating
improves the reliability of agentic RAN decisions under distribution shift
*without degrading throughput or service-level constraints*. The current data
supports **experimental methodology and software validation**, not yet a
universal improvement theorem or proven NR throughput gain.

Suggested structure: (1) Introduction & gap, (2) Related work and exact
novelty, (3) Architecture & fail-closed safety gates, (4) Mathematical problem
and offline support/positivity, (5) Dataset and train/validation/test split,
(6) Controlled experiments + predeclared baselines and ablations, (7) Real
NR simulator validation **once implemented**, (8) Statistical analysis and
negative cases, (9) Threats to validity, (10) Conclusions and reproducibility.

IEEE Access requires its own **double-column, single-spaced** template and
submission of matching editable (Word/LaTeX) and PDF manuscripts. These
scripts generate a source-grounded scientific *report*, **not a submission-ready
manuscript** or a guarantee of acceptance.

Official guidance:
- https://ieeeaccess.ieee.org/authors/submission-guidelines/
- https://ieeeaccess.ieee.org/authors/preparing-your-article/
- https://ieeeaccess.ieee.org/authors/reproducibility/