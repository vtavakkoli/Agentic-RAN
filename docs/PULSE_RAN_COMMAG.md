# PULSE-RAN on real Colosseum COMMAG data

## Why the previous command was wrong

The older research_twin_study CLI uses a **constructed synthetic RAN
simulator**. It does not load COMMAG, run ns-3, or start Colosseum.

The COMMAG traces originate from the **Colosseum wireless network emulator**,
which is a separate platform from ns-3. Processing recordings is not
executing a new controlled Colosseum experiment.

## A. Run dataset-backed PULSE-RAN shadow recommendations (PowerShell)

From the Agentic-RAN project root with Docker Desktop running:

~~~powershell
git pull --ff-only

docker compose -f docker-compose.publication.yml up --build --force-recreate --exit-code-from pulse-commag-shadow pulse-commag-shadow
~~~

This downloads/prepares the commit-pinned full COMMAG traces and analyzes
recorded three-slice buffer telemetry. The prepared dataset is saved at:

data/prepared/commag-publication/commag_publication_transitions.csv.gz

PULSE outputs:

- results/pulse-commag/pulse_commag_shadow_summary.json
- results/pulse-commag/pulse_commag_shadow_recommendations.csv.gz

If preparation has already completed, reuse the cached real dataset:

~~~powershell
docker compose -f docker-compose.publication.yml build pulse-commag-shadow
docker compose -f docker-compose.publication.yml run --rm --no-deps pulse-commag-shadow
~~~

## B. Evaluate existing RL and Agentic-RAN controllers on real COMMAG

~~~powershell
$env:AGENTIC_RAN_EXPERIMENTAL_SUPPORT_GUARD = "1"
$env:AGENTIC_RAN_EVAL_MATCHED_WRAPPERS = "1"
$env:AGENTIC_RAN_EVAL_CONFORMAL_RERANK = "1"
$env:AGENTIC_RAN_EVAL_BEHAVIOR_VARIANTS = "1"
$env:AGENTIC_RAN_EVAL_SAFE_IMPROVEMENT = "1"
$env:AGENTIC_RAN_EVAL_ADAPTIVE_HYBRID = "1"

docker compose -f docker-compose.publication.yml up --build --force-recreate --exit-code-from publication-test publication-test
~~~

This existing experiment evaluates FQI, CQL, Agentic-RAN and enabled
research controllers on the **same recorded data**. It does NOT yet run
PULSE-RAN as an actuated policy or include PULSE in causal-utility
rankings. COMMAG records per-slice scheduler/RBG action labels, whereas
PULSE chooses a joint three-slice target allocation mode. These action
spaces must be matched and tested in an interactive environment before
claiming comparative utility.

### What the new COMMAG PULSE shadow analysis does

- Groups measured downlink buffer bytes at matching scenario, training
  configuration, experiment, base station, timestamp and slice.
- Keeps only complete and aligned eMBB/mMTC/URLLC observations and reports
  missing alignment coverage; never fabricates omitted slices.
- Calibrates bytes-per-queue-unit from **training split only** (the
  default is the training-buffer 90th percentile). Original PULSE
  thresholds URLLC > 0.45 and eMBB > 1.0 are then applied to these
  dimensionless normalized queues. This adaptation is clearly recorded
  and is not a physically calibrated equivalence to the toy model.
- Emits proposed target shares, the recorded allocation shares and
  descriptive share-distance diagnostics.
- Preserves recorded COMMAG reward as *logged-action reward only*.
  It never attributes observed rewards to PULSE counterfactual actions.

This is a useful dataset-backed **shadow recommendation audit**, not a
closed-loop intervention or a validated causal superiority experiment.

## C. ns-3 is not included in this Docker command

No executable ns-3/5G-LENA closed-loop scenario is packaged with this
repo. A separate installed/configured ns-3 or 5G-LENA simulator must
accept PULSE actions, update network state, and return measured post-action
KPI/reward trajectories. The existing NS3WorldModelAdapter is an HTTP
prediction adapter, **not** a deployed ns-3 simulation.

## D. New Colosseum execution is not included either

The dataset was collected on Colosseum, but a fresh Colosseum experiment
requires external testbed access and a working xApp/DU actuation bridge.
Replay of CSV traces cannot establish effects of changing a scheduler or
a slice allocation. A properly controlled trial should use identical
scenarios, matched actions, safety gates, independent radio seeds, and
realized SLA/throughput metrics.

Citation: L. Bonati et al., Intelligence and Learning in O-RAN for
Data-driven NextG Cellular Networks, IEEE Communications Magazine,
59(10), 21-27, 2021. Dataset license GPL-3.0; not redistributed here.

Upstream: https://github.com/wineslab/colosseum-oran-commag-dataset
