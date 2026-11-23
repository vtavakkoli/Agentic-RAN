# IEEE Access research-scale Agentic-RAN Gym campaign

**Scope:** reproducible engineering and *interventional toy RAN* research. Not native 5G-LENA,
Colosseum, srsRAN or physical PRB/scheduler validation. This produces manuscript **candidate**
tables and figures with explicit claims restrictions; an IEEE Access acceptance or a
physical-network superiority claim cannot be produced merely by longer runs.

## Windows Docker: one command for the complete compatible-Gym study

From the latest Agentic-RAN main branch in PowerShell with Docker Desktop
configured for Linux containers:

```powershell
git switch main
git pull --ff-only origin main
powershell -ExecutionPolicy Bypass -File .\scripts\run_ieee_access_gym_campaign.ps1 -FullOffline
```

This executes the unit suite, multi-training-seed campaign smoke test, final
algorithm tests, **10 independently trained PPO and offline critic runs**,
validation-only weight-space editing, **40 held-out test seeds per training
run** across four scenarios, full independent offline COMMAG publication,
and PULSE shadow evaluation. Additional native simulator compilation is
opt-in and a separate experiment:

```powershell
$env:NSORAN_BUILD_JOBS="4"
powershell -ExecutionPolicy Bypass -File .\scripts\run_ieee_access_gym_campaign.ps1 -FullOffline -NativeSmoke
```

**Native caveat:** Upstream ns-O-RAN Gym's standard TrafficSteeringEnv applies
UE handover decisions. It **does not** apply the COMMAG-trained Agentic-RAN
scheduler/PRB decisions. This native smoke is a build+step integration check,
not publication evidence of PRB optimization; native source compilation
has not been verified on every Windows host.

For an already validated external **PRB/scheduler-capable** Gym interface,
freeze a trusted `publication_models.joblib` model checksum, provide the
real telemetry/action adapter `external/site-adapters/site_adapter.py`,
configure `configs/ran_gym_active.json`, and run the separate strict receipt,
throughput, latency and SLA gates:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_ieee_access_gym_campaign.ps1 -FullOffline -ExternalPhysical
```

The active path fails closed if the adapter/config is missing, the actual
Gym step does not acknowledge the requested action, its service KPIs are
missing, or evidence fails the final publication gate. Never use a toy
receipt as proof of actual NR scheduler actuation.

## Docker-only commands, without invoking the PowerShell wrapper

```powershell
docker compose -f docker-compose.ran-gym.yml build ran-gym-campaign
docker compose -f docker-compose.ran-gym.yml run --rm ran-gym-campaign-test
docker compose -f docker-compose.ran-gym.yml run --rm ran-gym-campaign
```

The campaign refuses to overwrite an existing output folder. Set a new
`$env:RAN_GYM_CAMPAIGN_NAME="ran-gym-campaign-2026-XX"` for each experiment;
this preserves reproducibility and prevents accidental test-result replacement.

To scale the experiment further:

```powershell
$env:RAN_GYM_CAMPAIGN_NAME = "ran-gym-10training-60test"
$env:RAN_GYM_TRAINING_RUNS = "10"
$env:RAN_GYM_TEST_SEEDS = "60"
$env:RAN_GYM_PPO_UPDATES = "64"
$env:RAN_GYM_LOGGED_EPISODES = "160"
$env:RAN_GYM_VALIDATION_EPISODES = "32"
$env:RAN_GYM_STEPS = "40"
docker compose -f docker-compose.ran-gym.yml run --build --rm ran-gym-campaign
```

## Frozen experimental protocol

- Environments: four toy scenarios `normal`, `congested`, `bursty`,
  `weak_channel`; identical five-action eMBB/mMTC/URLLC spectrum fractions.
- Six controller families: frozen PPO, validation-edited PPO-WSP, PULSE-RAN,
  Twin-CQL, distributional CVaR, Twin-CQL/CVaR hybrid.
- Independent full training seeds, random-action logged training episodes for
  offline critics, stochastic on-policy training for PPO; validation-only
  weight edits. Weak-channel shift is absent from WSP validation.
- Identical held-out environment seeds for all policy rollouts **within each
  training run**, different held-out seeds between independent training runs.
- No action-log counterfactual substitution: each reward follows a Gymnasium
  `step` on the actual toy simulator. Confirms paired initial state hashes,
  all actions recognized by the same Gym action space, and toy action receipts.
- Train-run-level bootstrap 95% confidence intervals and one-sided sign-flip
  randomization; Holm correction for five predeclared comparisons. No
  pseudoreplication of steps or scenarios. Reports effect sizes, scenario-wise
  differences, queue-SLA proxies and switching.
- The protocol JSON is written before the first training run. Evidence exports
  include parameter settings, trained PPO/WSP state dictionaries, file SHA-256,
  all held-out episode/step traces, and self-contained HTML with SVG figures.

## Files

Within `results/ran-gym-campaign-<timestamp>/` (default runner):

- `protocol_frozen.json`: precommitted hyperparameters and train/eval split
- `training_run_00/` ... `training_run_09/`: frozen/edited PPO weights,
  per-policy paired results, original executed step/episode traces
- `campaign_summary.json`: run-level paired inference and claim boundaries
- `training_run_means.csv`, `scenario_means.csv`
- `all_heldout_episodes.csv.gz`, `all_heldout_steps.csv.gz`
- `research_report.html`: offline manuscript candidate figures and tables
- `artifact_sha256.json`: SHA-256 for primary aggregate artifacts

These are artifacts suitable for **further manuscript preparation**, not a
certification of external scientific validity.

## Mandatory evidence gaps before strong IEEE Access claims

1. A simulator implementing the *same* COMMAG scheduler/PRB actions and 17
   feature semantics with validated bandwidth-part resource accounting.
2. Real executed control receipts, independently measured per-slice throughput,
   packet latency distribution, BLER and actual SLA violations.
3. Paired baseline performance against strong current same-action-space RAN
   agents (PPO, CQL, CVaR, joint allocation, risk-gated variants).
4. External-domain shift, realistic traffic/UE loads and independent
   simulator or testbed replication; demonstrate real-time RIC latency.
5. Clear novel method and reviewer-grade ablations isolating support gating,
   uncertainty, action projection, WSP, planning and hysteresis.
6. Independent verification of all study results, data and dependency
   provenance; disclosures, code release, full literature review and IEEE
   Access formatting. No significance guarantee or invented gains.

Read also:
- `docs/IEEE_ACCESS_FINAL_ALGORITHMS_AUDIT.md`
- `docs/PHYSICAL_CLOSED_LOOP_RESEARCH.md`
- `docs/PUBLICATION_CHECKLIST.md`
- `docs/RAN_GYM_ACTIVE_PUBLICATION.md`

**Responsible submission framing:** This can support a rigorously qualified
offline decision research manuscript and an interventional toy study. It
cannot by itself substantiate superior closed-loop NR scheduler performance.
