# IEEE Access review: final Gym research protocols

## What the evidence supports

The PULSE-RAN COMMAG shadow archive produced **214,910** aligned
base-station-second recommendation rows: 76,771 training, 21,115
validation, 19,405 seen test and 97,619 unseen test. Those are descriptive
recommendations generated from historical logged COMMAG state, not executed
PULSE interventions. The archive itself confirms:
`dataset_has_actuated_pulse_ran=false`,
`ns3_was_executed=false`,
`new_colosseum_experiment_was_executed=false`, and
`reported_counterfactual_utility=false`.

The recommendation CSV includes historical observed rewards, but **none
estimates the realized reward of PULSE-selected alternative actions**.
The dataset exhibits missing three-slice alignment (roughly 15% of
unseen base-station-seconds excluded), potential scenario shift and
action-unit incompatibility. Do not infer live gains from the observed
rewards or descriptive L1 allocation distances.

The normal full COMMAG publication pipeline produces a separate
cross-fitted offline direct-method comparison and an evidence-status
check. Its readiness flags attest to procedural consistency, NOT
true counterfactual identification or proof of physical gains.

## New final-algorithm Gymnasium test

The new executable `ran-gym-final-algorithms` Docker service:
- trains on-policy PPO **using training-only toy interventions**
- freezes PPO and uses **validation-only** sparse weight-space edits
- trains offline Twin-CQL and distributional CVaR on separate randomized
  training episodes from the same action-dependent toy simulator
- compares PPO-WSP against frozen PPO, PULSE-RAN, Twin-CQL,
  distributional CVaR and the CQL/CVaR hybrid
- runs all six on **actual Gymnasium reset/step calls**, with identical
  five-action spectrum allocations, identical scenarios and held-out seeds
- checks matched reset states, deterministic action receipt, train/test
  isolation, episode-length consistency and per-policy transition rows
- reports one-sided paired seed-cluster sign-flip tests, paired bootstrap
  95% intervals, Holm adjustment, queue-threshold SLA and switch rates
- **always labels output as an interventional TOY result**; it cannot
  turn a success in this environment into IEEE Access superiority evidence.

On Windows PowerShell (Docker Desktop Linux mode):

```powershell
git pull --ff-only origin main
docker compose -f docker-compose.ran-gym.yml build ran-gym-final-test ran-gym-final-algorithms
docker compose -f docker-compose.ran-gym.yml run --rm ran-gym-final-test
docker compose -f docker-compose.ran-gym.yml run --rm ran-gym-final-algorithms
```

Primary outputs:
- `results/ran-gym-final-algorithms/gym_final_summary.json`
- `results/ran-gym-final-algorithms/gym_final_episodes.csv`
- `results/ran-gym-final-algorithms/gym_final_steps.csv.gz`
- trained `ppo_frozen.pt` and `ppo_wsp.pt`.

Run the **actual native upstream ns-O-RAN** traffic-steering environment,
without confusing it with PRB/scheduler actuation:

```powershell
$env:NSORAN_BUILD_JOBS = "4"
docker compose -f docker-compose.ran-gym.yml --profile native up --build --exit-code-from ran-gym-nsoran-smoke ran-gym-nsoran-smoke
```

This compiles the ns-O-RAN system inside Docker. It is an integration
smoke for upstream *handover* actions only. Native source compilation
is resource-heavy and is not verified by the normal CI.

## Missing for IEEE Access PRB / scheduler claims

1. **Scientific novelty.** State the precise Agentic-RAN algorithm
   contribution compared with PULSE, CQL/CVaR, constrained PPO, model
   predictive control, JAGC, safety/uncertainty filters and relevant
   published OpenRAN Gym methods. Avoid presenting an accumulation of
   separate baselines as a coherent new method without ablations.
2. **Native-compatible action semantics.** Extend ns-3/5G-LENA or
   another independently verified simulator to accept the actual
   shared BWP, per-slice integer PRBs and physical scheduler actions;
   expose the original 17 COMMAG telemetry features with verified units.
   Stock ns-O-RAN Gym handover actions **are not a drop-in substitute**.
3. **Freeze candidate and baselines.** The actual frozen publication
   policy + checksum, trained PPO, JAGC and calibrated gates must be
   frozen before the held-out test episodes. Do not train on test
   scenarios or tune after seeing final results.
4. **Experiment.** Matched train/validation/test seeds, multiple loads,
   RF strengths, fading, changing UE counts, mobility, severe URLCC
   disturbances, minimum 10 independent test seeds, multiple training
   seeds and meaningful PHY/MAC resource and ACK verification.
5. **Metrics.** Actual per-slice throughput, latency distributions,
   BLER/packet loss, SLA threshold violations, PRB feasibility,
   reward and energy in real units where available, inference and
   end-to-end control latency, safety deferrals, and switching.
6. **Inference.** Paired seed-cluster bootstrap CIs, adjusted
   permutation p-values, effect sizes, trace provenance, system
   checksums and reproducible source/experiment manifests.
7. **Honest negative results.** No publication claim of improvement
   if a baseline wins or confidence intervals overlap zero. The
   `research_publication_gate` intentionally fails without
   compatible external evidence and measured KPIs.
8. **IEEE Access manuscript.** Complete original-contribution framing,
   literature comparison, uncertainty and threats to validity,
   references and formatting; verify all generated numbers against the
   exact frozen evidence revision.

The valid paper positioning *today* is a reproducible offline
safety-governed RAN configuration study with restricted
counterfactual claims, **not** demonstrated live closed-loop RAN
superiority. A publishability assessment is not a guarantee of
acceptance.

Useful authoritative guidance:
- https://ieeeaccess.ieee.org/authors/submission-guidelines/
- https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/research-reproducibility/

Technical attribution:
- https://github.com/wineslab/ns-o-ran-gym
- https://github.com/jjalcaraz-upct/network-slicing
