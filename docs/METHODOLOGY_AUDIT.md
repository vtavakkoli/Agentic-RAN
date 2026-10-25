# Methodology audit: action support, switching and ablations

This document records changes required before the Agentic-RAN manuscript's
numerical claims are updated or the software is described as a deployed
agentic RAN controller.

## What has been corrected

### 1. Two different switching metrics

- `policy_churn_rate` is now the rate at which a method's **selected**
  scheduler–PRB action changes between consecutive time-ordered rows within
  the same UE episode. The denominator is the count of eligible adjacent
  transitions, not the number of test rows.
- `logged_action_disagreement_rate` is the share of selected actions different
  from the original logged scheduler–PRB configuration. This is what earlier
  versions incorrectly called "churn".
- The new switching metric is an **open-loop replay diagnostic** on logged
  states. Since the next state is not generated from the proposed action, it
  does **not** measure real reconfigurations, stability after interventions,
  or closed-loop switching overhead.
- The previous paper's published 35–46 percentage-point churn result must
  **not** be reproduced as an observed network switching reduction. It referred
  to disagreement with the logged configuration, not sequential switching.
- The logged policy's adjacent switching is also measured in the revised
  results, with the two metrics kept separate.

### 2. State-conditional action overlap

The initial common-support gate tests whether a scheduler–PRB action has been
logged for the given slice in **both** experimental roles. This is a useful
minimum check but insufficient to establish action support near any chosen
state.

The additional `conditional_support.csv` file reports a *sampled* nearest-
neighbor overlap proxy for selected state/action pairs. It uses a fixed set of
context KPI features, excluding the current scheduler/PRB to avoid a trivial
diagnostic. It separately tests overlap with the policy-fit and evaluator-fit
training samples and reports their conjunction.

Thresholds are fitted within training roles only and do not use final-test
rows. Reference/query downsampling bounds runtime. This analysis is **not**
a causal support estimate: even the retained KPI features can already reflect
earlier configuration decisions. A low overlap score is adverse evidence; a
high overlap score does not prove ignorability, sequential positivity or
identified intervention effects. Do not claim that this diagnostic solves
counterfactual identifiability.

### 3. Fairer offline-RL value backups

FQI and linear CQL now mask unsupported slice-specific actions in next-state
Bellman backups, matching the common action set allowed at decision time.
CQL's conservative log-sum-exp penalty also uses the same slice support.
Any future new baseline must use the same action support and be tested on the
same final partitions.

This does **not** give FQI/CQL the full controller's safety/hysteresis policy.
A complete matched-wrapper comparison is still an important future experiment.

### 4. Paired controller ablations

`publication_ablations.csv` reports, for each cross-fit fold and held-out
split, the full policy plus:
- without SLA/uncertainty safety gates;
- without the OOD fallback gate;
- without the uncertainty term in immediate scoring;
- without the hysteresis comparison.

All variants use the same independent outcome-model predictions and final
held-out rows. Ablation utility is a model-based direct-method estimate;
it is not a new observed RAN outcome. Differences need uncertainty analysis
before publication claims.

### 5. Validation feasibility

The hyperparameter search records `constraint_feasible`. If no feasible
candidate exists, it still exports the best diagnostic candidate rather than
silently aborting, **but evidence readiness now fails**. The manuscript must
not treat an infeasible selection as meeting its SLA constraint.

## Reproduction

From the repository root:

```bash
python -m pip install -e ".[dev,oran]"
pytest -q tests/test_publication_methodology.py tests/test_publication_science_integration.py
LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) \
  docker compose -f docker-compose.publication.yml up --build --force-recreate publication-test
```

The publication benchmark needs the pinned upstream COMMAG traces and is
significantly more expensive than the synthetic integration tests.
Inspect:

```text
results/publication/publication_baselines.csv
results/publication/publication_ablations.csv
results/publication/conditional_support.csv
results/publication/tradeoff_summary.json
results/publication/publication_summary.json
results/publication/report.html
```

**Do not reuse Table III–V figures from the pre-audit PDF as if these were the
post-audit results.** The updated baselines, switching metrics, significance
tests, and abstract must be regenerated and reconciled with the exported data.

## Experiments still needed for a high-confidence journal claim

1. **Closed-loop intervention validation.** Replay decisions in a controllable
   Colosseum/SCOPE, srsRAN or equivalent environment that actually executes
   scheduler/PRB changes. Log post-action observations and compare predicted
   and realized SLA, energy and throughput. An adapter stub is not evidence.
2. **Strong identification design.** Log action propensities or use randomized
   safely bounded interventions; compare with a doubly robust/importance-
   weighted estimator when its assumptions are credible. Conditioning and
   non-confounding assumptions must be explicit.
3. **Matched safety/persistence wrappers.** Equip FQI and CQL with the same
   hysteresis, switching budget, fallback, and action constraints as the
   proposed selector to isolate policy-model improvements.
4. **Evaluator-family/seed uncertainty.** Independently refit multiple outcome
   models and bootstrap at the experiment-cluster level, including uncertainty
   in the evaluator, rather than solely bootstrapping fixed prediction arrays.
5. **Action-dependent feature audit.** Document which KPIs are available at
   decision time and which are consequences of the previous configuration.
   A state-conditional action model is not automatically a causal model.

The present work remains an *offline selection and evaluation benchmark*,
not a demonstrated near-real-time or causally validated agentic RAN controller.
