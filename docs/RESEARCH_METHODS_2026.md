# Experimental action-conditional support and closed-loop planning (October 2026)

## Scope and evidence separation

The default Agentic-RAN policy and current publication tables remain unchanged.
New modules are **research candidates**, not validated replacements:

- `agentic_ran/research_support.py`: independently fit policy/OPE-role
  state/action overlap estimators; use knn, Ledoit-Wolf shrinkage Gaussian
  Mahalanobis, or two-component diagonal Gaussian-mixture nonconformity.
  Each slice/action/role gets its own train/calibration split and conformal
  tail-score, with an action accepted only if *both* roles accept it.
- `rerank(...)`: sort available actions by utility minus switching cost,
  but choose only those passing common-action constraints, local support,
  predicted SLA and uncertainty requirements. If none is eligible, emit
  `-1` (ABSTAIN). **There is no silent incumbent/fallback bypass.**
- `agentic_ran/publication_research.py`: opt-in comparison on full COMMAG
  experiment-role-swap partitions. Independent OPE evaluates selected-action
  utility on accepted cases only, plus unconditional scores under an *explicit*
  abstention opportunity cost. Disjoint validation **episodes** calibrate a
  one-sided logged-action SLA critic residual and select method/alpha.
- `agentic_ran/research_planning.py`: separate action-dependent controlled
  queue/spectrum simulator, offline learned transition model, receding-horizon
  enumeration, deliberate dynamics bias, common exogenous episode noise, and
  paired-bootstrap horizon comparisons. This is not a 3GPP-calibrated twin.

**Terminology:** The conformal diagnostic describes overlap under strong
within-group exchangeability assumptions. It is *not* behavioral propensity,
causal identification, safety certification, or a substitute for direct
controlled network intervention. The SLA margin is calibrated on observed
logged actions only; its extrapolation to alternative actions is unverified.
The toy simulator intentionally permits genuine interventions, but does not
simulate validated O-RAN PHY/MAC interactions.

## Run the local deterministic benchmark

```bash
python -m pip install -e ".[dev,oran]"
python -m pytest -q tests/test_research_methods.py
python -m agentic_ran.research_benchmark \
  --seed 2026 --episodes 30 --output results/research_methods.json
```

PowerShell (same Python commands, without Bash line continuation):

```powershell
python -m pip install -e ".[dev,oran]"
python -m pytest -q tests/test_research_methods.py
python -m agentic_ran.research_benchmark --seed 2026 --episodes 30 --output results/research_methods.json
```

Research methods are compared for alpha={0.01,0.05,0.10,0.20}.
The validation split selects a single method/alpha; seen and shifted test
splits remain untouched until final evaluation. Utility is from a *known
synthetic oracle* and abstention has a prespecified score of -0.12. A
strict 1% alpha may give very little selectivity: method ranking is specific
to the tested data, not an endorsement for network deployment.

## Full COMMAG experimental evaluation (resource-intensive)

```powershell
$env:AGENTIC_RAN_EXPERIMENTAL_SUPPORT_GUARD = "1"
$env:AGENTIC_RAN_EVAL_MATCHED_WRAPPERS = "1"
$env:AGENTIC_RAN_EVAL_CONFORMAL_RERANK = "1"
$env:LOCAL_UID = "1000"
$env:LOCAL_GID = "1000"
docker compose -f docker-compose.publication.yml up --build --force-recreate publication-test
```

Additional outputs:

- `results/publication/research_conformal_validation.csv`
- `results/publication/research_conformal_final_tests.csv`
- `publication_summary.json` key `research_conformal_rerank`

These optional figures are deliberately **not merged into** the original
publication baseline leaderboard, whose statistical target measures
counterfactual utility on all test rows. Selective-policy conditional utility
is not numerically comparable with unconditional utility unless an explicit,
justified no-actuation outcome is specified.

## October 27 evidence upgrade: accountable abstention and paired inference

The optional COMMAG comparison now exports more than accepted-only utility:

- The **fixed-penalty** sensitivity uses an explicitly assumed abstention
  score of -0.12, and remains an arbitrary decision-accounting convention.
- The **hold-incumbent direct-method** estimand estimates the incumbent's
  utility *using the independent evaluator* only where its scheduler/PRB action
  has common model support. Missing support remains missing and is excluded
  from this paired comparison rather than imputed optimistically.
- The **observed logged-incumbent** proxy uses the logged observed outcome only
  when the logged action matches the incumbent. This deliberately mixes
  observed and model-based outcomes and is *descriptive, not causal*.
- All comparisons use the **same final-test rows/experimental clusters** and
  export paired cluster-bootstrap 95% CIs, sign-permutation p-values, matched
  row coverage, and counts. Cluster units are scenario × training configuration
  × experiment; experiment-role swapping and held-out selection remain intact.
- Candidate support uses **episode-blocked fit/calibration** and episode-level
  maximum nonconformity scores. SLA residual margins are also calibrated
  from episode maxima. Low alpha is rejected when too few independent
  calibration episodes exist. This can increase abstention dramatically; no
  superior utility is promised.
- This cannot establish conformal guarantees on temporally dependent,
  nonexchangeable action-conditional distributions, causal effect estimates,
  real-time RIC performance, or certified execution safety.

New output artifacts:

\`\`\`text
results/publication/research_conformal_cluster_evidence.csv
results/publication/research_conformal_slice_scenario.csv
results/publication/research_conformal_paired_statistics.json
\`\`\`

The JSON under \`research_conformal_rerank\` now exposes the estimands, coverage,
research limitations, and explicitly marks \`publication_ready: false\`.
The legacy 14 generic readiness gates remain separate: they do not certify
novel conformal control, blocked calibration guarantees, or live actuation.

### World-model planning stress test

Use \`python -m agentic_ran.research_benchmark --oracle-gap --episodes 12\`
to compare learned vs *oracle* model-predictive control over horizons 1, 2,
and 4 on common randomized disturbance sequences. The MPC safety-margin
penalty uses held-out learned-model error rather than leaking the injected
experimental bias. This remains a deliberately simplified queue/spectrum
simulator, **not** Colosseum/SCOPE or 3GPP-calibrated control.

## What must still be done for publication

1. Replace random point-wise conformal calibration inside training roles with
   episode-blocked or time-blocked calibration before making empirical coverage
   claims on temporally dependent COMMAG traces.
2. Refit and compare critics over seeds/families; quantify *counterfactual*
   risk uncertainty and common state-conditional support. Validate no
   post-action KPI leakage.
3. Validate controller decisions in a standards-capable Colosseum/SCOPE,
   srsRAN, ns-3 or similar interactive environment. The toy demo alone is
   insufficient for a strong network-control result.
4. Report paired cluster-bootstrap effects, conditional coverage by slice
   and scenario, per-action counts, false-safe estimates, abstention cost
   sensitivity, and p50/p95/p99 decision latency on full-scale traces.

## Relevant literature (method inspiration, not novelty claims)

- Zhu et al. (2026), *Conformal Risk-Averse Decision Making with Action
  Conditional Guarantee*, https://arxiv.org/abs/2606.05551
- Deb et al. (2026), *Model Predictive Control with Differentiable World
  Models for Offline Reinforcement Learning*, https://arxiv.org/abs/2603.22430
- Zhang et al. (2026), *Hierarchical Planning with Latent World Models*,
  https://arxiv.org/abs/2604.03208
- Basu et al. (2025), *Hybrid Adaptive Conformal Offline Reinforcement
  Learning*, https://arxiv.org/abs/2509.09772
- Wu et al. (2022), *Supported Policy Optimization for Offline RL*,
  https://arxiv.org/abs/2202.06239

No claim of exact reproduction, uniqueness, or guarantees from these papers
is made. This module combines standard ideas in a tested Agentic-RAN context.
