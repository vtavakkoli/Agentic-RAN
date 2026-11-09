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

## Feasibility and risk calibration correction (October 28, 2026)

The first episode-blocked COMMAG publication rerun produced **zero accepted
actions** on 1,067,734 held-out decisions. This is a scientific failure, not
a safety success. Two simultaneous causes were documented: 95%-quantile
**pointwise episode maxima** in the logged SLA residual (+0.567/+0.613)
already exceeded the 0.50 eligibility threshold, and `alpha=0.01` required
99 independent support-calibration episodes per slice/action/role while only
about 6–16 were present.

The optional research branch now:

1. Calibrates a 95% quantile of **episode-MEAN** logged-action SLA residual
   bias, keeps worst-step calibration **as a diagnostic**, and explicitly
   states that mean-risk screening is *not* per-step safety or a bound for
   alternative interventions.
2. Searches `alpha={0.10,0.15,0.20,0.25,0.30,0.35}`, retaining the
   per-action episode-resolution gate, instead of exclusively using
   scientifically unresolvable thresholds.
3. Exports four rejection-pipeline coverages (slice/OOD; predicted SLA and
   uncertainty; conditional support; joint eligible) to reveal where
   abstention occurs.
4. Requires at least 5% validation actuation coverage for a candidate to
   count as *validation-feasible*. When the entire grid is degenerate, the
   output explicitly records failure. It does not call an all-abstaining
   candidate a successful selected model.
5. Preserves the independent OPE role, episode-blocked role separation,
   fail-closed no-actuation decision, matched baselines, and statistical
   sensitivity analyses.

These modifications are a **new scientific hypothesis**, not a successful
COMMAG performance result. Lower observed risk margins arise from targeting
*episode-average* residuals instead of worst individual steps; they are not
a relaxation of a certified instantaneous-SLA guarantee (no such guarantee
was ever demonstrated). If an actual per-step SLA guarantee is required,
the previous zero-coverage controller is evidence that this dataset and
model cannot establish one under the old threshold.

Do not claim that conformal support p-values are causal propensity estimates,
or that action safety is validated by the logged-action residual margin.
Full-scale tests must establish risk, utility, coverage, latency, cluster CIs
and deployment applicability before selecting a paper headline.

## October 29: Boltzmann and behavior-guided offline controller study

An optional independent study compares four controllers against the existing
conformal re-ranker **under exactly the same supported action candidates and
logged-action episode-mean SLA, uncertainty and OOD gates**:

| Experimental controller | Decision rule |
| --- | --- |
| Boltzmann / softmax | Sample a supported action from a seeded, per-row reproducible Gibbs distribution over predicted immediate utility |
| Behavior cloning | Choose highest learned logged-behavior probability among eligible supported actions |
| Behavior-regularized Gibbs MAP | Choose highest utility *and* behavior likelihood using log p_behavior + utility / temperature, motivated by KL-constrained offline policy improvement |
| Behavior-tree safety supervisor | First prefer a fully eligible incumbent within a tunable utility margin; else choose the best supported and sufficiently likely action; else abstain |

The behavior model is a shallow, bounded-data HistGradientBoostingClassifier,
trained **only on the policy-fit experiment role** using six decision-context
features and slice type (never post-action targets or independent evaluator
outcomes). Missing-action probabilities are zero. Temperature, behavior prior
weight, minimum behavior likelihood, and hysteresis margins are selected using
**seen validation only**. The full seen and unseen test sets remain untouched
until after selection. Boltzmann policy sampling uses a fixed seed and
pre-decision row identifiers, and does not depend on batch size.

This work is **not** a reproduction of the cited papers, and a Gibbs policy
built from entropy regularization is not itself novel.

Run it in PowerShell:

\`\`\`powershell
cd Agentic-RAN
git pull --ff-only
$env:AGENTIC_RAN_EVAL_BEHAVIOR_VARIANTS = "1"
$env:AGENTIC_RAN_EVAL_MATCHED_WRAPPERS = "1"
$env:AGENTIC_RAN_EXPERIMENTAL_SUPPORT_GUARD = "1"
$env:LOCAL_UID = "1000"
$env:LOCAL_GID = "1000"
docker compose -f docker-compose.publication.yml up --build --force-recreate publication-test
\`\`\`

Setting \`AGENTIC_RAN_EVAL_BEHAVIOR_VARIANTS=1\` implicitly enables the
conformal experimental comparison as the shared candidate mask.

Additional data files:
- \`research_behavior_validation.csv\` (validation search, method-specific parameters)
- \`research_behavior_final_tests.csv\` (each method on held-out seen/unseen splits)
- \`research_behavior_cluster_evidence.csv\` (per-cluster matched outcomes)
- \`research_behavior_slice_scenario.csv\` (scenario/slice actuation coverage)
- \`research_behavior_paired_statistics.json\` (cluster bootstrap and paired tests)

The \`research_behavior_variants\` summary group is kept separate from the
legacy model leaderboard because the selective policies have abstentions and
therefore **conditional utility is not directly comparable** with unconditional
baseline utility. Fixed abstention penalty -0.12 is only a sensitivity
scenario, NOT a known RAN cost. Behavior-prior probabilities are not causal
behavior propensities when actions were configured statically. Boltzmann
stochastic selection is not recommended for production without interaction
studies. No online network performance or certified SLA safety is claimed.

Literature inspirations:
- Lin et al. (August 2026), *Decoupling Policy Extraction for Offline
  Reinforcement Learning*, https://arxiv.org/abs/2608.20909
- Cao et al. (July 2026), *Entropy Regularization Improves Policy Robustness
  in Continuous-Time Reinforcement Learning*, https://arxiv.org/abs/2607.03168
- Dong et al. (February 2026), *Proximal Action Replacement for Behavior
  Cloning Actor-Critic*, https://arxiv.org/abs/2602.07441
- Mao et al. (November 2025), *Adaptive Neighborhood-Constrained Q
  Learning*, https://arxiv.org/abs/2511.02567

## November 4: incumbent-relative control, constrained selection and Jev interface

The experimental research switch `AGENTIC_RAN_EVAL_SAFE_IMPROVEMENT=1`
adds **four optional offline policies** while retaining the original benchmark
unchanged when the flag is unset. Each uses the same policy-side candidate
scores and strict conformal/slice/OOD/predicted-SLA/uncertainty eligibility
gates. Independent OPE predictions are used *after* actions are chosen.

| Research mode | Proposal |
| --- | --- |
| `advantage_lcb` | Select switch only if predicted advantage over a *jointly eligible incumbent* exceeds an uncertainty-proxy penalty and switching cost |
| `constraint_conditioned` | Rank eligible actions by utility minus configurable normalized SLA and uncertainty burdens |
| `behavior_anchored_lcb` | Relative advantage with a policy-role behavior-cloning log-likelihood anchor |
| `verified_hold` | Hold a *jointly eligible* incumbent unless a high-gain switch passes a stronger advantage threshold |

These are **DPRL/C2IQL-inspired experimental baselines, not faithful
reproductions of DPRL, C2IQL or OCPO-SA**. The per-action critic uncertainty
is not a calibrated confidence interval on action-pair advantage; therefore
**LCB here means a conservative sensitivity heuristic, not an admissible
formal safe-improvement guarantee**. Safety, OOD and common action masks
may still reject many eMBB decisions. Never relax them silently to produce a
superior score.

A `jev_guided` selector and `validated_jev_hints` interface also exist.
They permit externally recorded, allow-listed numeric Jev recommendations
within [-0.25, 0.25]. **No Jev model or API is invoked**; Jev is
*intentionally excluded* from actual model comparisons until inference
traces, prompts, versions, energy and latency are recorded and replayed.
The interface cannot authorize an otherwise unsupported action.

Evaluation: tune each method on the seen-validation fold only; freeze
parameters and the validation-selected family before seen/unseen final
tests; report full-row coverage, accepted-only independent OPE utility,
fixed −0.12 abstention-penalty sensitivity, strictly supported hold-proxy,
within-cluster paired tests, and actual selected incumbent-hold frequency.

Run the full dataset evaluation (expect more model predictions/time than the
original four-variant sweep):

```powershell
cd Agentic-RAN
git pull --ff-only
$env:AGENTIC_RAN_EVAL_SAFE_IMPROVEMENT = "1"
$env:AGENTIC_RAN_EVAL_CONFORMAL_RERANK = "1"
$env:AGENTIC_RAN_EVAL_MATCHED_WRAPPERS = "1"
$env:AGENTIC_RAN_EXPERIMENTAL_SUPPORT_GUARD = "1"
$env:LOCAL_UID = "1000"
$env:LOCAL_GID = "1000"
docker compose -f docker-compose.publication.yml up --build --force-recreate --exit-code-from publication-test publication-test
```

The Compose file forwards this flag into the container. Behavior variants can
also be enabled with `AGENTIC_RAN_EVAL_BEHAVIOR_VARIANTS=1` if desired; the
new switch enables conformal shared gating independently of that choice.

Additional output files:
- `research_safe_improvement_validation.csv`
- `research_safe_improvement_final_tests.csv`
- `research_safe_improvement_cluster_evidence.csv`
- `research_safe_improvement_slice_scenario.csv`
- `research_safe_improvement_paired_statistics.json`
- `research_action_transition_audit.csv`

The action-transition audit compares action labels on adjacent timestamps
within the same episode and slice, records timestamp ties, and reports
configuration diversity *across* episodes separately. Static run-to-run
variety must **not** be mistaken for intervention outcomes.

An independent sanity check uses a **synthetic generative action-response
world**, not COMMAG. It exposes both an assumed −0.12 abstention cost and a
known-incumbent potential outcome. Run:

```bash
python -m agentic_ran.research_safe_benchmark --seed 2026 --episodes 60 --output results/research_safe_synthetic.json
```

The CI archive contains this JSON, explicitly labeled synthetic. It cannot
establish real RAN policy superiority. Publication still requires actual
action-dependent trajectories (e.g. independent ns-3/5G-LENA seeds or testbed
measurements), calibrated constraint violations, sensitivity to confounding,
latency and robustness across network slices.

## November 6: Adaptive Pareto Hybrid (APH) -- new optional controller

**Research goal:** combine conformal ranking, slice- and OOD-dependent
constraint weighting, learned logged-behavior preferences, and verified
incumbent hysteresis into one decision policy. Its main objective is better
**all-decision** estimated utility subject to preserving the existing
support/SLA/uncertainty/OOD eligibility envelope.

The model is implemented in `agentic_ran/research_hybrid.py`, and can be
enabled independently of the older optional candidates:

```powershell
cd Agentic-RAN
git pull --ff-only
$env:AGENTIC_RAN_EVAL_ADAPTIVE_HYBRID = "1"
$env:AGENTIC_RAN_EVAL_CONFORMAL_RERANK = "1"
$env:AGENTIC_RAN_EVAL_BEHAVIOR_VARIANTS = "1"
$env:AGENTIC_RAN_EVAL_SAFE_IMPROVEMENT = "1"
$env:AGENTIC_RAN_EVAL_MATCHED_WRAPPERS = "1"
$env:AGENTIC_RAN_EXPERIMENTAL_SUPPORT_GUARD = "1"
docker compose -f docker-compose.publication.yml up --build --force-recreate --exit-code-from publication-test publication-test
```

The publication Docker Compose forwards the new flag into the container.

Nine **preregistered** candidate profiles are tuned only on disjoint
seen-validation episodes: exact conformal anchor, three per-slice risk
profiles, behavior anchor, behavior+risk, incumbent hysteresis, and two
adaptive risk/behavior/switch-margin combinations. The full holdout is used
only **after** choosing one winner per independent crossfit fold.

Profile selection is **baseline-anchored**: a new profile must improve the
model-based full-row validation score by at least 0.00025, preserve >=99.5%
of conformal decision coverage, and avoid estimated SLA-threshold
exceedance regression >0.0025. Otherwise it selects the exact conformal
anchor and reports that fact. The -0.12 abstention score remains an
assumption and is *not* used to claim causal outcomes. As the hybrid and
conformal selector have **exactly the same rejection mask**, their
direct paired difference is invariant to the shared abstention penalty.

The policy does **not** relax hard conformal support or risk constraints to
recover eMBB coverage. Its benefit, if any, must come from more appropriate
eligible action choice. It cannot by itself overcome the logged data's
lack of within-episode scheduler/PRB changes or identify switch effects.
The behavior prior is fitted only on the policy-fit training role, never
on independent OPE or final-test outcomes.

New output files in `results/publication`:
- `research_hybrid_validation.csv`: nine-profile validation search per fold
- `research_hybrid_final_tests.csv`: frozen winner on seen and unseen tests
- `research_hybrid_cluster_evidence.csv`: paired experiment-level evidence
- `research_hybrid_slice_scenario.csv`: per-slice actuation coverage
- `research_hybrid_paired_statistics.json`: hybrid vs conformal;
  also vs Behavior-KL, behavior tree and constraint-conditioned when enabled

The publication summary gains `research_adaptive_hybrid` with selected
profiles, baseline comparisons, per-cluster p-values and CIs. **A
statistically significant model-based advantage is not real-network
superiority**. Future work must provide action-dependent closed-loop
trajectories, independent measurement of holding the incumbent, actual SLA,
PRB/scheduler switching, latency, energy and fresh test scenarios.

## November 9: Twin-CQL and Distributional CVaR — action-dependent experiment

Research methods: (1) a digital-twin-assisted linear finite-action
Conservative Q-Learning (CQL) controller, and (2) quantile-Huber fitted
Bellman return distributions with a lower-tail Conditional Value-at-Risk
(CVaR) objective. A third CQL/CVaR blend tests their complementarity.
The same dynamically logged toy training episodes feed both learners.

Files:

- agentic_ran/research_twin_environment.py: multi-slice action-dependent
  toy simulator for eMBB/mMTC/URLLC with five PRB-split actions;
  randomized, changing action trajectories and explicit executed holds.
- agentic_ran/research_twin_offline.py: conservative logsumexp Bellman
  critic, quantile critic with CVaR objective, episode-calibrated learned
  digital twin, and per-action local state support gate.
- agentic_ran/research_twin_study.py: independent train/validation/
  seen-test/unseen-test partitions, fixed paired episode seeds, hold-last
  outcomes measured by the simulator, Holm-adjusted comparisons.

The implementation is a finite-action linear CQL/quantile-TD research
variant, NOT a faithful deep CQL, QR-DQN or CODAC reproduction.
It is also NOT real Colosseum, ns-3/5G-LENA, srsRAN, or O-RAN deployment.
Toy SLA limits and queue dynamics are constructed, not externally
validated physics or measured power/latency. **Do not attribute the
resulting utility or SLA values to COMMAG or an actual wireless network.**

Run a larger independently evaluated simulated study:

~~~powershell
cd Agentic-RAN
git pull --ff-only
python -m agentic_ran.research_twin_study --seed 2026 --train-episodes 160 --validation-episodes 40 --test-episodes 60 --steps 24 --output results/twin_cql_cvar_study.json
~~~

The JSON reports the action-switch audit, the validation-only selected
hyperparameters, the separately calibrated world-model residual margins,
the validation-selected family, seen and unseen realized returns, all
three per-slice SLA exceedance rates, switching and abstention, and
paired episode-bootstrap 95% CIs plus Holm-adjusted sign-randomization
p-values. The neighboring *_episodes.csv preserves individual paired
episode results.

For honest interpretation:

- No test or validation transitions train the critics; the twin's risk
  residuals are calibrated on episodes disjoint from its training fit.
  Those margins are empirical diagnostics, not certified safety bounds.
- Every critic shares per-action support constraints. When no action is
  eligible, the actual simulator continues the incumbent and observes
  the resulting queue/reward; it is NOT credited an arbitrary penalty.
- CQL is compared with the same-mask greedy world-model baseline;
  distributional CVaR and the blend are compared both to that baseline
  and to a *CVaR-mask-matched* greedy policy to isolate the value of
  learning from the effect of stricter risk gating.
- Hyperparameters and promotion eligibility use validation episodes
  only; both final test environments are held out. Three contenders
  per split have Holm-adjusted p-values; multiple split comparisons
  and model selection still warrant independent replication.
- A positive paired test here shows only advantage in this toy
  intervention environment. It does not establish real-RAN causal
  performance, guarantee SLA compliance, or fix COMMAG's zero observed
  within-episode action switches.

Research foundations are Kumar et al., NeurIPS 2020 Conservative
Q-Learning, and Ma et al., NeurIPS 2021 Conservative Offline
Distributional Reinforcement Learning. Their theorems cannot be
transferred to our simplified linear implementation.

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
