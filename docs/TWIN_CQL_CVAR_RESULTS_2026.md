# Twin-CQL / Distributional-CVaR: first two interventional studies

Date: 2026-11-09. Both studies ran on the **constructed three-slice toy
digital twin**, not on COMMAG, Colosseum, ns-3, 5G-LENA or a live O-RAN.
Their result files are retained as GitHub Actions artifacts. The existing
publication benchmark has not changed.

## Implementations compared

- Greedy one-step learned twin, with per-action local support/queue-risk gate
- A finite-action, linear **CQL** Bellman critic with an explicit logsumexp
  conservative objective and the same greedy twin support/risk gate
- A quantile-Huber fitted distributional return critic with a lower-tail
  **CVaR** selector and stricter queue-tail risk gate; compared to greedy
  both with and without that same gate
- A weighted CQL/CVaR hybrid, with the same CVaR queue-tail risk gate
- **PULSE-RAN** (Priority-driven URLLC and Load-Sensitive Engine), the
  original queue-priority policy, distinct from the stochastic
  logging behavior and not restricted by the research action-support gate

The simulator actually executes the chosen PRB fractions and records the
next queue, realized reward and realized slice SLA violations. When a
controller declines action, its most recent PRB allocation continues
and the resulting consequences are measured. This makes the synthetic
environment action-dependent, unlike the current static COMMAG logs.

## Initial smoke (30/8/12 train/validation/test episodes; 12 steps)

Selected family: greedy twin. Higher return is better.

| Controller | Seen mean return | Shifted unseen mean return |
| --- | ---: | ---: |
| PULSE-RAN | 0.96452 | 0.76701 |
| Greedy twin | 0.96356 | 0.55401 |
| Twin-CQL | 0.92915 | 0.50731 |
| Distributional CVaR | 0.88479 | 0.46811 |
| CQL/CVaR blend | 0.88494 | 0.49102 |

All three learned challengers lost against greedy twin in this small
sample. Details and paired tests:

https://github.com/vtavakkoli/Agentic-RAN/actions/runs/42580698924

## Larger independent replication (160/40/60; 24 steps)

Training uses 160 independent episodes and 3,840 action-dependent logged
transitions, including **2,491 within-episode action changes**. The
learned twin calibrates upper queue prediction residuals on 32 separate
training-role episodes (768 rows). CQL alpha=0.25 and quantile-CVaR risk
weight=0.35 were chosen on validation only. Validation selected Twin-CQL.

| Controller | Seen return | Unseen weak-channel return | Unseen eMBB SLA violations | Unseen URLLC SLA violations |
| --- | ---: | ---: | ---: | ---: |
| PULSE-RAN | 0.97365 | 0.02591 | 31.60% | 0% |
| Greedy twin | 0.96450 | -1.34109 | 47.78% | 26.81% |
| Twin-CQL | **0.97385** | **-0.86599** | **40.49%** | 27.50% |
| Distributional CVaR | 0.95476 | -1.28206 | 55.42% | 28.26% |
| CQL/CVaR blend | 0.96417 | -1.15771 | 49.72% | 26.25% |

Twin-CQL versus matched greedy twin (60 independent test episodes per split):

| Split | Mean per-episode return advantage | Paired episode-bootstrap 95% CI | Holm-adjusted sign-randomization p |
| --- | ---: | --- | ---: |
| Seen | +0.00935 | [+0.00382, +0.01632] | 0.0015 |
| Unseen weak-channel | +0.47509 | [+0.21679, +0.76584] | 0.0052 |

Both pairwise CQL improvements are statistically resolved for this toy
experiment under the stated per-split tests; they are **not** universal
superiority. Episode-level win rates were 26.7% seen and 58.3% unseen:
the mean gains are not uniform across episodes. The unseen PULSE-RAN is much stronger than all learned
controllers, and CQL's unseen URLLC violation rate is **27.5%**, which
would not be suitable for reliability-critical deployment. The modest
URLLC increase over greedy was within an exploratory 1.5 percentage-point
tolerance, not evidence of operational safety.

The CVaR and hybrid models have no statistically resolved improvements
over their own same-risk-gate greedy control. They should not be promoted
as superior on the basis of these experiments.

Reproducible workflow and replication artifacts:

https://github.com/vtavakkoli/Agentic-RAN/actions/runs/42581181720

## PULSE-RAN naming and new comparison protocol

The previously reported `heuristic` controller is now **PULSE-RAN**,
**Priority-driven URLLC and Load-Sensitive Engine**. Its queue thresholds
and priority order are unchanged: first URLLC queue above 0.45, then
eMBB queue above 1.0, otherwise balanced PRB allocation. This is
a naming change, not a newly trained model or adjusted result.

From the newly generated studies onward, the controller is written
as `pulse_ran` in the JSON and per-episode CSV. `pulse_ran_gated`
is a **separate** diagnostic that applies the same learned-twin
action-support and queue-risk gate as Twin-CQL; when its preferred
action fails, the simulator executes an observed hold-last outcome.
The unshielded PULSE-RAN remains distinct, so direct comparisons
must disclose its different eligibility policy.

The updated experiment adds paired episode-level CIs and Holm-adjusted
tests **against PULSE-RAN** for every challenger, plus a separate
`validation_ranked_all` that includes PULSE-RAN.
The previous learned-family validation selector is unchanged and
does *not* automatically promote PULSE-RAN; its selection should not
be interpreted as proof of global superiority. Older artifacts
with the label `heuristic` remain historically valid.

Use `python -m agentic_ran.research_pulse_report --input <results.json>`
to inspect the new ranked comparison table.

## What these experiments establish—and do not

They establish that **the implemented learning mechanisms can be trained,
executed, and compared on action-dependent trajectories**, and that more
training data can change which controller validation selects. They do not
establish superior real RAN resource control, guaranteed SLA safety,
deployment readiness, or transfer of CQL/CODAC mathematical theorems
to this finite-action linear implementation.

The initial and larger runs differ in training size, independent seeds,
step horizon and test sizes; their absolute utility numbers should not
be interpreted as paired across runs. The larger replication was
exploratory and should be independently replicated before a publication
claim. Per-split Holm correction covers the three new challengers.
The stated confidence intervals do not correct for all future research
model selection or environmental misspecification.

The **current COMMAG dataset contains zero within-episode action changes**;
its original offline publication results are a separate study. Real-RAN
superiority requires ns-3/5G-LENA, Colosseum or an actual xApp testbed,
with real action interventions, correct energy units, measured URLLC tail
delay/packet-loss, radio latency, and independent radio/scenario seeds.
