# Experimental comparison — October 26, 2026

**Do not quote these as COMMAG or live-RAN outcomes.** These are a new
independent synthetic action-selection test and a toy closed-loop
interventional queue simulator, selected and evaluated using distinct splits.

## Action-support estimators

Set seed 2026, 3,000 policy-role and 3,000 evaluator-role reference rows,
850 validation rows, 1,400 seen-test rows and 1,400 shifted-test rows.
Competing estimators: kNN, Ledoit-Wolf shrinkage Gaussian, diagonal two-mode
GMM; alpha in {0.01,0.05,0.10,0.20}. The test includes a **calibration
resolution guard** so infeasible 1% tail probabilities do not automatically
accept all candidate actions.

Winner **shrinkage Gaussian alpha=0.05**, by a validation-only composite
of synthetic unconditional utility and abstention rate:

| Metric | Seen | Shifted |
|---|---:|---:|
| Candidate re-ranking mean utility (explicit abstention -0.12) | 0.697057 | 0.760902 |
| Same-gate immediate-deferral utility | 0.600052 | 0.744346 |
| Re-ranking gain | +0.097005 | +0.016556 |
| Re-ranking actuation coverage | 0.944286 | 0.900714 |
| Immediate-deferral actuation coverage | 0.806429 | 0.878571 |
| Conditional oracle risk over selected actions | 0.229481 | 0.225164 |

Ten additional seeds (2021-2030; smaller 1,350-role / 350-validation /
650-test data) selected: GMM five times, kNN three times, Gaussian twice.
Mean utility gain over matched immediate deferral: +0.1216 seen, +0.0386
shifted. These gains reflect re-ranking **within the same eligibility
constraints**; estimator rank is unstable, so **validation-select**, don't
declare one estimator universally superior.

## Interventional planning

30 paired episodes, 20 steps each, common exogenous disturbances.
Learned dynamics: held-out RMSE **0.048672** on synthetic random-action
state transitions, not on real network telemetry.

Paired bootstrap estimated return improvements (higher is better):

| Injected model bias | H2 - H1 (95% CI) | H4 - H1 (95% CI) |
|---|---:|---:|
| 0.00 | +11.25 [6.59,16.67] | +11.47 [6.78,17.06] |
| 0.12 | +9.95 [5.94,14.87] | +10.02 [5.93,15.01] |
| 0.30 | +3.31 [1.48,5.69] | +2.97 [1.08,5.45] |

CIs resample paired episodes, not RF/3GPP deployment populations.
These numerical effects depend on the toy queue dynamics, assumed
energy/switching costs, and model-error construction. There is no
controlled COMMAG outcome or deployment evidence.

## Reproduce

```powershell
python -m pip install -e ".[dev,oran]"
python -m pytest -q tests/test_research_methods.py
python -m agentic_ran.research_benchmark --seed 2026 --episodes 30 --output results/research_methods.json
```

The **optional full-COMMAG conformal reranking** is enabled with
`AGENTIC_RAN_EVAL_CONFORMAL_RERANK=1` in
`docker-compose.publication.yml`; its results are a distinct
`research_conformal_*.csv` and must be validated independently.
Until that run completes, these local results do not show that the revised
controller outperforms the existing Agentic-RAN on real COMMAG traces.
