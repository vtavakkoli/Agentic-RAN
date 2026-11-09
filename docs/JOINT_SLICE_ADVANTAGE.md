# Agentic-RAN v3 — Joint-Slice Advantage-Gated Control (JAGC)

**Status: experimental offline research option.** This is a new
controller candidate, not a demonstrated 10% gain or a safe live-RAN xApp.
The original Agentic-RAN controller remains the default and is the exact
fallback/validation anchor.

## What changes

The existing COMMAG publication controller makes per-UE/slice decisions
using trained policy critics, OOD gating, SLA screening, and estimated
utility. v3 adds a JOINT decision layer:

1. Build matched **eMBB, mMTC and URLLC** contexts at a common COMMAG
   scenario, training configuration, experiment, base station, and
   one-second timestamp. Aggregate the *recorded* multiple UE KPI
   observations by mean within the same BS/slice/second. Discard and
   report BS-seconds missing any slice; do not fabricate missing context.
2. Estimate a **PRB-budget proxy** from contemporaneous slice-PRB counts
   in the *policy-fit training role only* (90th percentile by base
   station). It is a statistical proxy for testing, **not** a measured
   physical gNB resource cap or a validated joint-actuation protocol.
3. Consider candidate recorded COMMAG scheduler/PRB action labels only
   when observed in both policy and independent OPE training roles.
   Apply the original policy-side SLA/uncertainty/OOD eligibility gates
   to every slice action and require evaluator action-label support.
4. Solve a **multiple-choice knapsack**: exactly one candidate per slice,
   total candidate PRB counts within the training proxy budget.
   Dynamic programming finds the highest-scoring feasible combination;
   no greedy action truncation removes cheap feasible choices.
5. Apply a **conservative advantage proxy** against the *unchanged*
   original Agentic-RAN decisions. Account for the original critic's
   uncertainty and any switching penalty; select the new joint actions
   only if their policy-side score passes a predefined improvement
   margin. This is NOT a calibrated advantage confidence interval.
   If no eligible joint option exists or the margin is unmet, retain
   the reference controller and report the reason.
6. Choose from seven predefined profiles **using validation only**.
   A new profile needs at least +0.002 estimated mean utility over the
   original on matched validation BS-seconds, no more than +0.005
   estimated SLA exceedance regression, and no deterioration in
   estimated budget compliance. Otherwise retain original Agentic-RAN.
7. Freeze one selected profile **per crossfit experiment fold** before
   evaluating seen and unseen test conditions. The independent
   outcome model is used *only* for validation and held-out reporting,
   not action scoring. Compare all controllers on the same aligned
   BS-second population, with paired experiment-cluster exploratory
   intervals and sign-randomization diagnostics.

## Run in Docker on real COMMAG recordings (Windows PowerShell)

~~~powershell
cd Agentic-RAN
git pull --ff-only

$env:AGENTIC_RAN_EVAL_JOINT_SLICE_ADVANTAGE = "1"
$env:AGENTIC_RAN_EVAL_MATCHED_WRAPPERS = "1"
$env:AGENTIC_RAN_EXPERIMENTAL_SUPPORT_GUARD = "1"

docker compose -f docker-compose.publication.yml up --build --force-recreate --exit-code-from publication-test publication-test
~~~

The existing full COMMAG data preparation is run first if needed.
The default publication workflow is otherwise unchanged when the flag is off.

Outputs under results/publication:

- research_joint_slice_validation.csv: original and six joint profiles
  per crossfit fold; held-out outcomes are not used here
- research_joint_slice_final_tests.csv: frozen validation-selected
  profile on seen/unseen, paired with original on **identical** aligned
  BS-second evaluation population
- research_joint_slice_decisions.csv.gz: selected/original three-slice
  actions, group-level independent-DM difference, budget diagnostics,
  cluster/scenario/config/experiment identifiers
- research_joint_slice_selection.json: training-only budget calibration,
  group alignment and paired-cluster uncertainty per split
- publication_summary.json: research_joint_slice_advantage section

Check the results with PowerShell:

~~~powershell
Import-Csv .\results\publication\research_joint_slice_final_tests.csv |
    Format-Table fold,split,profile,independent_dm_utility,original_independent_dm_utility,independent_dm_delta,estimated_sla_delta,joint_selection_rate,training_proxy_budget_compliance -AutoSize

Get-Content .\results\publication\research_joint_slice_selection.json
~~~

The previous measured-on-COMMAG benchmark value of 0.713381 came from
the full **UE-row** policy evaluator. JAGC reports a **joint BS-second**
estimand using aggregated three-slice states. **Do not compare their
absolute utility values**. Instead, use the original-vs-joint delta
provided by JAGC *on its matched population*.

## 10% target: validation-only optimistic model headroom

The selected-fold JSON also contains `validation_model_headroom`.
It reports the independent outcome evaluator's *optimistic* best
supported joint configuration on validation BS-seconds (under the
same PRB-budget proxy), compared with the original policy on those
exact same cells. It includes a +10% relative target indicator.
**It is not a statistically valid upper confidence bound, executable
control policy, proof of achievable improvement, or held-out result**.
The evaluator's winner can be biased upward by maximizing noisy
predictions, so the estimate is only a feasibility diagnostic.

The original COMMAG all-decision value (0.713381) is on a different
population; do not compare it to this aligned-cell oracle number.

## What JAGC does NOT yet establish

- COMMAG recorded historical policies; proposed joint interventions
  were not executed. All counterfactual gains are independent model-based
  direct-method estimates and cannot establish causal superiority.
- The PRB budget uses recorded per-slice units. These have not been
  verified as an exhaustive physical gNB/DU capacity constraint.
  Synchronizing counters is necessary but insufficient for an
  executable PRB allocation across slices.
- Critic uncertainty is not a calibrated action-advantage lower bound;
  predicted SLA screening is not a worst-case reliability guarantee.
- Independent UE samples within one BS-second and multiple BS-seconds
  within one experiment are correlated; primary exploratory inference
  aggregates scenario/config/experiment clusters.
- Missing aligned slice observations are excluded rather than inferred,
  so coverage/alignment rate must be reported.
- ns-3/5G-LENA or a new Colosseum run is **not executed** by this workflow.

Before claiming a 10% realized improvement or publishing JAGC as a
superior control algorithm, reproduce its action space in an
**interactive, action-dependent** ns-3/5G-LENA or Colosseum scenario,
use matched radio seeds and per-slice SLA budgets, and compare
executed original Agentic-RAN, PULSE-RAN, FQI, CQL, and JAGC.
The code will report an original-controller winner if validation
does not support the new policy; it does not force any gain.
