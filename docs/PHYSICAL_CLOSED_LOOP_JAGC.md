# Agentic-RAN: physical PRB capacities and external closed-loop JAGC

**Research integration only.** This PR adds validated data structures,
a strict external execution protocol, and training/evaluation logic.
No actual ns-3/5G-LENA executable, new Colosseum trial, or superior
realized JAGC result is provided. GitHub CI uses a fake JSON-lines
bridge to test the protocol, not radio physics.

## 1. Verified cell budget, not the 14-unit log proxy

The NR physical PRB capacity module computes FR1 carrier NRBs from
3GPP TS 38.104 Table 5.3.2-1. Example: 20 MHz and 30 kHz SCS
give 51 physical PRBs before resource reservations. Unsupported
bandwidth/SCS pairs, excess reservations, and any aggregate slice
overallocation are rejected.

For the existing COMMAG offline experiment, use an independently
validated base-station-to-cell map. Start with
configs/commag_physical_cells.example.json (illustrative only).
It has confirmed_commag_prb_units=false intentionally.
Set this to true ONLY after checking that COMMAG action tokens
and slice_prb measurements really count the same physical NRBs.
Unknown stations are rejected. Unless a real implementation
supports per-slice scheduling, the joint optimizer selects one
common scheduler across all slices.

PowerShell:

~~~powershell
cd Agentic-RAN
git pull --ff-only
$env:AGENTIC_RAN_EVAL_JOINT_SLICE_ADVANTAGE = "1"
$env:AGENTIC_RAN_NR_CELL_CONFIG = "/workspace/configs/commag_physical_cells.example.json"
docker compose -f docker-compose.publication.yml up --build --force-recreate --exit-code-from publication-test publication-test
~~~

The historical proxy path remains as a legacy, explicitly
nonphysical result for reproducibility. The physical option
does not magically make observational counterfactuals causal.

## 2. External ns-3/5G-LENA or Colosseum action-dependent bridge

research_external_ran.py implements a persistent JSON-lines
subprocess protocol named agentic-ran-physical-prb-v1.
A separately installed real simulator/testbed executable must:

- Reset on the requested scenario and seed, starting from matched
  initial conditions and exogenous disturbances.
- Receive and actually APPLY a three-slice, integer PRB allocation
  that respects the physical budget, plus an explicit scheduler.
- Advance time and return a matching applied-allocation receipt,
  measured next RAN state, realized reward, per-slice SLA flags,
  throughput and latency, a new step index, and done state.
- Support a fresh reset for each policy, with the same scenario
  and seed. The client fails closed on stale or absent receipt,
  malformed responses, invalid resource requests and timeouts.

The client itself cannot verify external ns-3 or Colosseum backend
identity. Archive the actual backend source, topology, simulator
version, PHY/MAC configuration and measurement logs before claiming
radio-system evidence.

The example configs/physical_ran_experiment.example.json contains
a NONFUNCTIONAL placeholder bridge_command, not an installed
ns-3 or Colosseum environment. Those installations and testbed
credentials are external prerequisites.

## 3. Supported advantage learning from ACTUATED exploration

research_realized_advantage.py performs uniform randomized executed
training actions with known nonzero propensities. It checks
action receipts, within-episode action changes and coverage of
all compared allocations. The module rejects static COMMAG logs.

A cluster-bootstrap ExtraTrees ensemble estimates action rewards;
the lower fifth percentile of a paired advantage distribution
provides an empirical conservative promotion gate, **not a formal
confidence interval or calibrated causal safety guarantee**.

The experiment runner fits only on training seeds, chooses one
of four prespecified margins on disjoint validation seeds,
then compares the selected controller with the FROZEN reference
on identical held-out scenario/seed pairs. It falls back to the
reference when validation shows no gain without SLA regression.

The runner requires a REAL original-controller factory, for example:

~~~python
def build_policy(cell):
    def original_agentic_ran(state, previous_allocation, timestep):
        # Map actually measured features into the frozen
        # original controller's expected schema.
        # Return a physical SliceAllocation.
        ...
    return original_agentic_ran
~~~

Important blocker: the existing original Agentic-RAN COMMAG
bundle expects 17 per-UE counters and per-slice scheduler/PRB labels,
while the external protocol supplies 9 joint features and physical
joint resource actions. A validated and faithful state/action
translation and frozen original model adapter are NOT yet included;
do not substitute a heuristic and call it original Agentic-RAN.
The supplied source module SHA-256 is captured for provenance.

Once the real simulator bridge and faithful baseline adapter exist,
invoke one line in PowerShell:

~~~powershell
python -m agentic_ran.research_physical_experiment --config configs/physical_ran_experiment.example.json --original-factory your_baseline_module:build_policy --output results/physical-jagc
~~~

This produces randomized training evidence, fitted model artifact,
validation choices, and paired realized test rewards/SLA metrics.

## Publication scope

The new physical resource model and strict action-dependent trial
*infrastructure* are implemented and covered by mock bridge tests.
Neither a real ns-3 topology nor Colosseum control session has been
provisioned or executed. The original-controller adapter must be
faithfully implemented, and actual held-out gains plus cluster-level
uncertainty intervals are needed before claiming superiority or
publication-readiness. No 10% improvement is asserted.

Reference: 3GPP TS 38.104 Table 5.3.2-1; confirm revision and
target carrier configuration in the testbed.
