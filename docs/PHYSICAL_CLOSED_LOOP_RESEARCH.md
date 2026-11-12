# Agentic-RAN v3: physical BWP and action-dependent JAGC

Research status: experimental. This implementation introduces a physical
cell-capacity contract, an external closed-loop simulator protocol, and
supported-advantage learning from executed randomized transitions. The
included ns-3 example is an action-dependent discrete-event demonstration,
NOT the 5G-LENA NR MAC/PHY and NOT a Colosseum run.

## A. Physical PRB capacity — historical 14-unit proxy removed

New JointAction = three integer PRB counts and three separately identified
scheduling policies for ONE named carrier and bandwidth part (BWP).
CellBudget includes carrier_id, bwp_id, total_prbs, reserved_prbs,
numerology, source and backend. Every issued action must satisfy:

    eMBB_prbs + mMTC_prbs + URLLC_prbs <= total_prbs - reserved_prbs

The old JAGC publication pipeline now fails closed without the environment
variable AGENTIC_RAN_JAGC_BWP_MANIFEST, pointing at an externally supplied
JSON manifest. Every base station must have an entry. Example SCHEMA ONLY
(values are NOT verified for COMMAG):

~~~json
{
  "schema": "jagc-phy-bwp-v1",
  "cells": {
    "bs1": {
      "backend": "ns3-5glena",
      "source": "nr_phy_observed",
      "carrier_id": "actual-radio-carrier",
      "bwp_id": "actual-downlink-BWP",
      "total_prbs": 52,
      "reserved_prbs": 4,
      "numerology": 1
    }
  }
}
~~~

Do not copy 52/4 into a real analysis without measuring/configuring the
actual NR PHY bandwidth, guard and reserved resources, and BWP identifiers.
The manifest is an externally asserted evidence contract, not independent
cryptographic attestation. Capacity of a different BWP/testbed cannot
validate the original COMMAG dataset's exact physical resource setup.

The original COMMAG-trained policy is frozen from a TRUSTED
publication_models.joblib. Its original independent per-slice decisions
can exceed shared BWP capacity. The executable comparison therefore
projects them to the closest feasible integer allocation and ALWAYS
reports the projection rate. The correct baseline name is
"capacity-projected frozen original Agentic-RAN", NOT "unmodified original".

## B. External ns-3/5G-LENA or Colosseum closed-loop protocol

ExternalClosedLoopSession starts a persistent process, with strict one
JSON object per line on stdout (send build logs to stderr):

~~~text
Simulator -> {"type":"hello","protocol":"jagc-closed-loop-v1",
              "backend":"ns3-5glena","real_actuation":true,
              "cell":{"carrier_id":"...","bwp_id":"...",
                      "total_prbs":...,"reserved_prbs":...,
                      "numerology":...,"source":"nr_phy_observed"}}
Controller -> {"type":"reset","seed":123,"scenario":"normal"}
Simulator -> {"type":"observation","step":0,
              "slices":{"eMBB":{"metrics":{...}},
                        "mMTC":{"metrics":{...}},
                        "URLLC":{"metrics":{...}}}}
Controller -> {"type":"step","action":{"carrier_id":"...",
              "bwp_id":"...","slices":{
                "eMBB":{"prbs":...,"scheduler":"proportional_fair"},
                "mMTC":{"prbs":...,"scheduler":"proportional_fair"},
                "URLLC":{"prbs":...,"scheduler":"proportional_fair"}}}}
Simulator -> {"type":"transition","reward":...,"done":false,
              "observation":{...NEXT STATE AFTER ACTION...}}
~~~

Each slice observation supplies the 17 numeric metrics named by
agentic_ran.offline_policy.STATE_COLUMNS. The backend must apply the
requested allocation to its scheduler before reporting the next
observation and a realized utility from real flow/PHY measurements.
The provisional URLLC check in the runner uses queue bytes, NOT a
verified 5QI packet delay SLA.

Real mode refuses all backend claims except ns3-5glena or colosseum,
NR-PHY-observed capacity, and real_actuation=true. This is a software
assertion, not proof of physically valid scheduling in an arbitrary
external process; the scheduler integration must be reviewed.

### The bundled ns-3 discrete-event demonstration

Copy ns3/scratch/jagc-closed-loop.cc into the scratch/ directory of
an installed ns-3 project (for example ns-3.48), build it, and run
the compiled scratch executable via the runner --sim-command option:

~~~bash
cp ns3/scratch/jagc-closed-loop.cc /path/to/ns-3.48/scratch/
cd /path/to/ns-3.48
./ns3 configure
./ns3 build
~~~

This example calls ns3::Simulator::Schedule and produces post-action
queue and reward changes. However its QUEUE model is simplified, its
reported budget comes from --totalPrbs/--reservedPrbs, and its backend
is ns3-event-demo with real_actuation=false. It cannot be mistaken for
real NR PHY/MAC. You must explicitly use --allow-event-demo.

### Complete the real 5G-LENA experiment on your simulator installation

The external NR scheduler still needs integration. Suggested path:
use 5G-LENA's NrMacSchedulerOfdmaAi with OpenGym for action application,
obtain BWP PRBs from the actual NrGnbPhy/MAC configuration, and
instrument DL per-flow queues, delay, throughput and SLA outcomes after
each applied decision. A PHY/grid feasibility audit must account for
RBG granularity, slots/symbols, HARQ, guard and control reservations.

Official resources:
- https://cttc-lena.gitlab.io/nr/html/cttc-nr-simple-qos-sched_8cc.html
- https://cttc-lena.gitlab.io/nr/html/classns3_1_1_nr_mac_scheduler_ns3.html
- https://5g-lena.cttc.es/

A Colosseum experiment requires an authorized live xApp/DU/near-RT RIC
actuation bridge. This PR does not contain hardware access, access
credentials, or a completed native 5G-LENA/Colosseum control deployment.

## C. Learn from genuine action variation and evaluate realized outcomes

After the original full COMMAG publication run has generated the trusted
model artifact:

~~~bash
python -m agentic_ran.research_closed_loop \
  --sim-command "/ABSOLUTE/PATH/TO/YOUR-5GLENA-CONTROLLER-RUNNER" \
  --models results/publication/publication_models.joblib \
  --output results/jagc-closed-loop \
  --train-episodes 24 --validation-episodes 6 \
  --test-episodes 12 --steps 20 --seed 20261111
~~~

Use --allow-event-demo ONLY for the bundled NON-5G-LENA event example.

The runner uses three disjoint seed partitions. On randomized training
episodes it repeatedly enacts different actual physical allocations,
records post-action realized rewards, rejects static action sequences,
and trains a bootstrap ExtraTrees reward model. A training-fitted local
nearest-neighbor distance checks whether a candidate state-action
resembles explored support. Disjoint exploratory validation episodes
calibrate an absolute prediction-residual margin.

Four predefined gates (including an exact original baseline anchor) are
compared on paired, executed VALIDATION episodes. A new gate is promoted
only if realized mean utility rises by >=0.002 with no more than 0.005
additional URLLC queue exceedance. The selected gate is then frozen
before unseen-seed TEST episodes. The experiment executes original and
JAGC independently under identical scenarios/seeds and reports paired
realized rewards and bootstrap confidence intervals.

Outputs in results/jagc-closed-loop:
- jagc_closed_loop_summary.json
- jagc_closed_loop_validation.csv
- jagc_closed_loop_test_episodes.csv
- jagc_closed_loop_executed_transitions.csv.gz

This test does not guarantee positive gains, physical safety, causal
identification outside the randomized simulator setting, or 10% relative
improvement. An event-demo score cannot be published as 5G-LENA
performance; real backend actuation remains to be completed and run.
