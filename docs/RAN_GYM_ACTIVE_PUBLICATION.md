# Active RAN Gym publication experiment

## Exact scope

A full COMMAG offline benchmark is **not** evidence that a controller improved
a network after its actions were executed. `pulse-commag` is a descriptive
shadow replay. A toy three-slice RAN is **only** a deterministic pipeline smoke test.

The optional **ns-O-RAN Gym** backend runs against a separately installed
ns-O-RAN/ns-3 simulator (wineslab/ns-o-ran-gym, `nsoran` package).
Its stock TrafficSteeringEnv operates on *UE handovers*, not COMMAG
scheduler/PRB actions; stock EnergySavingEnv operates on *cell energy states*.
The 17 COMMAG features and scheduler:prb labels must NEVER silently be
reinterpreted as traffic steering, handover or energy-saving commands.

This branch supplies a strict Gymnasium matching/evidence harness, a toy test
fixture, a native ns-o-ran traffic-steering environment factory, and a
**fail-closed interface** for a separately validated trained-policy adapter.
It does **not** supply a verified mapping for the existing COMMAG-trained
Agentic-RAN controller, install ns-3, run a real Colosseum experiment, or
demonstrate measured superiority. A source-identical comparison is impossible
until the target environment exposes compatible state/action semantics.

## Windows Docker quick commands

From the repository root, with Docker Desktop enabled:

~~~powershell
git pull --ff-only
# Lightweight unit tests and deterministic executed toy actions:
docker compose -f docker-compose.ran-gym.yml build ran-gym-unit ran-gym-smoke
docker compose -f docker-compose.ran-gym.yml run --rm ran-gym-unit
docker compose -f docker-compose.ran-gym.yml run --rm ran-gym-smoke

# Full offline results plus toy checks (NOT active publication evidence):
.\scripts\verify_publication.ps1
~~~

The complete automated verification uses the full COMMAG dataset and may be
expensive. Preparation caching is reused, while the cross-fitted publication
benchmark is rerun. Pass `-FullResearch` to include the optional support guard,
matched wrappers, conformal reranking, behavioral methods, safe improvement and
adaptive hybrid ablations. Add `-RunJointSlice` **only** after supplying a
verified physical BWP capacity manifest. The full-research run may take much
longer because the publication cross-fit loop is still sequential.


## Using the actual frozen COMMAG-trained Agentic-RAN controller

For an ns-3/5G-LENA or other external Gym **whose action space really allows
scheduler and physical PRB selection**, start from
`configs/ran_gym_prb_compatible.example.json`, not the stock traffic-steering
example. Copy it to `configs/ran_gym_active.json`.

The built-in
`agentic_ran.research_ran_gym_adapters:make_frozen_commag_agentic_policy`
loads a **specific frozen cross-fit policy fold** from
`results/publication/publication_models.joblib`, applies its calibrated
SLA/uncertainty/OOD thresholds and switching margin, and calls the repository's
actual `_policy_precompute` / `_choose_from_precomputed` logic.

It requires TWO externally implemented, scientific mappings in
`external/site-adapters/site_adapter.py`:

- `commag_frame_from_physical_observation(obs, previous, t, info, config)`:
  returns genuine per-slice rows with the **17 original COMMAG features**,
  slice type, logged/current scheduler+PRB action, episode ID and timestamp.
  Units, temporal aggregation and missing features must be reconciled with
  physical telemetry; never fill unobserved state with invented constants.
- `translate_scheduler_prb_to_executed_gym_action(selected_actions, frame,
  obs, previous, t, info, config)`: returns the exact Gym action encoding the
  three selected scheduler/PRB choices. Validate shared BWP PRB capacity
  and actuation semantics. Avoid silently projecting invalid allocations;
  report projection separately if the experiment defines one.

The fixed `policy_fold` and `frozen_model_sha256` must be chosen before
final-test simulations. Generate the checksum using:

~~~powershell
(Get-FileHash .\results\publication\publication_models.joblib -Algorithm SHA256).Hash.ToLower()
~~~

**This model inference path is implemented; the required external PHY/MAC
observation/action mapping and real simulator have not been implemented by this
repository.** No superiority result exists until these are independently
installed, verified and executed. Avoid invoking the built-in COMMAG policy on
the stock traffic-steering/energy-saving environments.

## Actual external ns-O-RAN Gym experiment

1. Install a **Linux-compatible compiled ns-O-RAN/ns-3** under an accessible
   host directory. Merely mounting a Windows binary or the source tree will
   not provide a runnable ns-3 simulation. Set `$env:RAN_GYM_NS3_DIR` to that
   directory.
2. Copy `configs/ran_gym_active.example.json` to
   `configs/ran_gym_active.json` and set the actual simulator revision,
   simulator scenario JSON path, model source and checksum. The example is
   intentionally **not runnable** until the missing scientific mapping is supplied.
3. Provide `external/site-adapters/site_adapter.py` with explicit factory
   functions for the frozen candidate and at least two genuine **same-action-
   space** baselines. Factory signature:
   `factory(config: dict, action_space) -> policy(obs, previous_action, t, info) -> action`.
   The environment factory signature is
   `factory(scenario: str, seed: int, config: dict) -> GymnasiumEnv`.
4. For the stock nsoran traffic-steering example: the controller must be a
   **traffic-steering-trained controller** with relevant features. Using
   COMMAG-fitted scheduler/PRB models requires a **different compatible**
   simulator environment with physically actuated scheduler and PRB allocation.
   If no verified transformation exists, do not set
   `observation_action_mapping_documented=true`.
5. The environment must return Gymnasium's five-tuple from `step` and
   acknowledge the **exact applied action** via `info["applied_action"]`.
   It must also expose `info["throughput_mbps"]`, `info["latency_ms"]`
   and `info["sla_violation_rate"]` as finite measured quantities for
   a publication-grade gate. Stock nsoran envs may need an **instrumented,
   independently validated** adapter to supply these. Merely echoing input
   commands is not an executed physical acknowledgement.
6. Run matched policy episodes with identical scenario/seed exogenous
   conditions. Confirm that resetting a scenario is deterministic and
   that environmental randomness does not depend on policy-specific
   training state. Use independent training, validation and final-test seeds.
7. Inspect the source hash of each policy factory and model checksum.
   Never load untrusted `.joblib` artifacts.

~~~powershell
$env:RAN_GYM_NS3_DIR = "C:\path\to\linux-compatible\ns3"
.\scripts\verify_publication.ps1 -FullResearch -External
~~~

Run the active experiment without repeating the expensive offline run:

~~~powershell
docker compose -f docker-compose.ran-gym.yml --profile external up --build --exit-code-from ran-gym-active ran-gym-active
docker compose -f docker-compose.ran-gym.yml run --rm publication-final-gate
~~~

`ran-gym-active` defaults to `require_applied_action_receipt=true` and
**fails closed** when the backend cannot return an exact action receipt.
The final gate requires at least two scenarios, ten distinct test seeds
(at least 20 pairs per policy), two baselines, paired bootstrap intervals,
Holm-adjusted sign-flip tests, complete throughput/latency/SLA fields,
and a conservative SLA non-regression check. You must extend the stock
`scenario-one` nsoran TrafficSteering setup to independent scenario
variants if using that backend; the supplied example intentionally fails
this gate until then.

Output files:
- `results/ran-gym-smoke/ran_gym_summary.json`
- `results/ran-gym-active/ran_gym_summary.json`
- `results/ran-gym-active/ran_gym_episodes.csv`
- `results/ran-gym-active/ran_gym_steps.csv.gz`
- `results/publication/final_evidence_gate.json`

## Interpretation and reporting

The benchmark compares candidate total realized episode rewards to each
baseline in matched independent runs (not rows or correlated time steps).
It calculates 95% paired episode bootstrap confidence intervals, one-sided
paired sign-flip tests, and Holm multiplicity adjustment. It separately
records switching, throughput, latency, SLA outcomes and missing telemetry.
A positive mean alone is not evidence of statistically robust superiority.

The current gate checks files and flags; it **cannot independently verify**
ns-3 fidelity, whether the simulator really applied commands, IEEE novelty,
source-data validity, or that a provided adapter is scientifically sound.
Do not use successful toy Gym checks or a user-supplied backend label as
a certification of a physical experiment.

References:
- https://openrangym.com/
- https://github.com/wineslab/ns-o-ran-gym
- https://gymnasium.farama.org/api/env/
