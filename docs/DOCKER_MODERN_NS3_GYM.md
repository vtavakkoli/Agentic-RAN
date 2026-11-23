# Verified ns-3.49 Gymnasium backend (native UDP scope)

**Real native test upstream:** [vtavakkoli/ns3-gym GitHub Actions](https://github.com/vtavakkoli/ns3-gym/actions/runs/45200458924).

## Dependency pins and method

- ns-3 Gym **source**: [vtavakkoli/ns3-gym](https://github.com/vtavakkoli/ns3-gym), commit `21e68565a6beaf13589fd19778b5332f4d0b29e0`.
- ns-3 **upstream**: ns-3.49, Git commit `418210e8914e9f6bcb5cddbcc94a2cecb7853b68`.
- Python **3.13**, base image `python:3.13-bookworm`, **C++23/CMake**.
- Protobuf is generated for Python and C++ with `protoc` in the same Docker build.
- Transport: native ZeroMQ request/response; no runtime dependency on upstream tkn-tub/ns3-gym, wineslab/ns-o-ran or Waf.
- `Dockerfile.ran-gym` builds the source at *both exact SHAs*, installs the Python package with `python3.13 -m pip` and compiles the ns-3 C++ scenario.

The native target **builds and executes real simulation tests during Docker build**.
A separate GitHub Actions workflow then runs full episodes again and saves
the actual observations, actions, rewards and measured UDP packet counters.

## Windows Docker Desktop commands

Retain local corporate `gateway.crt`/`root.crt` injection **only in your own local Dockerfile**. Do not commit certificate material. If local HTTPS interception is present, inject those certificates into the new `modern-ns3` stage as well.

```powershell
$env:MODERN_NS3_BUILD_JOBS = "4"
docker compose --progress plain -f docker-compose.ran-gym.yml --profile native build ran-gym-modern-ns3-smoke
if ($LASTEXITCODE -ne 0) { throw "Pinned native ns-3 build/real tests failed" }

docker compose -f docker-compose.ran-gym.yml --profile native run --no-build --rm ran-gym-modern-ns3-smoke
if ($LASTEXITCODE -ne 0) { throw "Real native closed loop failed" }
```

Output: `results/modern-ns3/real_closed_loop.json`.

## Tested, actual semantics

A two-node, 5 Mbps, 2 ms delay ns-3 point-to-point link. Actions
`Discrete(6)` mean **0 to 5 UDP packets of 256 bytes** sent in the
next 100 ms of simulation. Real receiver callbacks supply RX and bytes.
The native socket counter confirms the applied action in
`agentic_ran.research_modern_ns3_adapter.VerifiedNs3UdpEnv`.
It fails on invalid or mismatched receipt; the safety rule chooses a
no-send fallback for invalid *integer packet budgets* and rejects
non-packet (e.g. PRB) action types.

It tests fixed-seed determinism, multiple fresh native processes/episodes,
termination, per-step reward/throughput, timeouts, native cleanup and
strict unsupported-use-case exceptions. Regular Python CI may skip native
tests when `VTNS3GYM_BINARY` is absent; the separate native workflow
**must not** skip them.

## Critical scientific boundary

**This is not an NR PHY, PRB scheduler, E2 or COMMAG controller.**
The native action is UDP packet count; it cannot be reinterpreted as
the RAN slice action without writing a new action-dependent native NR
scenario and a documented state/action mapper.
`publication_superiority_ready` and
`commag_prb_controller_evaluated` remain **false**, including when
the native UDP smoke succeeds.

The existing safety-governed research pipeline, frozen COMMAG model,
paired tests, toy benchmarks and production Docker image are untouched.
The external PRB experiment continues to require a separate, independently
validated external backend; no synthetic fallback occurs.
