# Self-contained Docker: ns-O-RAN + ns-3 + Gymnasium

The **ns-O-RAN simulator is downloaded and compiled inside Docker**. No
host-side ns-3 installation, Python setup, source checkout, or VM setup is
required; Windows requires Linux containers in Docker Desktop.

## One command: build and test native upstream simulator

~~~powershell
docker compose -f docker-compose.ran-gym.yml --profile native up --build --exit-code-from ran-gym-nsoran-smoke ran-gym-nsoran-smoke
~~~

This invokes the `nsoran` Docker build target with:
- Python 3.13 runtime on Debian 12 Bookworm with Python 3.11 specifically for legacy Waf; Agentic-RAN, Gymnasium and upstream `nsoran==0.0.2`
- e2sim from pinned `wineslab/o-ran-e2sim` commit
- pinned ns3-mmwave O-RAN release and pinned O-RAN E2 interface
- native compiled `scenario-one` traffic-steering environment
- a real `reset(seed=2026)` and a few **no-handover** `step` actions

Success should create:
`results/nsoran-native/native_gym_smoke.json`.

## Fixed: pinned simulator is Waf-based, not CMake/ns3

The pinned `wineslab/ns-o-ran-ns3-mmwave` revision
`0efbfaf370350c170beb249951e5c349d7f89b34` has a root `waf`
launcher and `wscript`, but **no** `ns3` launcher. Consequently
`./ns3 configure` fails with exit code **127** (command not found).
It must instead be built with the legacy Waf toolchain:

~~~bash
cd /opt/ns3-mmwave-oran
/usr/bin/python3.11 ./waf configure --enable-examples --build-profile=optimized --out=build/optimized
/usr/bin/python3.11 ./waf build -j 4
test -s build/optimized/build-status.py
grep -q 'scenario-one' build/optimized/build-status.py
~~~

The upstream `nsoran` Python wrapper loads this Waf-generated
`build/optimized/build-status.py` to locate the `scenario-one`
executable. The newer ns-3 CMake `.lock-ns3_linux_build` is not
the correct marker for this pinned revision. Neither the pinned
`wscript` nor its documented configuration explicitly provides
`--disable-werror`, so the Docker build omits that flag.

The native simulator uses **Python 3.13 Bookworm** as requested, while installing
Debian's `/usr/bin/python3.11` for the pinned legacy Waf build scripts.
Bullseye reached the end of LTS on **31 August 2026**. Since September
2026, its `bullseye-security` metadata has referred to packages
that no longer exist at the public mirror; this yields HTTP 404
during `apt-get install`, even after `apt-get update` and without
a proxy/TLS failure. Using `python:3.13-bookworm` avoids the
unsupported Bullseye package indexes. Other research stages
also continue using Python 3.13.

Because the native ns-3/Waf source is older, Debian 12's newer C++
toolchain can expose separate compatibility problems in later build
steps; **APT repair does not itself prove that C++ compilation or
native Gym execution succeeds**.

Before re-running, inspect the native stage:
```powershell
Select-String -Path .\Dockerfile.ran-gym -Pattern '^FROM python:3.13-bookworm AS nsoran'
```
If that returns no match, your local file has not incorporated the
latest patch. Preserve your local proxy CA changes, back up the
Dockerfile, and apply the updated native base line.

Use the correct Compose global `--progress` position:
```powershell
$env:NSORAN_BUILD_JOBS = "4"
docker compose --progress plain -f docker-compose.ran-gym.yml --profile native build --pull ran-gym-nsoran-smoke
if ($LASTEXITCODE -ne 0) { throw "Native build failed: inspect the first error" }
docker compose -f docker-compose.ran-gym.yml --profile native run --rm ran-gym-nsoran-smoke
if ($LASTEXITCODE -ne 0) { throw "Native Gym execution failed" }
```

## Python 3.13's missing `pipes` module (Waf fix)

The pinned ns-O-RAN Waf tool (`waf 2.0.21`) is an older Python program.
Python 3.13 removed the standard-library `pipes` module, so running
`python3 ./waf configure` in the `python:3.13-bookworm` image fails
with `No module named 'pipes'` when it loads
`clang_compilation_database`. This is the user's actual Docker
`nsoran 17/18` error, not an SSL/proxy issue.

The native stage therefore adds
`python3.11 python3.11-dev python3-distutils` from Debian 12
and runs Waf **only** with `/usr/bin/python3.11`:

```dockerfile
FROM python:3.13-bookworm AS nsoran

# In the native APT dependencies:
# python3.11 python3.11-dev python3-distutils

RUN test -f ./waf \
    && /usr/bin/python3.11 -c 'import pipes' \
    && /usr/bin/python3.11 ./waf configure --enable-examples --build-profile=optimized --out=build/optimized \
    && /usr/bin/python3.11 ./waf build -j "${NSORAN_BUILD_JOBS}" \
    && test -s build/optimized/build-status.py \
    && grep -q 'scenario-one' build/optimized/build-status.py \
    && ln -s /opt/ns3-mmwave-oran /opt/ns3
```

The project package and Gym runtime still use the image's Python 3.13
(`python -m pip`, `python -m agentic_ran...`). Waf's compatible
Python 3.11 interpreter is a local build dependency, not a downgrade
of the container's primary Python version. A subsequent C++ compiler
or runtime issue must be diagnosed independently.

**The original failure** at `apt-get install` exit code 100 is
distinct from the previously repaired `./ns3: not found` exit code 127.
Do not use `--fix-missing` to paper over permanently absent
Bullseye security packages, and do not disable APT signature checks
to bypass this problem.

For Windows PowerShell (Docker Desktop Linux containers):

~~~powershell
git switch main
git pull --ff-only origin main
$env:NSORAN_BUILD_JOBS = "4"
docker compose --progress plain -f docker-compose.ran-gym.yml --profile native build --pull ran-gym-nsoran-smoke
if ($LASTEXITCODE -ne 0) { throw "Native build failed; inspect first actual compile error" }
docker compose -f docker-compose.ran-gym.yml --profile native run --rm ran-gym-nsoran-smoke
if ($LASTEXITCODE -ne 0) { throw "Native Gym execution failed" }
~~~

**Corporate proxy / private root CAs:** If your local
`Dockerfile.ran-gym` adds `COPY gateway.crt` and `COPY root.crt`,
keep those certificates **local**, outside git, and preserve your
trust-store modifications when applying the upstream Dockerfile fix.
Uncommitted local Dockerfile edits may conflict with `git pull`;
save the local file first and reapply only your local CA setup after
updating the repository. Do not commit or upload confidential
certificate material. This patch does not require changing any CA files.

**Verification boundary:** GitHub CI exercises the Waf/CMake
marker-selection tests but does **not** compile ns-3. An actual
native Docker build and `TrafficSteeringEnv.reset/step` are
only verified after the commands above succeed. This remains
handover/traffic-steering, *not* COMMAG PRB/scheduler execution.

**Build note:** The native C++ ns-O-RAN stack is old and complex and its Docker
build is not run in the standard GitHub CI because of resource requirements.
The source-build instructions are included and pinned, but compilation and
runtime compatibility have **not yet been verified on the user's Windows
Docker host**. A nonzero build/run result must be investigated; never replace
it with synthetic traces and call it a successful ns-O-RAN experiment.
To reduce C++ memory consumption use
`$env:NSORAN_BUILD_JOBS = "4"` before running the command.
Docker Desktop should have at least 8 GB allocated to the build.

The smoke-test policy is NOT the COMMAG-trained Agentic-RAN model. The
upstream TrafficSteeringEnv chooses *handover targets*, not RAN slice
scheduler/PRB allocations; a source-compatible controller cannot be
manufactured by changing action IDs. The smoke test is labeled
`publication_superiority_ready=false` and doesn't pass the publication gate.

## One command: all existing offline tests + real native Gym smoke

~~~powershell
powershell -ExecutionPolicy Bypass -File .\scripts\verify_publication.ps1 -FullResearch -NativeGym
~~~

It additionally checks publication COMMAG splits, cross-fit performance,
PULSE descriptive shadow results, Gym regression tests and toy actions.
Native Gym smoke remains separate from research evidence.

## Full active comparison for your *frozen* COMMAG model

The Docker target includes the ns-3 binary; it does NOT make the stock
handover/energy-saving environments execute PRB/scheduler actions. To test
the frozen COMMAG controller with real closed-loop actions requires:

1. A valid action-dependent Gym environment/modified ns-3 scenario that
   implements supported eMBB, mMTC, URLLC scheduler/PRB decisions with a
   real carrier/BWP resource budget and matching training-state telemetry.
2. A verified observation and action mapping, provided as Python source under
   `external/site-adapters/site_adapter.py`, with the functions shown in
   `configs/ran_gym_prb_compatible.example.json`. It must return a true
   applied-action receipt and measured throughput, latency and SLA KPIs.
3. A trusted `results/publication/publication_models.joblib` generated by
   the matched offline benchmark, and its SHA256 in the active configuration.
   The model must be frozen **before** held-out active tests.
4. Copy `configs/ran_gym_prb_compatible.example.json` to
   `configs/ran_gym_active.json` and configure the exact real environment,
   adapter functions, independent seeds, and simulator revision. Use
   `/opt/ns3` for the in-container ns-3 path.

Once **and only once** the compatible backend exists:

~~~powershell
docker compose -f docker-compose.ran-gym.yml --profile external up --build --exit-code-from ran-gym-active ran-gym-active
docker compose -f docker-compose.ran-gym.yml run --rm publication-final-gate
~~~

The active benchmark compares realized held-out episodes using matched
seeds, cluster bootstrap confidence intervals, multiplicity correction and
service/SLA checks. It refuses to claim statistically demonstrated
superiority if the comparisons fail.

## Source attributions

- Official ns-O-RAN build: https://openrangym.com/tutorials/ns-o-ran
- Upstream simulator: https://github.com/wineslab/ns-o-ran-ns3-mmwave
- Gym package: https://github.com/wineslab/ns-o-ran-gym
- Upstream E2 module: https://github.com/o-ran-sc/sim-ns3-o-ran-e2

Native ns-3 includes GPLv2 licensed components; respect corresponding
license terms if redistributing compiled Docker images.
