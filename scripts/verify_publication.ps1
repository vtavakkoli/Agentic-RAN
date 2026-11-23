# End-to-end verification. Runs actual externally configured RAN Gym only with -External.
param(
    [switch]$External,
    [switch]$RunJointSlice,
    [switch]$FullResearch,
    [switch]$NativeGym
)
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))
function Invoke-Docker([string[]]$Arguments) {
    Write-Host ("docker " + ($Arguments -join " ")) -ForegroundColor Cyan
    & docker @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Docker step failed with exit code $LASTEXITCODE" }
}
try {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "Docker CLI is not installed"
    }
    Invoke-Docker -Arguments @("info", "--format", "{{.NCPU}} available CPUs")
    if ($FullResearch) {
        $env:AGENTIC_RAN_EXPERIMENTAL_SUPPORT_GUARD = "1"
        $env:AGENTIC_RAN_EVAL_MATCHED_WRAPPERS = "1"
        $env:AGENTIC_RAN_EVAL_CONFORMAL_RERANK = "1"
        $env:AGENTIC_RAN_EVAL_BEHAVIOR_VARIANTS = "1"
        $env:AGENTIC_RAN_EVAL_SAFE_IMPROVEMENT = "1"
        $env:AGENTIC_RAN_EVAL_ADAPTIVE_HYBRID = "1"
    } else {
        $env:AGENTIC_RAN_EXPERIMENTAL_SUPPORT_GUARD = "0"
        $env:AGENTIC_RAN_EVAL_MATCHED_WRAPPERS = "0"
        $env:AGENTIC_RAN_EVAL_CONFORMAL_RERANK = "0"
        $env:AGENTIC_RAN_EVAL_BEHAVIOR_VARIANTS = "0"
        $env:AGENTIC_RAN_EVAL_SAFE_IMPROVEMENT = "0"
        $env:AGENTIC_RAN_EVAL_ADAPTIVE_HYBRID = "0"
    }
    if ($RunJointSlice) {
        if (-not (Test-Path "configs/jagc_phy_bwp.json")) {
            throw "JAGC requires verified physical BWP capacities in configs/jagc_phy_bwp.json"
        }
        $env:AGENTIC_RAN_EVAL_JOINT_SLICE_ADVANTAGE = "1"
    } else {
        $env:AGENTIC_RAN_EVAL_JOINT_SLICE_ADVANTAGE = "0"
    }
    # 1. Existing full COMMAG benchmark; dependency downloads/caches data.
    Invoke-Docker -Arguments @("compose", "-f", "docker-compose.publication.yml",
        "up", "--build", "--exit-code-from", "publication-test", "publication-test")
    # 2. PULSE shadow replay (descriptive, not actuated).
    Invoke-Docker -Arguments @("compose", "-f", "docker-compose.publication.yml",
        "run", "--rm", "--no-deps", "pulse-commag-shadow")
    # 3. New Gym benchmark regression suite and reproducible toy smoke.
    Invoke-Docker -Arguments @("compose", "-f", "docker-compose.ran-gym.yml", "build", "ran-gym-unit", "ran-gym-smoke")
    Invoke-Docker -Arguments @("compose", "-f", "docker-compose.ran-gym.yml", "run", "--rm", "ran-gym-unit")
    Invoke-Docker -Arguments @("compose", "-f", "docker-compose.ran-gym.yml", "run", "--rm", "ran-gym-smoke")
    # 4. All six compatible final learned algorithms, matched Gym action-dependent rollouts.
    # Explicitly TOY ONLY: not a native NR scheduler/PRB experiment.
    Invoke-Docker -Arguments @("compose", "-f", "docker-compose.ran-gym.yml", "run", "--rm", "ran-gym-final-test")
    Invoke-Docker -Arguments @("compose", "-f", "docker-compose.ran-gym.yml", "run", "--rm", "ran-gym-final-algorithms")
    if ($NativeGym) {
        # Verified native ns-3.49 UDP example (action is packet count, NOT PRB).
        # Never count this separate physical packet-delivery check as a
        # COMMAG scheduler performance experiment.
        Invoke-Docker -Arguments @("compose", "-f", "docker-compose.ran-gym.yml", "--profile", "native",
            "up", "--build", "--exit-code-from", "ran-gym-modern-ns3-smoke", "ran-gym-modern-ns3-smoke")
    }
    if (-not $External) {
        Write-Host "Offline pipeline and matched final-algorithm TOY Gym checks complete. No native PRB controller result was tested." -ForegroundColor Yellow
        Write-Host "For UDP native ns-3.49 verification use -NativeGym; PRB/COMMAG needs a separate validated external engine (-External)." -ForegroundColor Yellow
        exit 0
    }
    # 4. Real PRB/COMMAG requires site-supplied validated native RAN model/action adapter.
    if (-not (Test-Path "configs/ran_gym_active.json")) {
        throw "Create configs/ran_gym_active.json with a separately validated PRB-compatible simulator and model/adapter; generic ns-3 UDP is insufficient"
    }
    if (-not (Test-Path "external/site-adapters")) {
        throw "Install the real policy adapter in external/site-adapters"
    }
    Invoke-Docker -Arguments @("compose", "-f", "docker-compose.ran-gym.yml", "--profile", "external",
        "up", "--build", "--exit-code-from", "ran-gym-active", "ran-gym-active")
    # 5. FAILS unless held-out realized candidate gains, receipts and service KPIs are verified.
    Invoke-Docker -Arguments @("compose", "-f", "docker-compose.ran-gym.yml",
        "run", "--rm", "publication-final-gate")
    Write-Host "Artifact evidence gate passed; manual methodology and manuscript review remain." -ForegroundColor Green
} catch {
    Write-Error $_
    exit 1
}
