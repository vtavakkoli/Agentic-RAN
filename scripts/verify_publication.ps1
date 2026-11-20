# End-to-end verification. Runs actual externally configured RAN Gym only with -External.
param(
    [switch]$External,
    [switch]$RunJointSlice
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
    if (-not $External) {
        Write-Host "Offline pipeline and toy Gym checks complete. No ns-O-RAN active result was tested." -ForegroundColor Yellow
        Write-Host "For active verification configure external ns-O-RAN and re-run with -External." -ForegroundColor Yellow
        exit 0
    }
    # 4. Actual ns-O-RAN requires site-supplied model/action adapter + compiled ns-3.
    if (-not (Test-Path "configs/ran_gym_active.json")) {
        throw "Create configs/ran_gym_active.json from the example with a real compatible model/adapter"
    }
    if (-not (Test-Path "external/site-adapters")) {
        throw "Install the real policy adapter in external/site-adapters"
    }
    if (-not $env:RAN_GYM_NS3_DIR) {
        throw "Set RAN_GYM_NS3_DIR to a Linux-compatible ns-O-RAN/ns-3 installation"
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
