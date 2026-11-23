# Full reproducible research campaign, with optional offline and native integration.
# Uses the same six compatible toy-Gym actions; no physical NR PHY claim.
param(
    [switch]$FullOffline,
    [switch]$NativeSmoke,
    [switch]$ExternalPhysical,
    [int]$TrainingRuns = 5,
    [int]$TestSeeds = 40
)
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))
function Invoke-Docker([string[]] $Arguments) {
    Write-Host ("docker " + ($Arguments -join " ")) -ForegroundColor Cyan
    & docker @Arguments
    if ($LASTEXITCODE -ne 0) { throw ("Docker command failed (exit " + $LASTEXITCODE + ")") }
}
if ($TrainingRuns -lt 5 -or $TestSeeds -lt 20) {
    throw "Research preset requires at least five training runs and twenty test seeds per run"
}
Invoke-Docker -Arguments @("info", "--format", "Docker CPUs: {{.NCPU}}")
New-Item -ItemType Directory -Force -Path "results", "artifacts", "data/raw/commag" | Out-Null
$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMdd-HHmmss")
$env:RAN_GYM_CAMPAIGN_NAME = "ran-gym-campaign-" + $stamp
$env:RAN_GYM_SMOKE_NAME = "ran-gym-campaign-smoke-" + $stamp
$env:RAN_GYM_TRAINING_RUNS = [string]$TrainingRuns
$env:RAN_GYM_TEST_SEEDS = [string]$TestSeeds
$compose = "docker-compose.ran-gym.yml"

# Check the existing repository/unit and integration suite.
Invoke-Docker -Arguments @("compose", "-f", $compose, "build", "ran-gym-unit", "ran-gym-campaign")
Invoke-Docker -Arguments @("compose", "-f", $compose, "run", "--rm", "ran-gym-unit")
Invoke-Docker -Arguments @("compose", "-f", $compose, "run", "--rm", "ran-gym-final-test")
Invoke-Docker -Arguments @("compose", "-f", $compose, "run", "--rm", "ran-gym-campaign-test")

# Tiny end-to-end check with 2 training seeds; no manuscript performance claim.
Invoke-Docker -Arguments @("compose", "-f", $compose, "run", "--rm", "ran-gym-campaign-smoke")

# Five or more independent training seeds; strict validation/test separation;
# four scenarios, 40 held-out seeds per run, 6 controllers, 32-step episodes.
Invoke-Docker -Arguments @("compose", "-f", $compose, "run", "--rm", "ran-gym-campaign")
$output = Join-Path "results" $env:RAN_GYM_CAMPAIGN_NAME
$summary = Join-Path $output "campaign_summary.json"
if (-not (Test-Path $summary)) { throw "Missing research summary: $summary" }
$result = Get-Content $summary -Raw | ConvertFrom-Json
if ($result.publication_ready_for_ieee_access -eq $true) {
    throw "Invalid evidence: toy Gym cannot certify IEEE Access physical-RAN claims"
}
Write-Host ("Toy-Gym research report: " + (Join-Path $output "research_report.html")) -ForegroundColor Green

if ($FullOffline) {
    # Expensive independent experiment: full COMMAG publication pipeline
    # must NOT combine logged rewards with unexecuted candidate counterfactuals.
    $env:AGENTIC_RAN_EXPERIMENTAL_SUPPORT_GUARD = "1"
    $env:AGENTIC_RAN_EVAL_MATCHED_WRAPPERS = "1"
    $env:AGENTIC_RAN_EVAL_CONFORMAL_RERANK = "1"
    $env:AGENTIC_RAN_EVAL_BEHAVIOR_VARIANTS = "1"
    $env:AGENTIC_RAN_EVAL_SAFE_IMPROVEMENT = "1"
    $env:AGENTIC_RAN_EVAL_ADAPTIVE_HYBRID = "1"
    Invoke-Docker -Arguments @("compose", "-f", "docker-compose.publication.yml",
        "up", "--build", "--exit-code-from", "publication-test", "publication-test")
    Invoke-Docker -Arguments @("compose", "-f", "docker-compose.publication.yml",
        "run", "--rm", "--no-deps", "pulse-commag-shadow")
}
if ($NativeSmoke -or $ExternalPhysical) {
    # Native upstream ns-O-RAN Gym is TRAFFIC STEERING and not scheduler/PRB actuation.
    if (-not $env:NSORAN_BUILD_JOBS) { $env:NSORAN_BUILD_JOBS = "4" }
    Invoke-Docker -Arguments @("compose", "-f", $compose, "--profile", "native",
        "up", "--build", "--exit-code-from", "ran-gym-nsoran-smoke", "ran-gym-nsoran-smoke")
}
if ($ExternalPhysical) {
    if (-not (Test-Path "configs/ran_gym_active.json")) {
        throw "Physical active Gym requires configs/ran_gym_active.json with verified PRB/scheduler mapping"
    }
    if (-not (Test-Path "external/site-adapters/site_adapter.py")) {
        throw "Physical active Gym requires the real site adapter; stock handover Gym is incompatible"
    }
    Invoke-Docker -Arguments @("compose", "-f", $compose, "--profile", "external",
        "up", "--build", "--exit-code-from", "ran-gym-active", "ran-gym-active")
    if (-not $FullOffline) {
        throw "Active publication final gate needs -FullOffline and the generated publication_summary.json"
    }
    Invoke-Docker -Arguments @("compose", "-f", $compose, "run", "--rm", "publication-final-gate")
}
Write-Host "Research bundle is complete. IEEE Access readiness still requires independent physical evidence and manuscript review." -ForegroundColor Yellow
