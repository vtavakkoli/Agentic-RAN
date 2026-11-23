#Requires -Version 5.1
<#
.SYNOPSIS
  Reproducible Docker research campaign: train, validate, test, simulate,
  and generate an IEEE Access evidence audit. Never certifies NR PRB actuation.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\scripts\run_ieee_access_complete.ps1 -Preset research
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\scripts\run_ieee_access_complete.ps1 -Preset smoke -SkipOffline -SkipNative
.NOTES
  All runs get a unique results/ieee-access-*/ directory. Builds may download
  public COMMAG data and compile large C++/PyTorch dependencies; ensure adequate
  Docker disk/CPU/memory, network access, and license compliance.
#>
[CmdletBinding()]
param(
    [ValidateSet('smoke', 'research')][string]$Preset = 'research',
    [int]$TrainingRuns = 10,
    [int]$TestSeeds = 40,
    [int]$PPOUpdates = 48,
    [int]$LoggedEpisodes = 120,
    [int]$ValidationEpisodes = 24,
    [int]$Steps = 32,
    [int]$WspRounds = 3,
    [int]$CriticIterations = 5,
    [int]$EpisodesPerUpdate = 12,
    [int]$BuildJobs = 4,
    [switch]$SkipOffline,
    [switch]$SkipNative,
    [switch]$ExternalPhysical,
    [switch]$SelfTest,
    [string]$RunId = ''
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $root

function Assert-NativeExit([string]$What) {
    if ($LASTEXITCODE -ne 0) { throw "$What failed with exit code $LASTEXITCODE" }
}
function Invoke-Verified([string]$Name, [string[]]$Command) {
    Write-Host ("[IEEE Access] " + $Name + ' :: ' + ($Command -join ' ')) -ForegroundColor Cyan
    & docker @Command
    Assert-NativeExit $Name
}
if ($Preset -eq 'research') {
    if ($TrainingRuns -lt 8 -or $TestSeeds -lt 20 -or $Steps -lt 20 -or
        $ValidationEpisodes -lt 12 -or $LoggedEpisodes -lt 72) {
        throw 'Research preset requires >=8 training runs, >=20 test seeds, >=20 steps, >=12 validation episodes and >=72 logged episodes.'
    }
}
if ($BuildJobs -lt 1 -or $BuildJobs -gt 32) { throw 'BuildJobs must be between 1 and 32.' }
if ($RunId -eq '') {
    $RunId = 'ieee-access-' + (Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmss') + '-' + ([guid]::NewGuid().ToString('N').Substring(0, 6))
}
if ($RunId -notmatch '^ieee-access-[a-zA-Z0-9-]+$' -or $RunId.Contains('..')) {
    throw 'RunId must match ieee-access-[a-zA-Z0-9-]+ and must not contain traversal.'
}
$runRoot = Join-Path 'results' $RunId
if (Test-Path $runRoot) { throw "Refusing to overwrite an existing run: $runRoot" }
New-Item -ItemType Directory -Force -Path $runRoot, 'artifacts', 'data/raw/commag' | Out-Null
# Host bind mounts must be writable by the container user. On Linux GitHub
# runners the host uid is commonly 1001 while Compose's fallback is 1000.
# Respect explicitly configured LOCAL_UID/LOCAL_GID; do not guess on Windows.
$linuxPlatform = Get-Variable -Name IsLinux -ErrorAction SilentlyContinue
$runningOnLinux = ($null -ne $linuxPlatform -and $linuxPlatform.Value -eq $true)
$hostId = if ($runningOnLinux) {
    Get-Command id -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
} else { $null }
if ($null -ne $hostId -and [string]::IsNullOrWhiteSpace($env:LOCAL_UID) -and
    [string]::IsNullOrWhiteSpace($env:LOCAL_GID)) {
    $hostUid = ((& $hostId.Source -u) | Select-Object -First 1).Trim()
    $hostGid = ((& $hostId.Source -g) | Select-Object -First 1).Trim()
    if ($hostUid -match '^\d+$' -and $hostGid -match '^\d+$') {
        $env:LOCAL_UID = $hostUid
        $env:LOCAL_GID = $hostGid
        Write-Host "Docker bind-mount uid:gid = ${hostUid}:${hostGid}"
    }
}
$manifestPath = Join-Path $runRoot 'run_manifest.json'
$failureMessage = $null
$runStarted = (Get-Date).ToUniversalTime().ToString('o')
# Hashtable is a PowerShell Constrained Language Mode core type; ordered
# dictionaries are not and their .Contains() method fails under WDAC/AppLocker.
$stages = @{}
$includeOffline = -not $SkipOffline
$includeNative = -not $SkipNative

$gitSha = 'unavailable'
try {
    $gitSha = ((& git rev-parse HEAD) | Select-Object -First 1).Trim()
    if ($LASTEXITCODE -ne 0) { $gitSha = 'unavailable' }
} catch { $gitSha = 'unavailable' }

function Save-RunManifest {
    $manifest = [ordered]@{
        schema = 'agentic-ran-ieee-access-run-v1'
        run_id = $RunId
        git_sha = $gitSha
        started_utc = $runStarted
        updated_utc = (Get-Date).ToUniversalTime().ToString('o')
        options = [ordered]@{
            preset = $Preset
            training_runs = $TrainingRuns
            test_seeds = $TestSeeds
            ppo_updates = $PPOUpdates
            logged_episodes = $LoggedEpisodes
            validation_episodes = $ValidationEpisodes
            steps = $Steps
            include_offline = $includeOffline
            include_native = $includeNative
            include_physical = [bool]$ExternalPhysical
        }
        verified_dependency_pins = [ordered]@{
            ns3_release = '3.49'
            ns3_sha = '418210e8914e9f6bcb5cddbcc94a2cecb7853b68'
            ns3_gym_sha = '21e68565a6beaf13589fd19778b5332f4d0b29e0'
            native_python = '3.13'
        }
        stages = $stages
        failure = $failureMessage
    }
    # Use native PowerShell cmdlets: corporate WDAC/AppLocker Constrained
    # Language Mode blocks New-Object UTF8Encoding and System.IO.File calls.
    # Windows PowerShell 5.1 writes UTF-8 with a BOM; the Python report
    # deliberately reads JSON as utf-8-sig to handle 5.1 and PowerShell 7.
    $manifest | ConvertTo-Json -Depth 24 |
        Set-Content -LiteralPath (Join-Path $root $manifestPath) -Encoding UTF8
}
function Invoke-Stage([string]$Stage, [scriptblock]$Body) {
    # Avoid .Contains() on a non-core OrderedDictionary in constrained mode.
    if ($null -ne $stages[$Stage]) { throw "Duplicate experiment stage: $Stage" }
    $stages[$Stage] = [ordered]@{ status = 'running'; started_utc = (Get-Date).ToUniversalTime().ToString('o') }
    Save-RunManifest
    try {
        & $Body
        $stages[$Stage]['status'] = 'passed'
    } catch {
        $stages[$Stage]['status'] = 'failed'
        $stages[$Stage]['error'] = $_.Exception.Message
        throw
    } finally {
        $stages[$Stage]['completed_utc'] = (Get-Date).ToUniversalTime().ToString('o')
        Save-RunManifest
    }
}

# Exercise stage registration and BOM-compatible manifest serialization without
# Docker. Intended for restricted-corporate PowerShell regression tests.
if ($SelfTest) {
    Invoke-Stage 'constrained_language_manifest_smoke' {
        Write-Host ("PowerShell language mode: " + $ExecutionContext.SessionState.LanguageMode)
    }
    $check = Get-Content -LiteralPath (Join-Path $root $manifestPath) -Raw |
        ConvertFrom-Json
    if ($check.stages.constrained_language_manifest_smoke.status -ne 'passed') {
        throw 'Manifest smoke test failed to record a passed stage.'
    }
    Write-Host 'Constrained-language manifest self-test passed.'
    return
}

$env:RAN_GYM_TRAINING_RUNS = [string]$TrainingRuns
$env:RAN_GYM_TEST_SEEDS = [string]$TestSeeds
$env:RAN_GYM_PPO_UPDATES = [string]$PPOUpdates
$env:RAN_GYM_PPO_EPISODES_PER_UPDATE = [string]$EpisodesPerUpdate
$env:RAN_GYM_LOGGED_EPISODES = [string]$LoggedEpisodes
$env:RAN_GYM_VALIDATION_EPISODES = [string]$ValidationEpisodes
$env:RAN_GYM_STEPS = [string]$Steps
$env:RAN_GYM_WSP_ROUNDS = [string]$WspRounds
$env:RAN_GYM_CRITIC_ITERATIONS = [string]$CriticIterations
$env:RAN_GYM_CAMPAIGN_NAME = "$RunId/campaign"
$env:RAN_GYM_SMOKE_NAME = "$RunId/smoke"
$env:MODERN_NS3_BUILD_JOBS = [string]$BuildJobs
$env:AGENTIC_RAN_NATIVE_REPORT_PATH = "/workspace/results/$RunId/native/real_closed_loop.json"
$env:AGENTIC_RAN_PUBLICATION_OUTPUT = "/workspace/results/$RunId/offline"
$env:AGENTIC_RAN_PULSE_OUTPUT = "/workspace/results/$RunId/pulse"
$env:AGENTIC_RAN_PUBLICATION_SEED = '2026'
# Opt into documented experimental offline ablations; not physical proof.
$env:AGENTIC_RAN_EXPERIMENTAL_SUPPORT_GUARD = '1'
$env:AGENTIC_RAN_EVAL_MATCHED_WRAPPERS = '1'
$env:AGENTIC_RAN_EVAL_CONFORMAL_RERANK = '1'
$env:AGENTIC_RAN_EVAL_BEHAVIOR_VARIANTS = '1'
$env:AGENTIC_RAN_EVAL_SAFE_IMPROVEMENT = '1'
$env:AGENTIC_RAN_EVAL_ADAPTIVE_HYBRID = '1'
# This requires real independently confirmed BWP/PHY capacities, not example data.
$env:AGENTIC_RAN_EVAL_JOINT_SLICE_ADVANTAGE = '0'

$gymCompose = 'docker-compose.ran-gym.yml'
$pubCompose = 'docker-compose.publication.yml'
try {
    Invoke-Stage 'docker_preflight' {
        if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw 'Docker CLI not available.' }
        Invoke-Verified 'Docker engine' @('info', '--format', '{{.NCPU}} CPUs; {{.ServerVersion}} engine')
        Invoke-Verified 'Gym Compose validates' @('compose', '-f', $gymCompose, '--profile', 'native', '--profile', 'external', 'config', '--quiet')
        if ($includeOffline) { Invoke-Verified 'Publication Compose validates' @('compose', '-f', $pubCompose, 'config', '--quiet') }
    }
    Invoke-Stage 'build_and_regression_tests' {
        $targets = @('ran-gym-unit', 'ran-gym-campaign-test', 'ran-gym-campaign-smoke', 'ran-gym-final-test')
        if ($Preset -eq 'research') { $targets += 'ran-gym-campaign' }
        Invoke-Verified 'Build Python/PyTorch Gym study' (@('compose', '-f', $gymCompose, 'build') + $targets)
        foreach ($service in @('ran-gym-unit', 'ran-gym-campaign-test', 'ran-gym-final-test')) {
            Invoke-Verified "Regression $service" @('compose', '-f', $gymCompose, 'run', '--rm', $service)
        }
    }
    Invoke-Stage 'independent_training_heldout_test' {
        if ($Preset -eq 'research') {
            Invoke-Verified 'Independent training/validation/held-out campaign' @('compose', '-f', $gymCompose, 'run', '--rm', 'ran-gym-campaign')
        } else {
            # Same output basename, explicit "smoke" protocol prevents paper claims.
            $env:RAN_GYM_SMOKE_NAME = "$RunId/campaign"
            Invoke-Verified 'Small TRAIN/TEST smoke, not manuscript evidence' @('compose', '-f', $gymCompose, 'run', '--rm', 'ran-gym-campaign-smoke')
        }
    }
    if ($includeOffline) {
        Invoke-Stage 'offline_commag_train_and_test' {
            Invoke-Verified 'Full COMMAG download/prepare/train/evaluate' @('compose', '-f', $pubCompose,
                'up', '--build', '--force-recreate', '--exit-code-from', 'publication-test', 'publication-test')
        }
        Invoke-Stage 'pulse_logged_shadow' {
            Invoke-Verified 'PULSE descriptive logged shadow (not actuation)' @('compose', '-f', $pubCompose,
                'run', '--rm', '--no-deps', 'pulse-commag-shadow')
        }
    }
    if ($includeNative) {
        Invoke-Stage 'native_ns3_build_and_real_udp_test' {
            New-Item -ItemType Directory -Force -Path (Join-Path $runRoot 'native') | Out-Null
            Invoke-Verified 'Compile actual ns-3.49 C++23 Gym adapter' @('compose', '-f', $gymCompose,
                '--profile', 'native', 'build', 'ran-gym-modern-ns3-smoke')
            Invoke-Verified 'Real native UDP simulator episodes and action receipts' @('compose', '-f', $gymCompose,
                '--profile', 'native', 'run', '--rm', 'ran-gym-modern-ns3-smoke')
        }
    }
    if ($ExternalPhysical) {
        Invoke-Stage 'external_prb_actuation_gate' {
            if (-not $includeOffline) { throw 'External physical gate requires offline frozen COMMAG models.' }
            if (-not (Test-Path 'configs/ran_gym_active.json')) { throw 'Missing validated configs/ran_gym_active.json.' }
            if (-not (Test-Path 'external/site-adapters/site_adapter.py')) { throw 'Missing independently validated real PRB/scheduler adapter.' }
            New-Item -ItemType Directory -Force -Path (Join-Path $runRoot 'physical') | Out-Null
            Invoke-Verified 'External physical PRB scenario (not UDP)' @('compose', '-f', $gymCompose,
                '--profile', 'external', 'run', '--rm', 'ran-gym-active',
                '--config', '/workspace/configs/ran_gym_active.json',
                '--output', "/workspace/results/$RunId/physical")
            Invoke-Verified 'Strict active publication evidence gate' @('compose', '-f', $gymCompose,
                'run', '--rm', 'publication-final-gate',
                '--offline', "/workspace/results/$RunId/offline/publication_summary.json",
                '--active', "/workspace/results/$RunId/physical/ran_gym_summary.json",
                '--output', "/workspace/results/$RunId/physical/final_evidence_gate.json")
        }
    }
} catch {
    $failureMessage = $_.Exception.Message
    Write-Warning "Research stage failed: $failureMessage"
} finally {
    Save-RunManifest
    try {
        Invoke-Verified 'Build report Python runtime' @('compose', '-f', $gymCompose, 'build', 'ran-gym-unit')
        Invoke-Verified 'Generate comprehensive evidence reports and tables' @('compose', '-f', $gymCompose,
            'run', '--rm', '--entrypoint', 'python', 'ran-gym-unit',
            '-m', 'agentic_ran.research_ieee_access_report',
            '--run-root', "/workspace/results/$RunId",
            '--manifest', "/workspace/results/$RunId/run_manifest.json",
            '--output', "/workspace/results/$RunId/final-report")
    } catch {
        if (-not $failureMessage) { $failureMessage = "Report generation failed: $($_.Exception.Message)" }
        Write-Warning "Final report generation error: $($_.Exception.Message)"
    }
    Save-RunManifest
}
Write-Host "Research output: $runRoot" -ForegroundColor Green
Write-Host "HTML report: $(Join-Path $runRoot 'final-report/ieee_access_report.html')" -ForegroundColor Green
Write-Host "Evidence JSON: $(Join-Path $runRoot 'final-report/ieee_access_evidence.json')" -ForegroundColor Green
$evidencePath = Join-Path $runRoot 'final-report/ieee_access_evidence.json'
if (Test-Path $evidencePath) {
    $assessment = Get-Content $evidencePath -Raw | ConvertFrom-Json
    if ($assessment.research_package_complete -ne $true) {
        Write-Warning "Scientific evidence incomplete or smoke-only: $($assessment.experiment_status). Read the report before paper claims."
    }
}
if ($failureMessage) {
    Write-Error "Study incomplete; inspect run_manifest.json and generated report. $failureMessage"
    exit 1
}
Write-Host 'Workflow completed. Read evidence gates before making any IEEE Access scientific claims.' -ForegroundColor Green