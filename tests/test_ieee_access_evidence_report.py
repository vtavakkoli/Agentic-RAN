"""Unit tests for the reporting gate. Fixtures are NOT simulator evidence.

Native integration truth is enforced separately by GitHub Actions' compiled
ns-3 workflow; these tests only validate parsing and fail-closed semantics.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from agentic_ran.research_ieee_access_report import inspect_native, produce


def _synthetic_protocol_fixture():
    """Manufactured protocol-only fixture, NEVER evidence of a real ns-3 run."""
    def step(index, action, tx, rx):
        return {
            "step": index, "requested_action": action, "applied_action": action,
            "observation": [float(rx), float(tx), float(index+1)], "reward": float(action),
            "terminated": index == 1, "truncated": False,
            "kpis": {"rx_packets": rx, "tx_packets": tx, "rx_bytes": 256*rx,
                     "throughput_mbps": action*256*8/(0.1*1e6),
                     "applied_action": action, "action_receipt_verified": True},
        }
    trace = [step(0, 2, 2, 2), step(1, 1, 3, 3)]
    episodes = [{"seed": 2026, "trace": deepcopy(trace),
                 "final_kpis": {"rx_packets": 3, "tx_packets": 3, "rx_bytes": 768}}
                for _ in range(2)]
    return {
        "backend": "verified-ns3-3.49-udp-real", "scenario": "two-node-udp-packets",
        "ns3_sha": "418210e8914e9f6bcb5cddbcc94a2cecb7853b68",
        "ns3_gym_sha": "21e68565a6beaf13589fd19778b5332f4d0b29e0",
        "actual_native_episodes": 2, "actual_native_steps": 4,
        "commag_prb_controller_evaluated": False, "publication_superiority_ready": False,
        "episodes": episodes,
    }


def test_native_evidence_contract_accepts_consistent_fixture_but_not_nr_claim():
    result = inspect_native(_synthetic_protocol_fixture())
    assert result["passed"] is True
    assert result["episodes"][0]["rx_packets"] == 3
    assert result["deterministic_replay_observed"] is True


def test_fabricated_receipts_and_rewards_fail():
    case = _synthetic_protocol_fixture()
    case["episodes"][0]["trace"][0]["kpis"]["applied_action"] = 99
    assert inspect_native(case)["passed"] is False
    case = _synthetic_protocol_fixture()
    case["episodes"][0]["trace"][0]["reward"] = 7.0
    assert inspect_native(case)["passed"] is False


def test_unknown_simulator_provenance_and_prb_claim_fail():
    case = _synthetic_protocol_fixture()
    case["ns3_sha"] = "moving-branch"
    case["commag_prb_controller_evaluated"] = True
    assert inspect_native(case)["passed"] is False


def test_no_same_seed_replay_fails():
    case = _synthetic_protocol_fixture()
    case["episodes"][1]["seed"] = 42
    assert inspect_native(case)["passed"] is False


def test_report_missing_experiments_writes_typed_incomplete_evidence(tmp_path: Path):
    out = tmp_path / "final-report"
    manifest = {
        "run_id": "ieee-access-test", "git_sha": "a"*40,
        "options": {"include_offline": True, "include_native": True},
        "stages": {"docker_preflight": {"status": "passed"}},
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    report = produce(tmp_path, out)
    assert report["research_package_complete"] is False
    assert report["publication_ready_for_ieee_access"] is False
    assert report["native"]["passed"] is False
    assert (out / "ieee_access_report.html").stat().st_size > 0
    hashes = json.loads((out / "sha256_manifest.json").read_text())
    assert "ieee_access_evidence.json" in hashes
    assert "table_toy_comparisons.csv" in hashes
    assert "sha256_manifest.json" not in hashes


def test_native_fixture_alone_never_certifies_ran_paper(tmp_path: Path):
    (tmp_path / "native").mkdir()
    (tmp_path / "native" / "real_closed_loop.json").write_text(json.dumps(_synthetic_protocol_fixture()))
    (tmp_path / "run_manifest.json").write_text(json.dumps({
        "run_id": "ieee-access-test", "git_sha": "a"*40,
        "options": {"include_offline": False, "include_native": True},
        "stages": {"native_ns3_build_and_real_udp_test": {"status": "passed"}},
    }))
    report = produce(tmp_path, tmp_path / "output")
    assert report["native"]["passed"] is True
    assert report["research_package_complete"] is False
    assert report["publication_ready_for_ieee_access"] is False
    assert report["checks"]["real_prb_scheduler_closed_loop_gate"] is False


def test_smoke_only_never_claims_research_package_even_if_all_commands_pass(tmp_path: Path):
    from agentic_ran.research_ieee_access_report import inspect_campaign
    data = {"backend": "agentic-ran-interventional-toy-gymnasium",
            "publication_ready_for_ieee_access": False,
            "physical_network_sla_evidence": False,
            "protocol": {"preset": "smoke"},
            "training_runs": 2, "test_seeds_per_run": 4,
            "total_executed_episodes": 24, "total_executed_steps": 120,
            "comparisons": {"baseline_" + str(i): {
                "mean_reward_delta": 0.0, "train_seed_bootstrap_ci95": [0.0, 0.0],
                "holm_adjusted_p": 1.0} for i in range(5)}}
    result = inspect_campaign(data, tmp_path)
    assert result["smoke"] is True
    assert result["passed"] is False  # no genuine hash-proven evidence

def test_manifest_with_windows_powershell_51_utf8_bom(tmp_path: Path):
    """PowerShell 5.1 writes BOM with Set-Content -Encoding UTF8 in CLM."""
    from agentic_ran.research_ieee_access_report import read_json

    manifest = {"run_id": "ieee-access-bom", "stages": {
        "constrained_language_manifest_smoke": {"status": "passed"}
    }}
    path = tmp_path / "run_manifest.json"
    path.write_text("\ufeff" + json.dumps(manifest), encoding="utf-8")
    loaded, error = read_json(path)
    assert error is None
    assert loaded == manifest


def test_powershell_orchestrator_uses_only_core_manifest_apis():
    """Regression guard for the two Constrained Language Mode failures."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_ieee_access_complete.ps1"
    if not script.exists():
        # The wheel build may install only the Python package.
        return
    source = script.read_text(encoding="utf-8")
    assert "$stages = @{}" in source
    assert "if ($null -ne $stages[$Stage])" in source
    assert "Set-Content -LiteralPath" in source
    assert "ConvertTo-Json -Depth 24" in source
    assert "New-Object System.Text.UTF8Encoding" not in source
    assert "[System.IO.File]::WriteAllText" not in source
    assert "$stages.Contains(" not in source
    assert "[switch]$SelfTest" in source
