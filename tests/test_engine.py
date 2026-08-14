from __future__ import annotations

from agentic_ran.domain import NetworkObservation
from agentic_ran.service import PolicyService


def observation(**overrides) -> NetworkObservation:
    base = {
        "cell_id": "test-cell",
        "slice_type": "eMBB",
        "prb_utilization": 0.62,
        "active_users": 160,
        "downlink_mbps": 120.0,
        "uplink_mbps": 18.0,
        "latency_ms": 31.0,
        "jitter_ms": 5.0,
        "packet_loss_pct": 0.4,
        "throughput_demand_mbps": 145.0,
        "energy_load": 0.66,
        "handover_failure_pct": 0.8,
        "rsrp_dbm": -94.0,
        "sinr_db": 16.0,
    }
    base.update(overrides)
    return NetworkObservation(**base)


def test_decision_is_safe_and_explainable(trained_service: PolicyService) -> None:
    decision = trained_service.decide(observation())
    selected = next(item for item in decision.candidates if item.policy == decision.selected_policy)
    assert selected.safe
    assert decision.selected_policy in trained_service.policies
    assert decision.explanation
    assert decision.trace.selected_policy == decision.selected_policy
    assert decision.expected_kpis


def test_urllc_blocks_energy_saver(trained_service: PolicyService) -> None:
    decision = trained_service.decide(
        observation(
            slice_type="URLLC",
            prb_utilization=0.34,
            latency_ms=18.0,
            packet_loss_pct=0.25,
            throughput_demand_mbps=44.0,
            downlink_mbps=48.0,
            energy_load=0.35,
        )
    )
    energy = [candidate for candidate in decision.candidates if candidate.policy == "energy_saver"]
    if energy:
        assert not energy[0].safe
        assert any("URLLC" in reason for reason in energy[0].safety_reasons)
    assert decision.selected_policy != "energy_saver"


def test_critical_congestion_rejects_throughput_boost(trained_service: PolicyService) -> None:
    decision = trained_service.decide(
        observation(
            prb_utilization=1.05,
            packet_loss_pct=4.2,
            latency_ms=76.0,
            downlink_mbps=70.0,
            throughput_demand_mbps=260.0,
        )
    )
    boost = [candidate for candidate in decision.candidates if candidate.policy == "throughput_boost"]
    if boost:
        assert not boost[0].safe
    assert decision.selected_policy != "throughput_boost"


def test_weak_radio_adds_coverage_candidate(trained_service: PolicyService) -> None:
    decision = trained_service.decide(
        observation(rsrp_dbm=-121.0, sinr_db=-2.0, handover_failure_pct=7.2, packet_loss_pct=2.0)
    )
    assert any(candidate.policy == "coverage_recovery" for candidate in decision.candidates)
    assert decision.selected_policy == "coverage_recovery"


def test_decision_is_deterministic_for_same_model_and_observation(trained_service: PolicyService) -> None:
    sample = observation(slice_type="mMTC", active_users=960, prb_utilization=0.78, packet_loss_pct=1.5)
    first = trained_service.decide(sample)
    second = trained_service.decide(sample)
    assert first.selected_policy == second.selected_policy
    assert first.expected_kpis == second.expected_kpis


def test_no_candidate_can_be_force_approved_after_hard_sla_rejection(trained_service: PolicyService) -> None:
    decision = trained_service.decide(
        observation(
            slice_type="URLLC",
            prb_utilization=0.80,
            latency_ms=23.8,
            packet_loss_pct=0.93,
            throughput_demand_mbps=50.4,
            downlink_mbps=44.7,
            energy_load=0.74,
            sinr_db=12.0,
        ),
        intent="urllc-strict",
    )
    selected = next(item for item in decision.candidates if item.policy == decision.selected_policy)

    assert decision.selected_policy == "balanced"
    assert not selected.safe
    assert not decision.approved_for_execution
    assert decision.trace.pareto_policies == []
    assert decision.trace.rejected_policies.count("balanced") == 1
    assert "non-executable" in decision.explanation
