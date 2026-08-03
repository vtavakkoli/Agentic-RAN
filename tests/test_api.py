from __future__ import annotations

from fastapi.testclient import TestClient

from agentic_ran.api import create_app
from agentic_ran.service import PolicyService


SAMPLE = {
    "cell_id": "api-cell",
    "slice_type": "URLLC",
    "prb_utilization": 0.91,
    "active_users": 80,
    "downlink_mbps": 45.0,
    "uplink_mbps": 10.0,
    "latency_ms": 23.0,
    "jitter_ms": 6.0,
    "packet_loss_pct": 0.7,
    "throughput_demand_mbps": 60.0,
    "energy_load": 0.84,
    "handover_failure_pct": 1.0,
    "rsrp_dbm": -100.0,
    "sinr_db": 9.0,
}


def test_health_ready_and_policy_catalog(trained_service: PolicyService) -> None:
    client = TestClient(create_app(trained_service))
    assert client.get("/healthz").status_code == 200
    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["policy_count"] == 7
    policies = client.get("/v1/policies").json()["policies"]
    assert {item["name"] for item in policies} >= {"balanced", "latency_guard"}


def test_single_and_batch_decisions(trained_service: PolicyService) -> None:
    client = TestClient(create_app(trained_service))
    response = client.post("/v1/decisions", json=SAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert body["selected_policy"]
    assert body["candidates"]

    batch = client.post("/v1/decisions/batch", json={"observations": [SAMPLE, SAMPLE]})
    assert batch.status_code == 200
    assert len(batch.json()) == 2


def test_invalid_observation_is_rejected(trained_service: PolicyService) -> None:
    client = TestClient(create_app(trained_service))
    invalid = {**SAMPLE, "prb_utilization": -1}
    response = client.post("/v1/decisions", json=invalid)
    assert response.status_code == 422


def test_metrics_count_decisions(trained_service: PolicyService) -> None:
    client = TestClient(create_app(trained_service))
    client.post("/v1/decisions", json=SAMPLE)
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "agentic_ran_decisions_total 1" in metrics.text
