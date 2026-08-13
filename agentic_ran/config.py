"""Runtime configuration and stable feature names."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from agentic_ran.domain import ExecutionMode


NUMERIC_FEATURES = [
    "prb_utilization",
    "active_users",
    "downlink_mbps",
    "uplink_mbps",
    "latency_ms",
    "jitter_ms",
    "packet_loss_pct",
    "throughput_demand_mbps",
    "energy_load",
    "handover_failure_pct",
    "rsrp_dbm",
    "sinr_db",
]
CATEGORICAL_FEATURES = ["slice_type"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_COLUMN = "policy_label"


@dataclass(frozen=True, slots=True)
class Settings:
    dataset_path: Path = Path("data/runtime/ran_policy_sample.csv")
    model_path: Path = Path("artifacts/policy_selector.joblib")
    policy_config_path: Path = Path("configs/policies.yaml")
    results_dir: Path = Path("results")
    audit_path: Path = Path("results/audit/decisions.jsonl")
    model_registry_dir: Path = Path("artifacts/registry")
    top_k: int = 4
    random_seed: int = 42
    auto_bootstrap: bool = True
    planning_horizon: int = 3
    execution_mode: ExecutionMode = ExecutionMode.RECOMMEND
    srsran_ws_url: str = "ws://127.0.0.1:8001"
    e2_bridge_url: str = "http://127.0.0.1:8090"
    default_intent: str = "balanced"
    canary_cells: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "Settings":
        mode_value = os.getenv("AGENTIC_RAN_EXECUTION_MODE", "recommend").strip().lower()
        try:
            mode = ExecutionMode(mode_value)
        except ValueError:
            mode = ExecutionMode.RECOMMEND
        canary = tuple(filter(None, (item.strip() for item in os.getenv("AGENTIC_RAN_CANARY_CELLS", "").split(","))))
        return cls(
            dataset_path=Path(os.getenv("AGENTIC_RAN_DATASET", "data/runtime/ran_policy_sample.csv")),
            model_path=Path(os.getenv("AGENTIC_RAN_MODEL", "artifacts/policy_selector.joblib")),
            policy_config_path=Path(os.getenv("AGENTIC_RAN_POLICIES", "configs/policies.yaml")),
            results_dir=Path(os.getenv("AGENTIC_RAN_RESULTS", "results")),
            audit_path=Path(os.getenv("AGENTIC_RAN_AUDIT", "results/audit/decisions.jsonl")),
            model_registry_dir=Path(os.getenv("AGENTIC_RAN_REGISTRY", "artifacts/registry")),
            top_k=max(1, min(7, int(os.getenv("AGENTIC_RAN_TOP_K", "4")))),
            random_seed=int(os.getenv("AGENTIC_RAN_SEED", "42")),
            auto_bootstrap=os.getenv("AGENTIC_RAN_AUTO_BOOTSTRAP", "1") not in {"0", "false", "False"},
            planning_horizon=max(1, min(12, int(os.getenv("AGENTIC_RAN_PLANNING_HORIZON", "3")))),
            execution_mode=mode,
            srsran_ws_url=os.getenv("SRSRAN_WS_URL", "ws://127.0.0.1:8001"),
            e2_bridge_url=os.getenv("AGENTIC_RAN_E2_BRIDGE_URL", "http://127.0.0.1:8090"),
            default_intent=os.getenv("AGENTIC_RAN_DEFAULT_INTENT", "balanced"),
            canary_cells=canary,
        )
