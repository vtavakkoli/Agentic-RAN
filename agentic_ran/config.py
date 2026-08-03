"""Runtime configuration and stable feature names."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


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
    """Environment-backed application settings."""

    dataset_path: Path = Path("data/runtime/ran_policy_sample.csv")
    model_path: Path = Path("artifacts/policy_selector.joblib")
    policy_config_path: Path = Path("configs/policies.yaml")
    results_dir: Path = Path("results")
    top_k: int = 4
    random_seed: int = 42
    auto_bootstrap: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            dataset_path=Path(os.getenv("AGENTIC_RAN_DATASET", "data/runtime/ran_policy_sample.csv")),
            model_path=Path(os.getenv("AGENTIC_RAN_MODEL", "artifacts/policy_selector.joblib")),
            policy_config_path=Path(os.getenv("AGENTIC_RAN_POLICIES", "configs/policies.yaml")),
            results_dir=Path(os.getenv("AGENTIC_RAN_RESULTS", "results")),
            top_k=max(1, min(7, int(os.getenv("AGENTIC_RAN_TOP_K", "4")))),
            random_seed=int(os.getenv("AGENTIC_RAN_SEED", "42")),
            auto_bootstrap=os.getenv("AGENTIC_RAN_AUTO_BOOTSTRAP", "1") not in {"0", "false", "False"},
        )
