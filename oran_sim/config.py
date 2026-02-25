from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from oran_sim.seed import SEED

FEATURE_ORDER = [
    "dl_cqi",
    "ul_sinr",
    "sum_granted_prbs",
    "sum_requested_prbs",
    "dl_buffer_bytes",
    "ul_buffer_bytes",
    "num_ues",
    "dl_mcs",
    "ul_mcs",
    "tx_errors_dl_pct",
    "rx_errors_ul_pct",
    "slicing_enabled",
    "slice_id",
    "slice_prb",
    "scheduling_policy",
    "latency_ms",
    "jitter_ms",
    "payload_bytes",
]


DEFAULT_FEATURE_COUNT = len(FEATURE_ORDER)


def get_feature_columns(feature_count: int) -> list[str]:
    if feature_count <= 0:
        raise ValueError("feature_count must be > 0")
    if feature_count > len(FEATURE_ORDER):
        raise ValueError(f"feature_count={feature_count} exceeds available features ({len(FEATURE_ORDER)})")
    return FEATURE_ORDER[:feature_count]


@dataclass(frozen=True)
class Scenario:
    name: str
    model: str
    feature_count: int


SCENARIOS: Dict[str, Scenario] = {
    f"scenario_{idx}": Scenario(
        name=f"scenario_{idx}",
        model="ridge" if idx <= 4 else "hgb",
        feature_count=min(6 + idx, len(FEATURE_ORDER)),
    )
    for idx in range(1, 9)
}


def supported_scenarios() -> List[str]:
    return list(SCENARIOS.keys())
