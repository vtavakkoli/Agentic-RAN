from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

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
    "latency_ms",
    "jitter_ms",
    "payload_bytes",
    "slicing_enabled",
    "slice_id",
    "slice_prb",
    "scheduling_policy",
]


@dataclass(frozen=True)
class Scenario:
    name: str
    kind: str
    features: int


SCENARIOS: Dict[str, Scenario] = {
    "lightweight-32": Scenario("lightweight-32", "ridge", 10),
    "lightweight-64": Scenario("lightweight-64", "ridge", 12),
    "balanced-small": Scenario("balanced-small", "hgb", 14),
    "balanced-medium": Scenario("balanced-medium", "hgb", 16),
    "deep-performance": Scenario("deep-performance", "hgb", 18),
    "ultra-performance": Scenario("ultra-performance", "hgb", 18),
    "attention-baseline": Scenario("attention-baseline", "ridge", 15),
    "liquid-baseline": Scenario("liquid-baseline", "ridge", 11),
    "xlstm-baseline": Scenario("xlstm-baseline", "ridge", 13),
}


def get_feature_columns(feature_count: int, include_categoricals: bool = True) -> List[str]:
    if feature_count <= 0:
        raise ValueError("feature_count must be > 0")
    cols = FEATURE_ORDER[: min(feature_count, len(FEATURE_ORDER))]
    if not include_categoricals:
        cols = [c for c in cols if c != "scheduling_policy"]
    return cols


def supported_scenarios() -> List[str]:
    return list(SCENARIOS.keys())
