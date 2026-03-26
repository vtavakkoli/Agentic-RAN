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
    seq_len: int | None = None
    architecture: str | None = None


SCENARIOS: Dict[str, Scenario] = {
    "lightweight-32": Scenario("lightweight-32", "tabular", 10),
    "lightweight-64": Scenario("lightweight-64", "tabular", 12),
    "balanced-small": Scenario("balanced-small", "tabular", 14),
    "balanced-medium": Scenario("balanced-medium", "tabular", 16),
    "deep-performance": Scenario("deep-performance", "temporal", 17, seq_len=16, architecture="lstm"),
    "ultra-performance": Scenario("ultra-performance", "temporal", 18, seq_len=24, architecture="lstm"),
    "attention-baseline": Scenario("attention-baseline", "temporal", 15, seq_len=16, architecture="attention"),
    "liquid-baseline": Scenario("liquid-baseline", "temporal", 11, seq_len=12, architecture="liquid"),
    "xlstm-baseline": Scenario("xlstm-baseline", "temporal", 13, seq_len=24, architecture="lstm"),
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
