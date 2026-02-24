from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

# oran_sim/config.py

# oran_sim/config.py
FEATURE_ORDER = [
    # UE radio/link quality (core)
    "rsrp",
    "dl_snr",
    "pl",
    "cfo",
    "ul_ta",
    # MCS / reliability (PHY/MAC)
    "dl_mcs",
    "ul_mcs",
    "dl_bler",
    "ul_bler",
    # Throughput / buffer (MAC)
    "dl_brate_ue",
    "ul_brate_ue",
    "ul_buff",
    # RF indicators
    "rf_o",
    "rf_u",
    "rf_l",
    # Context
    "ue_count",
]


@dataclass(frozen=True)
class Scenario:
    name: str
    kind: str
    features: int
    hidden_sizes: List[int] | None = None
    d_model: int | None = None
    nhead: int | None = None
    num_layers: int | None = None
    dim_feedforward: int | None = None
    dropout: float | None = None
    hidden_size: int | None = None
    dt: float | None = None


SCENARIOS: Dict[str, Scenario] = {
    "lightweight-32": Scenario("lightweight-32", "lstm", 6, hidden_sizes=[32]),
    "lightweight-64": Scenario("lightweight-64", "lstm", 6, hidden_sizes=[64]),
    "balanced-small": Scenario("balanced-small", "lstm", 8, hidden_sizes=[64, 32]),
    "balanced-medium": Scenario("balanced-medium", "lstm", 8, hidden_sizes=[100, 50]),
    "deep-performance": Scenario("deep-performance", "lstm", 10, hidden_sizes=[128, 100, 64]),
    "ultra-performance": Scenario("ultra-performance", "lstm", 16, hidden_sizes=[512, 256, 128]),
    "attention-baseline": Scenario(
        "attention-baseline",
        "attention",
        8,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    ),
    "liquid-baseline": Scenario("liquid-baseline", "liquid", 6, hidden_size=64, dt=0.1),
}


def supported_scenarios() -> List[str]:
    return list(SCENARIOS.keys())
