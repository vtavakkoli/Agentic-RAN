from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    model_type: str
    backend: str
    logical_profile: str
    sequence_length: int
    hidden_size: int
    num_layers: int
    dropout: float
    batch_size: int
    learning_rate: float


SCENARIOS: dict[str, ScenarioConfig] = {
    "lightweight-32": ScenarioConfig("lightweight-32", "mlp", "pytorch", "lightweight", 1, 32, 2, 0.1, 64, 1e-3),
    "lightweight-64": ScenarioConfig("lightweight-64", "mlp", "pytorch", "lightweight", 1, 64, 2, 0.1, 64, 1e-3),
    "balanced-small": ScenarioConfig("balanced-small", "mlp", "pytorch", "balanced", 1, 96, 3, 0.15, 64, 8e-4),
    "balanced-medium": ScenarioConfig("balanced-medium", "mlp", "pytorch", "balanced", 1, 128, 3, 0.2, 64, 8e-4),
    "deep-performance": ScenarioConfig("deep-performance", "mlp", "pytorch", "deep", 1, 256, 5, 0.2, 64, 6e-4),
    "ultra-performance": ScenarioConfig("ultra-performance", "mlp", "pytorch", "ultra", 1, 384, 6, 0.25, 64, 5e-4),
    "attention-baseline": ScenarioConfig("attention-baseline", "attention", "pytorch", "sequence-attention", 16, 96, 2, 0.15, 64, 1e-3),
    "liquid-baseline": ScenarioConfig("liquid-baseline", "liquid", "pytorch", "liquid-dynamics", 16, 96, 2, 0.1, 64, 1e-3),
    "xlstm-baseline": ScenarioConfig("xlstm-baseline", "xlstm", "pytorch", "xlstm-sequence", 24, 128, 2, 0.2, 64, 8e-4),
}
