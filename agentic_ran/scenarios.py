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
    loss: str = "mse"
    peak_weight: float = 2.0


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
    "residual-mlp-128": ScenarioConfig("residual-mlp-128", "residual_mlp", "pytorch", "residual-mlp", 1, 128, 4, 0.15, 64, 8e-4, "weighted_huber", 2.0),
    "residual-mlp-256": ScenarioConfig("residual-mlp-256", "residual_mlp", "pytorch", "residual-mlp", 1, 256, 6, 0.2, 64, 6e-4, "weighted_huber", 2.0),
    "residual-tcn-16": ScenarioConfig("residual-tcn-16", "residual_tcn", "pytorch", "residual-temporal", 16, 128, 4, 0.15, 64, 8e-4, "weighted_huber", 2.0),
    "residual-tcn-32": ScenarioConfig("residual-tcn-32", "residual_tcn", "pytorch", "residual-temporal", 32, 128, 5, 0.15, 64, 7e-4, "weighted_huber", 2.0),
    "residual-liquid-tcn-16": ScenarioConfig("residual-liquid-tcn-16", "residual_liquid_tcn", "pytorch", "residual-liquid-temporal", 16, 128, 4, 0.15, 64, 8e-4, "weighted_huber", 2.0),
    "residual-liquid-tcn-32": ScenarioConfig("residual-liquid-tcn-32", "residual_liquid_tcn", "pytorch", "residual-liquid-temporal", 32, 128, 5, 0.15, 64, 7e-4, "weighted_huber", 2.0),
    "without_time_features": ScenarioConfig("without_time_features", "mlp", "pytorch", "ablation", 1, 128, 3, 0.2, 64, 8e-4),
    "with_time_features": ScenarioConfig("with_time_features", "mlp", "pytorch", "ablation", 1, 128, 3, 0.2, 64, 8e-4),
    "with_time_and_traffic_features": ScenarioConfig("with_time_and_traffic_features", "residual_mlp", "pytorch", "ablation", 1, 128, 4, 0.2, 64, 8e-4),
    "with_agentic_policy_features": ScenarioConfig("with_agentic_policy_features", "residual_mlp", "pytorch", "ablation", 1, 128, 4, 0.2, 64, 8e-4),
    "agentic_residual_mlp": ScenarioConfig("agentic_residual_mlp", "agentic_residual_mlp", "pytorch", "agentic", 1, 128, 5, 0.2, 64, 8e-4, "weighted_huber", 2.0),
    "agentic_liquid_residual": ScenarioConfig("agentic_liquid_residual", "agentic_liquid_residual", "pytorch", "agentic", 16, 128, 3, 0.2, 64, 8e-4, "weighted_huber", 2.0),
    "agentic_sequence_attention": ScenarioConfig("agentic_sequence_attention", "agentic_sequence_attention", "pytorch", "agentic", 16, 128, 3, 0.2, 64, 8e-4, "weighted_huber", 2.0),
}
