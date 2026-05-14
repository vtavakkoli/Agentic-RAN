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


# Main study scope: only models with an explicit learned action head are kept.
# Forecast-only baselines and feature-only ablations were removed because they do
# not produce valid action_accuracy/action_macro_f1 and are therefore outside the
# agentic policy study.
SCENARIOS: dict[str, ScenarioConfig] = {
    "agentic_residual_mlp": ScenarioConfig(
        "agentic_residual_mlp", "agentic_residual_mlp", "pytorch", "agentic",
        1, 128, 5, 0.20, 64, 8e-4, "weighted_huber", 2.0
    ),
    "agentic_liquid_residual": ScenarioConfig(
        "agentic_liquid_residual", "agentic_liquid_residual", "pytorch", "agentic",
        16, 128, 3, 0.20, 64, 8e-4, "weighted_huber", 2.0
    ),
    "agentic_sequence_attention": ScenarioConfig(
        "agentic_sequence_attention", "agentic_sequence_attention", "pytorch", "agentic",
        16, 128, 3, 0.20, 64, 8e-4, "weighted_huber", 2.0
    ),
    "agentic_patch_kan_mixer": ScenarioConfig(
        "agentic_patch_kan_mixer", "agentic_patch_kan_mixer", "pytorch", "agentic-sota",
        32, 128, 4, 0.15, 128, 7e-4, "weighted_huber", 2.0
    ),
}


def is_action_model(model_type: str) -> bool:
    return model_type.startswith("agentic_")
