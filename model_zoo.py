#!/usr/bin/env python3
"""Backward-compatible façade for model definitions and complexity utilities."""

from oran_sim.config import SCENARIOS, supported_scenarios
from oran_sim.models import (
    AttentionRegressor,
    Complexity,
    LSTMRegressor,
    LiquidRegressor,
    build_model,
    compute_complexity,
    count_parameters,
    estimate_model_size_mb,
    lstm_layer_complexity,
)

SCENARIO_CONFIGS = {k: v.__dict__ for k, v in SCENARIOS.items()}


def scenario_feature_count(model_type: str) -> int:
    return SCENARIOS[model_type].features


__all__ = [
    "SCENARIO_CONFIGS",
    "Complexity",
    "LSTMRegressor",
    "AttentionRegressor",
    "LiquidRegressor",
    "build_model",
    "compute_complexity",
    "count_parameters",
    "estimate_model_size_mb",
    "lstm_layer_complexity",
    "scenario_feature_count",
    "supported_scenarios",
]
