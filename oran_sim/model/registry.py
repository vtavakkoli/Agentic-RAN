from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from .attention_baseline import AttentionBaselineModel
from .balanced_medium import BalancedMediumModel
from .balanced_small import BalancedSmallModel
from .base import ScenarioModelSpec
from .deep_performance import DeepPerformanceModel
from .lightweight_32 import Lightweight32Model
from .lightweight_64 import Lightweight64Model
from .liquid_baseline import LiquidBaselineModel
from .ultra_performance import UltraPerformanceModel
from .xlstm_baseline import XLSTMBaselineModel

SCENARIO_MODEL_SPECS: dict[str, ScenarioModelSpec] = {
    s.name: s
    for s in [
        Lightweight32Model(),
        Lightweight64Model(),
        BalancedSmallModel(),
        BalancedMediumModel(),
        DeepPerformanceModel(),
        UltraPerformanceModel(),
        AttentionBaselineModel(),
        LiquidBaselineModel(),
        XLSTMBaselineModel(),
    ]
}


def build_model(model_name: str, seed: int):
    # Backward-compatible aliases.
    if model_name == "ridge":
        return Ridge(random_state=seed)
    if model_name == "hgb":
        return HistGradientBoostingRegressor(random_state=seed)

    if model_name not in SCENARIO_MODEL_SPECS:
        raise ValueError(f"Unsupported model: {model_name}")
    return SCENARIO_MODEL_SPECS[model_name].build(seed)


def get_model_metadata(model_name: str) -> dict[str, str | bool]:
    if model_name == "ridge":
        return {"backend": "Ridge", "logical_profile": False, "profile_note": "Direct sklearn ridge baseline."}
    if model_name == "hgb":
        return {"backend": "HistGradientBoostingRegressor", "logical_profile": False, "profile_note": "Direct sklearn hist-gradient-boosting baseline."}

    spec_notes = {
        "lightweight-32": ("Ridge", False, "Tabular ridge regression scenario."),
        "lightweight-64": ("Ridge", False, "Tabular ridge regression scenario."),
        "balanced-small": ("HistGradientBoostingRegressor", False, "Tabular gradient boosting scenario."),
        "balanced-medium": ("HistGradientBoostingRegressor", False, "Tabular gradient boosting scenario."),
        "deep-performance": ("HistGradientBoostingRegressor", True, "Logical deep profile implemented with sklearn HistGradientBoostingRegressor (not a torch deep model)."),
        "ultra-performance": ("HistGradientBoostingRegressor", True, "Logical ultra profile implemented with sklearn HistGradientBoostingRegressor (not a torch deep model)."),
        "attention-baseline": ("Ridge", True, "Logical attention profile implemented with sklearn Ridge (not a torch attention model)."),
        "liquid-baseline": ("Ridge", True, "Logical liquid profile implemented with sklearn Ridge (not a torch liquid network)."),
        "xlstm-baseline": ("Ridge", True, "Logical xLSTM profile implemented with sklearn Ridge (not a torch xLSTM model)."),
    }
    if model_name not in spec_notes:
        raise ValueError(f"Unsupported model: {model_name}")
    backend, logical_profile, profile_note = spec_notes[model_name]
    return {"backend": backend, "logical_profile": logical_profile, "profile_note": profile_note}
