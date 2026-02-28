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
