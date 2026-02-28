from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor

from .base import ScenarioModelSpec


class BalancedSmallModel(ScenarioModelSpec):
    name = "balanced-small"

    def build(self, seed: int) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(max_depth=6, learning_rate=0.08, random_state=seed)
