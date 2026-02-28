from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor

from .base import ScenarioModelSpec


class BalancedMediumModel(ScenarioModelSpec):
    name = "balanced-medium"

    def build(self, seed: int) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(max_depth=8, learning_rate=0.06, random_state=seed)
