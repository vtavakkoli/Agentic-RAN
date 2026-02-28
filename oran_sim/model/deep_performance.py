from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor

from .base import ScenarioModelSpec


class DeepPerformanceModel(ScenarioModelSpec):
    name = "deep-performance"

    def build(self, seed: int) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(max_depth=10, learning_rate=0.05, random_state=seed)
