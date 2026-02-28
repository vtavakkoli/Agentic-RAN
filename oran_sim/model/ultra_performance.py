from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor

from .base import ScenarioModelSpec


class UltraPerformanceModel(ScenarioModelSpec):
    name = "ultra-performance"

    def build(self, seed: int) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(max_depth=12, learning_rate=0.04, random_state=seed)
