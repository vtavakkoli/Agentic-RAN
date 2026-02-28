from __future__ import annotations

from sklearn.linear_model import Ridge

from .base import ScenarioModelSpec


class LiquidBaselineModel(ScenarioModelSpec):
    name = "liquid-baseline"

    def build(self, seed: int) -> Ridge:
        return Ridge(alpha=1.5, random_state=seed)
