from __future__ import annotations

from sklearn.linear_model import Ridge

from .base import ScenarioModelSpec


class AttentionBaselineModel(ScenarioModelSpec):
    name = "attention-baseline"

    def build(self, seed: int) -> Ridge:
        return Ridge(alpha=1.2, random_state=seed)
