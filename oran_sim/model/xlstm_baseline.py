from __future__ import annotations

from sklearn.linear_model import Ridge

from .base import ScenarioModelSpec


class XLSTMBaselineModel(ScenarioModelSpec):
    name = "xlstm-baseline"

    def build(self, seed: int) -> Ridge:
        return Ridge(alpha=0.6, random_state=seed)
