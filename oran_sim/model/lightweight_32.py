from __future__ import annotations

from sklearn.linear_model import Ridge

from .base import ScenarioModelSpec


class Lightweight32Model(ScenarioModelSpec):
    name = "lightweight-32"

    def build(self, seed: int) -> Ridge:
        return Ridge(alpha=1.0, random_state=seed)
