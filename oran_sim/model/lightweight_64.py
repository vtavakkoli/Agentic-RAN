from __future__ import annotations

from sklearn.linear_model import Ridge

from .base import ScenarioModelSpec


class Lightweight64Model(ScenarioModelSpec):
    name = "lightweight-64"

    def build(self, seed: int) -> Ridge:
        return Ridge(alpha=0.8, random_state=seed)
