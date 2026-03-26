from __future__ import annotations

from abc import ABC, abstractmethod


class ScenarioModelSpec(ABC):
    name: str

    @abstractmethod
    def build(self, seed: int):
        """Build and return a sklearn-compatible regressor."""
