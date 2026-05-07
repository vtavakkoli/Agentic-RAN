from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseModel(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def fit(self, X_train, y_train): ...

    @abstractmethod
    def predict(self, X_test): ...

    @abstractmethod
    def save(self, path: str | Path): ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path): ...

    @abstractmethod
    def get_params(self) -> dict: ...
