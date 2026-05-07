from __future__ import annotations

import pickle
from pathlib import Path


class GradientBoostingBaseline:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.backend = "sklearn"
        self.model = self._build_model(**kwargs)

    @property
    def model_name(self) -> str:
        return "gradient_boosting_baseline"

    def _build_model(self, **kwargs):
        try:
            from xgboost import XGBRegressor
            self.backend = "xgboost"
            return XGBRegressor(**kwargs)
        except Exception:
            pass
        try:
            from lightgbm import LGBMRegressor
            self.backend = "lightgbm"
            return LGBMRegressor(**kwargs)
        except Exception:
            pass
        try:
            from catboost import CatBoostRegressor
            self.backend = "catboost"
            return CatBoostRegressor(verbose=False, **kwargs)
        except Exception:
            pass
        self.backend = "numpy_linear_fallback"
        class _Fallback:
            def fit(self, X, y):
                import numpy as np
                Xb=np.c_[np.ones(len(X)), X]
                self.coef_=np.linalg.pinv(Xb)@y
                return self
            def predict(self, X):
                import numpy as np
                Xb=np.c_[np.ones(len(X)), X]
                return Xb@self.coef_
        return _Fallback()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)

    def save(self, path: str | Path):
        Path(path).write_bytes(pickle.dumps({"model": self.model, "backend": self.backend, "kwargs": self.kwargs}))

    @classmethod
    def load(cls, path: str | Path):
        payload = pickle.loads(Path(path).read_bytes())
        obj = cls(**payload.get("kwargs", {}))
        obj.model = payload["model"]
        obj.backend = payload.get("backend", "unknown")
        return obj

    def get_params(self) -> dict:
        return {"backend": self.backend, **self.kwargs}
