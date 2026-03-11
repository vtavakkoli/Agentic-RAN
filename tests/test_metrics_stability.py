from __future__ import annotations

import numpy as np

from scripts.train import _metrics


def test_metrics_include_smape_wmape() -> None:
    y_true = np.array([0.01, 0.02, 1.0, 2.0])
    y_pred = np.array([0.02, 0.01, 1.1, 1.8])
    m = _metrics(y_true, y_pred)
    assert "sMAPE" in m and "wMAPE" in m
    assert np.isfinite(m["sMAPE"])
    assert np.isfinite(m["wMAPE"])
