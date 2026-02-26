import numpy as np

from scripts.predict import compute_pct_error


def test_compute_pct_error_uses_absolute_denominator_and_handles_zero() -> None:
    y_true = np.array([10.0, -10.0, 0.0])
    y_pred = np.array([8.0, -12.0, 5.0])

    pct = compute_pct_error(y_true, y_pred)

    # error = y_true - y_pred -> [2, 2, -5]
    # denominator = abs(y_true) -> [10, 10, 0]
    assert np.isclose(pct[0], 20.0)
    assert np.isclose(pct[1], 20.0)
    assert np.isnan(pct[2])
