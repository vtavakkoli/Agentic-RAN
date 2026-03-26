import pandas as pd

from scripts.aggregate_report import _prepare_preds_for_plot


def test_prepare_preds_for_plot_sorts_by_time_ms() -> None:
    preds = pd.DataFrame(
        {
            "time_ms": [30, 10, 20],
            "y_true": [3.0, 1.0, 2.0],
            "y_pred": [2.8, 1.2, 2.1],
        }
    )

    sorted_preds = _prepare_preds_for_plot(preds)

    assert sorted_preds["time_ms"].tolist() == [10, 20, 30]
    assert sorted_preds["y_true"].tolist() == [1.0, 2.0, 3.0]


def test_prepare_preds_for_plot_prefers_global_index_when_available() -> None:
    preds = pd.DataFrame(
        {
            "global_index": [2, 0, 1],
            "time_ms": [100, 300, 200],
            "y_true": [3.0, 1.0, 2.0],
            "y_pred": [2.8, 1.2, 2.1],
        }
    )

    sorted_preds = _prepare_preds_for_plot(preds)

    assert sorted_preds["global_index"].tolist() == [0, 1, 2]
