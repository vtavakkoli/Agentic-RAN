from pathlib import Path

import pandas as pd

from scripts.run_scenario import _combine_split_predictions


def test_combine_split_predictions_appends_global_index_and_split(tmp_path: Path) -> None:
    train = pd.DataFrame({"index": [0, 1], "y_true": [1.0, 2.0], "y_pred": [1.1, 1.9]})
    val = pd.DataFrame({"index": [0], "y_true": [3.0], "y_pred": [2.9]})
    test = pd.DataFrame({"index": [0, 1], "y_true": [4.0, 5.0], "y_pred": [4.1, 5.2]})

    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    test_path = tmp_path / "test.csv"

    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)

    combined = _combine_split_predictions(
        [
            ("train", train_path),
            ("val", val_path),
            ("test", test_path),
        ]
    )

    assert combined["split"].tolist() == ["train", "train", "val", "test", "test"]
    assert combined["global_index"].tolist() == [0, 1, 2, 3, 4]
