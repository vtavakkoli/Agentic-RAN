from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def rank_features_by_importance(
    train_df: pd.DataFrame,
    candidate_features: list[str],
    target_col: str = "target",
    *,
    random_state: int = 42,
    n_estimators: int = 200,
) -> pd.DataFrame:
    """Rank features by RandomForestRegressor importance using train split only."""
    features = [c for c in candidate_features if c in train_df.columns]
    if not features:
        raise ValueError("No candidate features found in training dataframe")

    x = train_df[features].copy()
    for col in x.columns:
        if not pd.api.types.is_numeric_dtype(x[col]):
            x[col] = x[col].astype("category").cat.codes
    x = x.fillna(0.0)
    y = train_df[target_col].to_numpy()

    rf = RandomForestRegressor(random_state=random_state, n_estimators=n_estimators)
    rf.fit(x, y)

    importance_df = pd.DataFrame({"feature": features, "importance": rf.feature_importances_})
    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return importance_df


def select_top_k_features(importance_df: pd.DataFrame, k: int) -> list[str]:
    if k <= 0:
        raise ValueError("k must be > 0")
    return importance_df["feature"].head(k).tolist()


def write_feature_importance_artifacts(
    importance_df: pd.DataFrame,
    *,
    json_path: Path,
    csv_path: Path | None = None,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_importance": [
            {"rank": int(i + 1), "feature": row[0], "importance": float(row[1])}
            for i, row in enumerate(importance_df.itertuples(index=False, name=None))
        ]
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        importance_df.to_csv(csv_path, index=False)
