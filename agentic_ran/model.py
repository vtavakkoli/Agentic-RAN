"""Lightweight policy proposer based on calibrated multinomial classification."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from agentic_ran.config import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET_COLUMN
from agentic_ran.data import validate_dataset


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    version: str
    trained_at: str
    rows: int
    classes: list[str]
    accuracy: float
    macro_f1: float
    random_seed: int
    feature_names: list[str]


class PolicyProposer:
    """Serializable model that proposes policy candidates and probabilities."""

    def __init__(self, pipeline: Pipeline, metadata: ModelMetadata):
        self.pipeline = pipeline
        self.metadata = metadata

    def predict_probabilities(self, record: dict[str, Any]) -> dict[str, float]:
        frame = pd.DataFrame([{name: record[name] for name in FEATURES}])
        probabilities = self.pipeline.predict_proba(frame)[0]
        classes = self.pipeline.named_steps["classifier"].classes_
        return {str(label): float(probability) for label, probability in zip(classes, probabilities, strict=True)}

    def top_k(self, record: dict[str, Any], k: int = 4) -> list[tuple[str, float]]:
        values = self.predict_probabilities(record)
        return sorted(values.items(), key=lambda item: item[1], reverse=True)[:k]

    def save(self, destination: Path | str) -> Path:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "metadata": asdict(self.metadata)}, path, compress=3)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "PolicyProposer":
        payload = joblib.load(path)
        return cls(payload["pipeline"], ModelMetadata(**payload["metadata"]))


def train_policy_proposer(frame: pd.DataFrame, seed: int = 42) -> tuple[PolicyProposer, dict[str, Any]]:
    validate_dataset(frame)
    x = frame[FEATURES].copy()
    y = frame[TARGET_COLUMN].astype(str)
    stratify = y if y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=stratify,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    classifier = HistGradientBoostingClassifier(
        max_iter=80,
        max_leaf_nodes=31,
        learning_rate=0.08,
        l2_regularization=0.1,
        random_state=seed,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))
    macro_f1 = float(f1_score(y_test, predictions, average="macro"))
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

    version = f"histgb-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    metadata = ModelMetadata(
        version=version,
        trained_at=datetime.now(timezone.utc).isoformat(),
        rows=len(frame),
        classes=sorted(str(value) for value in np.unique(y)),
        accuracy=accuracy,
        macro_f1=macro_f1,
        random_seed=seed,
        feature_names=list(FEATURES),
    )
    metrics = {
        "model_version": version,
        "rows": len(frame),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "classification_report": report,
    }
    return PolicyProposer(pipeline, metadata), metrics


def write_training_metrics(metrics: dict[str, Any], path: Path | str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return destination
