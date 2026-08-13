"""Compare lightweight policy proposers for realistic deployment suitability."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from agentic_ran.config import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET_COLUMN
from agentic_ran.data import load_dataset, validate_dataset
from agentic_ran.model import ModelMetadata, PolicyProposer


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


def _candidates(seed: int) -> dict[str, Any]:
    return {
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=120, max_leaf_nodes=31, learning_rate=0.07, l2_regularization=0.15, random_state=seed
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=180, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=180, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=seed
        ),
        "logistic_regression": LogisticRegression(max_iter=700, class_weight="balanced", random_state=seed),
    }


def _robustness(pipeline: Pipeline, frame: pd.DataFrame, seed: int) -> float:
    rng = np.random.default_rng(seed)
    original = pipeline.predict(frame[FEATURES])
    perturbed = frame[FEATURES].copy()
    for name in NUMERIC_FEATURES:
        values = pd.to_numeric(perturbed[name], errors="coerce").astype(float)
        scale = max(float(values.std()), max(abs(float(values.mean())), 1.0) * 0.02)
        perturbed[name] = values + rng.normal(0.0, 0.05 * scale, len(values))
    return float(np.mean(original == pipeline.predict(perturbed)))


def _latency_p95_ms(pipeline: Pipeline, frame: pd.DataFrame, limit: int = 100) -> float:
    samples = frame[FEATURES].head(limit)
    timings = []
    for row in samples.to_dict(orient="records"):
        started = time.perf_counter_ns()
        pipeline.predict_proba(pd.DataFrame([row]))
        timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return float(np.percentile(timings, 95)) if timings else 0.0


def select_production_model(
    synthetic_path: Path | str,
    real_path: Path | str,
    model_path: Path | str,
    metrics_path: Path | str,
    seed: int = 42,
) -> dict[str, Any]:
    """Rank candidate proposers using quality, real-data agreement, robustness, calibration, and latency."""

    synthetic = load_dataset(synthetic_path)
    real = pd.read_csv(real_path)
    validate_dataset(real)
    train, test = train_test_split(
        synthetic,
        test_size=0.25,
        random_state=seed,
        stratify=synthetic[TARGET_COLUMN].astype(str),
    )

    results = []
    trained: dict[str, Pipeline] = {}
    for name, classifier in _candidates(seed).items():
        pipeline = Pipeline([("preprocessor", _preprocessor()), ("classifier", classifier)])
        pipeline.fit(train[FEATURES], train[TARGET_COLUMN].astype(str))
        trained[name] = pipeline
        prediction = pipeline.predict(test[FEATURES])
        probabilities = pipeline.predict_proba(test[FEATURES])
        classes = list(pipeline.named_steps["classifier"].classes_)
        real_prediction = pipeline.predict(real[FEATURES])
        macro_f1 = float(f1_score(test[TARGET_COLUMN], prediction, average="macro"))
        accuracy = float(accuracy_score(test[TARGET_COLUMN], prediction))
        real_agreement = float(f1_score(real[TARGET_COLUMN], real_prediction, average="macro", zero_division=0))
        calibration = float(log_loss(test[TARGET_COLUMN], probabilities, labels=classes))
        robustness = _robustness(pipeline, test, seed)
        p95_ms = _latency_p95_ms(pipeline, test)
        calibration_score = 1.0 / (1.0 + calibration)
        latency_score = 1.0 / (1.0 + p95_ms / 5.0)
        production_score = (
            0.45 * macro_f1
            + 0.25 * real_agreement
            + 0.15 * robustness
            + 0.10 * calibration_score
            + 0.05 * latency_score
        )
        results.append(
            {
                "model": name,
                "synthetic_accuracy": accuracy,
                "synthetic_macro_f1": macro_f1,
                "real_expert_macro_f1": real_agreement,
                "robustness_stability": robustness,
                "log_loss": calibration,
                "p95_inference_ms": p95_ms,
                "production_score": production_score,
            }
        )

    results.sort(key=lambda item: item["production_score"], reverse=True)
    winner = results[0]
    winner_name = str(winner["model"])
    capped_real = real.sample(n=min(len(real), max(1, len(synthetic) // 4)), random_state=seed)
    final_frame = pd.concat([synthetic, capped_real], ignore_index=True)
    final_pipeline = Pipeline([("preprocessor", _preprocessor()), ("classifier", _candidates(seed)[winner_name])])
    final_pipeline.fit(final_frame[FEATURES], final_frame[TARGET_COLUMN].astype(str))
    version = f"realbench-{winner_name}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    metadata = ModelMetadata(
        version=version,
        trained_at=datetime.now(timezone.utc).isoformat(),
        rows=len(final_frame),
        classes=sorted(final_frame[TARGET_COLUMN].astype(str).unique().tolist()),
        accuracy=float(winner["synthetic_accuracy"]),
        macro_f1=float(winner["synthetic_macro_f1"]),
        random_seed=seed,
        feature_names=list(FEATURES),
    )
    proposer = PolicyProposer(final_pipeline, metadata)
    proposer.save(model_path)
    metrics = {
        "selected_model": winner_name,
        "selected_model_version": version,
        "selection_formula": "45% synthetic macro-F1 + 25% real expert-reference macro-F1 + 15% perturbation stability + 10% calibration + 5% latency",
        "real_target_semantics": "expert-derived reference labels; not operator ground truth",
        "synthetic_rows": len(synthetic),
        "real_rows": len(real),
        "real_rows_added_to_final_training": len(capped_real),
        "candidates": results,
        "winner": winner,
        "metadata": asdict(metadata),
    }
    destination = Path(metrics_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
