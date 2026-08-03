from __future__ import annotations

from pathlib import Path

from agentic_ran.data import generate_dataset
from agentic_ran.model import PolicyProposer, train_policy_proposer


def test_training_produces_all_policy_classes() -> None:
    proposer, metrics = train_policy_proposer(generate_dataset(rows=700, seed=4), seed=4)
    assert len(proposer.metadata.classes) == 7
    assert metrics["accuracy"] >= 0.72
    assert metrics["macro_f1"] >= 0.68


def test_model_round_trip_preserves_probabilities(tmp_path: Path) -> None:
    frame = generate_dataset(rows=420, seed=5)
    proposer, _ = train_policy_proposer(frame, seed=5)
    record = frame.iloc[0].to_dict()
    before = proposer.predict_probabilities(record)
    path = proposer.save(tmp_path / "model.joblib")
    after = PolicyProposer.load(path).predict_probabilities(record)
    assert before.keys() == after.keys()
    for name in before:
        assert abs(before[name] - after[name]) < 1e-12


def test_top_k_is_ordered() -> None:
    frame = generate_dataset(rows=350, seed=8)
    proposer, _ = train_policy_proposer(frame, seed=8)
    values = proposer.top_k(frame.iloc[10].to_dict(), k=4)
    assert len(values) == 4
    assert values == sorted(values, key=lambda item: item[1], reverse=True)
