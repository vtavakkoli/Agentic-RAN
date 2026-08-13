from __future__ import annotations

from pathlib import Path

from agentic_ran.data import generate_dataset, write_dataset
from agentic_ran.model import PolicyProposer, train_policy_proposer
from agentic_ran.model_selection import select_production_model


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


def test_production_model_selection_ranks_and_saves_winner(tmp_path: Path) -> None:
    synthetic = write_dataset(generate_dataset(rows=280, seed=21), tmp_path / "synthetic.csv")
    real_frame = generate_dataset(rows=180, seed=22)
    real_path = tmp_path / "real.csv.gz"
    real_frame.to_csv(real_path, index=False, compression="gzip")
    model_path = tmp_path / "winner.joblib"
    metrics_path = tmp_path / "selection.json"
    metrics = select_production_model(synthetic, real_path, model_path, metrics_path, seed=21)
    assert len(metrics["candidates"]) == 4
    assert metrics["selected_model"] in {item["model"] for item in metrics["candidates"]}
    assert model_path.exists()
    assert metrics_path.exists()
    assert PolicyProposer.load(model_path).metadata.version.startswith("realbench-")
