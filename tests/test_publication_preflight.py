"""Fast-fail regression checks for optional physical-JAGC evaluation."""
from __future__ import annotations

import pytest

from agentic_ran.publication import evaluate


def test_missing_physical_manifest_fails_before_data_load(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTIC_RAN_EVAL_JOINT_SLICE_ADVANTAGE", "1")
    monkeypatch.setenv("AGENTIC_RAN_JAGC_BWP_MANIFEST", str(tmp_path / "missing.json"))
    with pytest.raises(FileNotFoundError, match="JAGC physical BWP manifest"):
        evaluate(tmp_path / "nonexistent.csv.gz", tmp_path / "output", None, tmp_path / "science.yaml")


def test_unset_physical_manifest_fails_before_data_load(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTIC_RAN_EVAL_JOINT_SLICE_ADVANTAGE", "1")
    monkeypatch.delenv("AGENTIC_RAN_JAGC_BWP_MANIFEST", raising=False)
    with pytest.raises(RuntimeError, match="AGENTIC_RAN_JAGC_BWP_MANIFEST is unset"):
        evaluate(tmp_path / "nonexistent.csv.gz", tmp_path / "output", None, tmp_path / "science.yaml")
