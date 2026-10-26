"""Regression tests for two-role local-support selective offline policies."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agentic_ran.publication_support import CONTEXT_COLUMNS, StateConditionalSupportAudit


def make_frame(center: float, count: int = 64, action: str = "A") -> pd.DataFrame:
    rng = np.random.default_rng(151)
    data = {
        name: rng.normal(loc=center, scale=0.1, size=count)
        for name in CONTEXT_COLUMNS
    }
    return pd.DataFrame({**data, "slice_type": "eMBB", "action": action})


def gate() -> StateConditionalSupportAudit:
    return StateConditionalSupportAudit(
        make_frame(0.0), make_frame(0.0),
        min_reference=8, max_reference=64,
    )


def test_near_is_supported_far_and_unseen_action_rejected():
    model = gate()
    near = make_frame(0.0)
    far = make_frame(50.0)
    assert model.candidate_supported(near, ["A"] * len(near)).all()
    assert not model.candidate_supported(far, ["A"] * len(far)).any()
    assert not model.candidate_supported(near, ["B"] * len(near)).any()


def test_abstention_preserves_supported_incumbent_and_falls_back_if_needed():
    model = gate()
    far = make_frame(50.0)
    count = len(far)
    result = model.guard_selections(
        far, ["A"] * count,
        ["A"] * (count // 2) + ["Z"] * (count // 2),
        allowed_by_slice={"eMBB": {"A", "B"}},
        fallback_by_slice={"eMBB": "B"},
        batch_size=7,
    )
    assert result.support_deferral.all()
    assert result.loc[:count // 2 - 1, "selected_action"].eq("A").all()
    assert result.loc[count // 2:, "selected_action"].eq("B").all()
    assert result.unsupported_incumbent_fallback.sum() == count // 2


def test_nonfinite_context_rejected_closed():
    model = gate()
    frame = make_frame(0.0)
    frame.loc[0, CONTEXT_COLUMNS[0]] = np.nan
    mask = model.candidate_supported(frame, ["A"] * len(frame), batch_size=3)
    assert not mask[0]
    assert mask[1:].all()


def test_invalid_input_rejected():
    model = gate()
    with pytest.raises(ValueError, match="length"):
        model.candidate_supported(make_frame(0.0), ["A"])
    with pytest.raises(ValueError, match="positive"):
        model.candidate_supported(make_frame(0.0), ["A"] * 64, batch_size=0)
    with pytest.raises(ValueError, match="fallback"):
        model.guard_selections(
            make_frame(99.0), ["A"] * 64, ["Z"] * 64,
            allowed_by_slice={"eMBB": {"A"}},
            fallback_by_slice={"eMBB": "B"},
        )
