from __future__ import annotations

import pytest

from oran_sim.config import supported_scenarios
from oran_sim.model import SCENARIO_MODEL_SPECS, build_model, get_model_metadata


def test_registry_has_entry_for_each_scenario() -> None:
    assert set(supported_scenarios()) == set(SCENARIO_MODEL_SPECS.keys())


def test_build_model_supports_tabular_and_legacy_aliases() -> None:
    assert build_model("lightweight-32", 42).__class__.__name__ == "Ridge"
    assert build_model("balanced-small", 42).__class__.__name__ == "HistGradientBoostingRegressor"
    assert build_model("ridge", 42).__class__.__name__ == "Ridge"
    assert build_model("hgb", 42).__class__.__name__ == "HistGradientBoostingRegressor"


def test_temporal_scenario_not_built_by_tabular_factory() -> None:
    with pytest.raises(ValueError):
        build_model("attention-baseline", 42)


def test_metadata_exposes_real_backend_profile() -> None:
    meta = get_model_metadata("attention-baseline")
    assert meta["backend"].startswith("Torch")
    assert meta["logical_profile"] == "temporal_sequence_model"
