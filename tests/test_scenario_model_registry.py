from __future__ import annotations

from oran_sim.config import supported_scenarios
from oran_sim.model import SCENARIO_MODEL_SPECS, build_model


def test_registry_has_entry_for_each_scenario() -> None:
    assert set(supported_scenarios()) == set(SCENARIO_MODEL_SPECS.keys())


def test_build_model_supports_scenario_and_legacy_aliases() -> None:
    assert build_model("lightweight-32", 42).__class__.__name__ == "Ridge"
    assert build_model("balanced-small", 42).__class__.__name__ == "HistGradientBoostingRegressor"
    assert build_model("ridge", 42).__class__.__name__ == "Ridge"
    assert build_model("hgb", 42).__class__.__name__ == "HistGradientBoostingRegressor"
