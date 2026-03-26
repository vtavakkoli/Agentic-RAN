from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from oran_sim.config import SCENARIOS


def build_model(model_name: str, seed: int):
    if model_name == "ridge":
        return Ridge(random_state=seed)
    if model_name == "hgb":
        return HistGradientBoostingRegressor(random_state=seed)

    if model_name not in SCENARIOS:
        raise ValueError(f"Unsupported model: {model_name}")

    scenario = SCENARIOS[model_name]
    if scenario.kind != "tabular":
        raise ValueError(f"Scenario '{model_name}' is temporal and must be trained with the temporal pipeline")

    if model_name in {"lightweight-32", "lightweight-64"}:
        return Ridge(random_state=seed)
    return HistGradientBoostingRegressor(random_state=seed)


SCENARIO_MODEL_SPECS: dict[str, str] = {name: SCENARIOS[name].kind for name in SCENARIOS}


def get_model_metadata(model_name: str) -> dict[str, str]:
    if model_name == "ridge":
        return {
            "backend": "Ridge",
            "logical_profile": "tabular_baseline",
            "profile_note": "Direct sklearn ridge baseline.",
        }
    if model_name == "hgb":
        return {
            "backend": "HistGradientBoostingRegressor",
            "logical_profile": "tree_boosting_profile",
            "profile_note": "Direct sklearn gradient-boosting baseline.",
        }

    if model_name not in SCENARIOS:
        raise ValueError(f"Unsupported model: {model_name}")

    scenario = SCENARIOS[model_name]
    if scenario.kind == "tabular":
        backend = "Ridge" if model_name in {"lightweight-32", "lightweight-64"} else "HistGradientBoostingRegressor"
        profile = "tabular_baseline" if backend == "Ridge" else "tree_boosting_profile"
        return {
            "backend": backend,
            "logical_profile": profile,
            "profile_note": "Real sklearn tabular model.",
        }

    backend_map = {
        "attention": "TorchAttentionRegressor",
        "liquid": "TorchLiquidRegressor",
        "lstm": "TorchLSTMRegressor",
    }
    backend = backend_map.get(scenario.architecture or "", "TorchSequenceModel")
    return {
        "backend": backend,
        "logical_profile": "temporal_sequence_model",
        "profile_note": "Real temporal model trained on rolling windows.",
    }
