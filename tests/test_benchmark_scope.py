from agentic_ran.scenarios import SCENARIOS


def test_only_action_head_models_in_main_scope():
    assert SCENARIOS
    assert all(cfg.model_type.startswith("agentic_") for cfg in SCENARIOS.values())


def test_forecast_only_models_removed_from_main_scope():
    removed = {"mlp", "residual_mlp", "attention", "liquid", "xlstm", "patchtst", "tsmixer", "kan"}
    assert not any(cfg.model_type in removed for cfg in SCENARIOS.values())
