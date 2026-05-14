from agentic_ran.scenarios import SCENARIOS


def test_gradient_boosting_removed_from_action_study():
    assert all("gradient" not in cfg.model_type for cfg in SCENARIOS.values())
