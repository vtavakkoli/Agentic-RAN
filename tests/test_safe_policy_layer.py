from agentic_ran.agentic_policy import ACTION_SPACE, recommend_action


def test_agentic_policy_returns_valid_action_for_unsafe_pressure():
    decision = recommend_action({"prb_pressure": 1.5, "traffic_class": "URLLC", "ratio_granted_req": 0.8})
    assert decision["action_id"] in ACTION_SPACE
    assert "action_name" in decision
