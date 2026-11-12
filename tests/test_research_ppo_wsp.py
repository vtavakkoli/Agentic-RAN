"""Optional PPO/WSP smoke test; requires the separately installed torch extra."""
import pytest

pytest.importorskip("torch")

from agentic_ran.research_ppo_wsp import PPOActorCritic, _edit, run
import numpy as np


def test_sparse_actor_edits_do_not_touch_critic():
    model = PPOActorCritic()
    before = model.critic.weight.detach().clone()
    edited = _edit(model, action=1, signature=np.ones(64), amplitude=0.5)
    assert (edited.critic.weight == before).all()
    assert (model.critic.weight == before).all()
    assert not (edited.actor.weight == model.actor.weight).all()


def test_ppo_wsp_leaves_test_out_of_selection():
    result, original, edited = run(
        seed=123, updates=1, episodes_per_update=2,
        validation_episodes=4, test_episodes=4, steps=5, rounds=1,
    )
    assert result["test_used_for_edit_selection"] is False
    assert result["paired_vs_frozen_ppo"]["ppo_wsp"]["episodes"] == 4
    assert "ppo_random_edit_control" in result["test_results"]
