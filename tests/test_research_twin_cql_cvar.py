"""Action-dependent digital twin, offline CQL, and distributional-CVaR checks."""
from __future__ import annotations

import numpy as np
import pytest

from agentic_ran.research_twin_environment import (
    ACTIONS, SLA_QUEUE_LIMITS, MultiSliceToyTwin,
    generate_logged_transitions, rollout_episode,
)
from agentic_ran.research_twin_offline import (
    choose_action, fit_calibrated_twin, fit_linear_cql, fit_quantile_td,
)
from agentic_ran.research_twin_study import run_twin_study, _paired_inference


def test_actions_are_valid_allocation_vectors():
    assert ACTIONS.shape == (5, 3)
    assert np.all(ACTIONS >= 0)
    assert np.allclose(ACTIONS.sum(axis=1), 1)
    assert SLA_QUEUE_LIMITS.shape == (3,)


def test_closed_loop_intervention_changes_next_state_without_changing_noise_seed():
    a = MultiSliceToyTwin(seed=12, scenario="congested")
    b = MultiSliceToyTwin(seed=12, scenario="congested")
    assert np.array_equal(a.state, b.state)
    assert np.array_equal(a.state, a.state.copy())
    after_a, reward_a, _ = a.step(1)
    after_b, reward_b, _ = b.step(2)
    assert not np.allclose(after_a[:3], after_b[:3])
    assert reward_a != reward_b
    with pytest.raises(ValueError, match="invalid"):
        a.step(-1)


def test_training_episodes_have_real_toy_action_changes():
    batch = generate_logged_transitions(seed=17, episodes=18, steps=12)
    assert batch.state.shape == (216, 9)
    assert np.all(np.isfinite(batch.reward))
    changed = (batch.action[1:] != batch.action[:-1]) & (
        batch.episode[1:] == batch.episode[:-1]
    )
    assert changed.sum() > 15
    assert np.unique(batch.action).size == 5


def test_models_train_on_offline_trajectories_and_return_finite_scores():
    logged = generate_logged_transitions(seed=22, episodes=18, steps=9)
    twin = fit_calibrated_twin(logged)
    cql = fit_linear_cql(logged, alpha=.2, iterations=2, steps_per_iter=14)
    distributional = fit_quantile_td(logged, iterations=2, steps_per_iter=16)
    state = logged.state[0]
    assert cql.predict(state, 0, 0).shape == (5,)
    assert distributional.distribution(state, 0, 0).shape == (5, 7)
    assert np.isfinite(cql.weights).all()
    assert np.isfinite(distributional.weights).all()
    assert np.isfinite(twin.queue_cvar_margin).all()
    assert np.all(twin.queue_cvar_margin >= twin.queue_upper_margin)
    for mode in ("twin_greedy", "twin_greedy_cvar_gate",
                 "twin_cql", "distributional_cvar", "twin_cql_cvar_hybrid"):
        action = choose_action(
            mode, state, 0, 0, twin=twin, cql=cql,
            quantile=distributional, risk_weight=.35,
        )
        assert -1 <= action < 5


def test_unseen_far_out_of_support_state_fails_closed():
    logged = generate_logged_transitions(seed=23, episodes=18, steps=9)
    twin = fit_calibrated_twin(logged)
    ood_state = np.full(9, 100.0)
    eligible, _ = twin.eligible(ood_state, 4, risk_mode="cvar")
    assert not eligible.any()
    assert choose_action(
        "twin_greedy", ood_state, 0, 4, twin=twin,
    ) == -1


def test_abstention_is_actual_hold_last_in_simulated_world():
    base = rollout_episode(
        lambda state, previous, t: -1, seed=33,
        scenario="weak_channel", steps=8,
    )
    ref = rollout_episode(
        lambda state, previous, t: 0, seed=33,
        scenario="weak_channel", steps=8,
    )
    assert base == ref | {"abstention_rate": 1.0}
    assert base["abstention_rate"] == 1.0


def test_paired_inference_refuses_wrong_episode_alignment():
    a = [
        {"seed": 1, "scenario": "normal", "return": .8},
        {"seed": 2, "scenario": "normal", "return": .7},
        {"seed": 3, "scenario": "normal", "return": .6},
        {"seed": 4, "scenario": "normal", "return": .5},
    ]
    ref = [dict(x) for x in a]
    summary = _paired_inference(a, ref, seed=50, nboot=100)
    assert summary["mean_episode_delta"] == 0
    assert summary["ci95"] == [0.0, 0.0]
    ref[1]["seed"] = 120
    with pytest.raises(ValueError, match="paired"):
        _paired_inference(a, ref, seed=50, nboot=100)


def test_small_end_to_end_study_has_frozen_selection_and_no_commag_claim():
    summary, table = run_twin_study(
        seed=44, train_episodes=18,
        validation_episodes=6, test_episodes=6, steps=7,
    )
    assert summary["action_audit"]["observed_within_episode_switches"] > 0
    assert not summary["validation_selection_used_final_test"]
    assert summary["experiment"] == "three_slice_action_dependent_toy_digital_twin"
    assert not summary["publication_ready_as_real_ran_superiority"]
    assert summary["validation_selected_family"] in (
        "twin_greedy", "twin_cql", "distributional_cvar", "twin_cql_cvar_hybrid",
    )
    assert len(table) == 2 * 6 * 6
    assert set(table["split"]) == {"seen", "unseen"}
    assert set(table["policy"]) == {
        "heuristic", "twin_greedy", "twin_greedy_cvar_gate",
        "twin_cql", "distributional_cvar", "twin_cql_cvar_hybrid",
    }
    for split in ("seen", "unseen"):
        assert len(summary["paired_vs_original_twin_greedy"][split]) == 3
        assert len(summary["paired_vs_same_risk_gate_greedy"][split]) == 3
        for name, outcome in summary["final_test_metrics"][split].items():
            assert np.isfinite(outcome["return"])
            assert 0 <= outcome["sla_URLLC"] <= 1
            assert 0 <= outcome["abstention_rate"] <= 1
