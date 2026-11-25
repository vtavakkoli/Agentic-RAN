"""PA-SPI invariants; synthetic fixtures are not scientific evidence."""
from pathlib import Path

import numpy as np
import pytest

from agentic_ran.research_pulse_anchored_spi import (
    fit, features, make_world, TEST_SCENARIOS, N, PulseSPI, samples,
)
from agentic_ran.research_pulse_ran import select_pulse_ran
pytest.importorskip("torch")
pytest.importorskip("gymnasium")

from agentic_ran.research_paspi_campaign import run, REFERENCE, METHODS  # noqa: E402


def test_pulse_thresholds_frozen():
    s = np.ones(9)
    s[:3] = [1., .1, .45]
    assert select_pulse_ran(s) == 0
    s[0] = 1.00001
    assert select_pulse_ran(s) == 2
    s[2] = .45001
    assert select_pulse_ran(s) == 1


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_toy_stress_reset_and_executed_action(scenario):
    a = make_world(123, scenario)
    b = make_world(123, scenario)
    np.testing.assert_array_equal(a.state, b.state)
    for i in range(8):
        x, y = a.step(i % N), b.step(i % N)
        np.testing.assert_array_equal(x[0], y[0])
        assert x[1] == y[1]
        np.testing.assert_array_equal(x[2], y[2])
        assert a.previous_action == i % N


def test_no_invalid_observations():
    with pytest.raises(ValueError):
        features(np.full(9, np.nan), 0, 0)
    with pytest.raises(ValueError):
        features(np.ones(9), N, 0)


def test_support_risk_and_switch_gate(monkeypatch):
    m = PulseSPI(None, None, [], None, 1., np.zeros(N), np.zeros((N, 3)))
    s = np.ones(9)
    means = np.zeros((N, 4))
    means[2, 0] = 1.
    std = np.zeros_like(means)
    monkeypatch.setattr(m, "estimate", lambda *args: (0, means.copy(), std.copy(), 2.))
    action, trace = m.decide(s, 0, 0)
    assert action == 0 and trace["reason"] == "unsupported_reference_fallback"
    assert m.decide(s, 0, 0, mode="pa_no_support")[0] == 2
    means[2, 1] = .5
    monkeypatch.setattr(m, "estimate", lambda *args: (0, means.copy(), std.copy(), .1))
    assert m.decide(s, 0, 0)[0] == 0
    assert m.decide(s, 0, 0, mode="pa_no_risk")[0] == 2
    means[2, 1] = 0.
    means[2, 0] = .07
    assert m.decide(s, 0, 0)[0] == 0
    assert m.decide(s, 0, 0, mode="pa_no_switch")[0] == 2


def test_calibration_and_baseline_anchor():
    spi, audit = fit(71, 12, 4, 5, horizon=2)
    assert audit["train_seed"] != audit["calibration_seed"]
    assert not audit["formal_safety_certificate"]
    s = make_world(991, "weak_channel").state
    ref, mean, std, distance = spi.estimate(s, 0, 0)
    np.testing.assert_allclose(mean[ref], 0.)
    np.testing.assert_allclose(std[ref], 0.)
    a, trace = spi.decide(s, 0, 0)
    assert 0 <= a < N and np.isfinite(distance)
    if not trace["state_support"]:
        assert a == ref


def test_branch_labels_reference_exact_zero():
    x, y, groups = samples(999, 12, 5, 2)
    assert len(np.unique(groups)) == 12
    for row, outcome in zip(x, y):
        np.testing.assert_allclose(outcome[select_pulse_ran(row[:9])], 0.)


def test_full_confirmatory_smoke(tmp_path: Path):
    r = run(tmp_path / "study", training_runs=2, test_seeds=4, steps=5,
            logged_episodes=12, validation_episodes=4, ppo_updates=1,
            episodes_per_update=2, wsp_rounds=1, critic_iterations=1,
            workers=1, smoke=True)
    assert r["executed_episodes"] == 2 * 4 * len(TEST_SCENARIOS) * len(METHODS)
    assert r["executed_steps"] == r["executed_episodes"] * 5
    assert not r["primary_superiority_and_sla_noninferiority"]
    assert not r["physical_nr_evidence"]
    assert not r["publication_ready_for_ieee_access"]
    assert REFERENCE in r["all_comparisons"]
    assert (tmp_path / "study" / "artifact_sha256.json").is_file()
    assert (tmp_path / "study" / "report.html").is_file()
