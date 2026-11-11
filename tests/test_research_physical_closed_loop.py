"""Tests for physically valid PRB allocations and strict real-actuation protocol.

A deterministic fake *bridge* validates JSON-lines protocol behavior only.
It is NOT ns-3, Colosseum, or evidence of real network gains.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from agentic_ran.research_external_ran import ExternalRANSession, run_realized_episode
from agentic_ran.research_physical_prb import (
    NRCellCapacity, SliceAllocation, allocate_shares, commag_physical_budget_config,
)
from agentic_ran.research_realized_advantage import (
    action_library, fit_advantages, logged_from_external,
)
from agentic_ran.research_joint_slice_advantage import exact_multiple_choice_knapsack


@pytest.fixture
def physical_cell():
    return NRCellCapacity("bs1", 20, 30, reserved_prbs=3)


@pytest.fixture
def fake_bridge(tmp_path):
    script = tmp_path / "fake_ran_bridge.py"
    script.write_text(
        """
import json,sys
schema="agentic-ran-physical-prb-v1"
seed=0
t=0
state=[0.2,0.3,0.1,0.9,0.4,0.5,1.0,0.8,0.9]
for line in sys.stdin:
    req=json.loads(line)
    if req["op"]=="reset":
        seed=req["seed"]; t=0
        state=[0.2,0.3,0.1,0.9,0.4,0.5,1.0,0.8,0.9]
        out={"schema":schema,"type":"observation","step_index":0,"state":state}
    else:
        t+=1
        action=req["allocation"]
        counts=[action["slice_prbs"][s] for s in ("eMBB","mMTC","URLLC")]
        scale=action["available_prbs"]
        throughput=[float(v/scale*15) for v in counts]
        state[:3]=[max(0.0,state[i]+0.20-counts[i]/scale) for i in range(3)]
        violations=[bool(x>0.95) for x in state[:3]]
        reward=float((counts[0]+0.8*counts[1]+1.1*counts[2])/scale
                     -sum(violations)*0.1)
        out={"schema":schema,"type":"transition","step_index":t,
             "next_state":state,"reward":reward,
             "sla_violations":violations,"done":False,
             "kpis":{"throughput_mbps":throughput,
                     "latency_ms":[float(1+10*x) for x in state[:3]]},
             "applied_allocation":action}
    sys.stdout.write(json.dumps(out)+"\\n")
    sys.stdout.flush()
""".strip() + "\n", encoding="utf-8",
    )
    return [sys.executable, "-u", str(script)]


def test_physical_nr_capacity_and_action_constraints(physical_cell):
    assert physical_cell.carrier_prbs == 51
    assert physical_cell.available_prbs == 48
    alloc = allocate_shares(physical_cell, (.5, .2, .3), minimums=(1, 1, 1))
    assert sum(alloc.prbs) == 48
    alloc.validate(physical_cell, minimums=(1, 1, 1))
    with pytest.raises(ValueError, match="overallocation"):
        SliceAllocation((30, 20, 10)).validate(physical_cell)
    with pytest.raises(ValueError, match="unsupported FR1"):
        NRCellCapacity("bs1", 250, 30).available_prbs
    with pytest.raises(ValueError, match="reserved_prbs"):
        NRCellCapacity("bs1", 20, 30, reserved_prbs=51).available_prbs
    with pytest.raises(ValueError, match="sum to one"):
        allocate_shares(physical_cell, (.2, .2, .2))


def test_physical_commag_mapping_fails_closed(tmp_path):
    cfg = tmp_path / "cells.json"
    cfg.write_text(json.dumps({"cells": {"bs1": {
        "cell_id": "bs1", "bandwidth_mhz": 20, "scs_khz": 30
    }}}), encoding="utf-8")
    with pytest.raises(ValueError, match="explicitly confirm"):
        commag_physical_budget_config(str(cfg))
    data = json.loads(cfg.read_text())
    data["confirmed_commag_prb_units"] = True
    cfg.write_text(json.dumps(data))
    b = commag_physical_budget_config(str(cfg))
    assert b["budget_by_base_station"]["bs1"] == 51
    assert b["shared_scheduler_required"] is True
    assert b["physical_capacity_confirmed"]


def test_knapsack_enforces_common_scheduler():
    # Without scheduler coordination each slice prefers a different action.
    score = np.array([[.9,.2,.2],[.1,.9,.1],[.2,.2,.9]])
    valid = np.ones((3,3), dtype=bool)
    costs = np.ones(3, dtype=int)
    free, free_score = exact_multiple_choice_knapsack(score, valid, costs, 3)
    assert free.tolist() == [0,1,2]
    one, restricted = exact_multiple_choice_knapsack(
        score, valid, costs, 3,
        scheduler_names=("rr","wf","pf"),
    )
    assert len(set(one.tolist())) == 1
    assert restricted < free_score


def test_external_actuation_changes_next_state_and_has_receipt(fake_bridge, physical_cell):
    small = SliceAllocation((1,1,1))
    large = allocate_shares(physical_cell, (.5,.2,.3))
    with ExternalRANSession(fake_bridge, physical_cell) as session:
        initial = session.reset(scenario="A", seed=7, steps=5)
        low = session.step(small)
        assert low.reward > 0
        assert not np.array_equal(low.next_state, initial)
    with ExternalRANSession(fake_bridge, physical_cell) as session:
        session.reset(scenario="A", seed=7, steps=5)
        high = session.step(large)
    assert high.reward != low.reward
    assert high.next_state[0] < low.next_state[0]


def test_randomized_executed_training_requires_variation(fake_bridge, physical_cell):
    choices = action_library(physical_cell)
    records = logged_from_external(
        fake_bridge, physical_cell, scenarios=("normal",),
        seeds=(1,2,3,4), steps=12,
    )
    assert len(records) == 48
    assert all(x["action_verified"] and x["behavior_randomized"] for x in records)
    model = fit_advantages(records, choices, min_support=2, ensemble=4, min_action_switches=4)
    original = choices[0]
    picked = model.choose(np.array(records[0]["state"]), original)
    assert picked in choices
    bad = [dict(x, behavior_randomized=False) for x in records]
    with pytest.raises(ValueError, match="genuine randomized"):
        fit_advantages(bad, choices, min_support=2, ensemble=4)
    static = [dict(x, action_index=0) for x in records]
    with pytest.raises(ValueError, match="variation"):
        fit_advantages(static, choices, min_support=2, ensemble=4)


def test_matched_episode_actuates_not_static_replay(fake_bridge, physical_cell):
    choices = action_library(physical_cell)
    a, _ = run_realized_episode(
        fake_bridge, physical_cell, lambda s, p, t: choices[0],
        seed=19, scenario="normal", steps=4,
    )
    b, _ = run_realized_episode(
        fake_bridge, physical_cell, lambda s, p, t: choices[1],
        seed=19, scenario="normal", steps=4,
    )
    assert a["mean_realized_reward"] != b["mean_realized_reward"]
