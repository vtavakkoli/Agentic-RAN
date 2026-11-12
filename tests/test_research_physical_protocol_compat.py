"""Regression coverage for the coexistence of both physical PRB protocols."""
import json

import pytest

from agentic_ran.research_physical_prb import (
    CellBudget, JointAction, NRCellCapacity, SliceAllocation,
    allocate_shares, commag_physical_budget_config,
)


def test_legacy_capacity_and_exact_shared_allocation():
    cell = NRCellCapacity("bs1", bandwidth_mhz=20, scs_khz=30, reserved_prbs=3)
    assert cell.carrier_prbs == 51
    assert cell.available_prbs == 48
    action = allocate_shares(cell, (0.50, 0.20, 0.30))
    assert sum(action.prbs) == cell.available_prbs
    assert action.to_wire(cell)["slice_prbs"]["URLLC"] == action.prbs[2]
    SliceAllocation((0, 0, 0)).validate(cell)
    with pytest.raises(ValueError, match="overallocation"):
        SliceAllocation((30, 20, 10)).validate(cell)
    with pytest.raises(ValueError, match="sum to one"):
        allocate_shares(cell, (0.2, 0.2, 0.2))


def test_invalid_physical_configuration_fails_closed():
    for config in [
        {"cell_id": "bs", "bandwidth_mhz": 250, "scs_khz": 30},
        {"cell_id": "bs", "bandwidth_mhz": 20, "scs_khz": 30, "reserved_prbs": 51},
        {"cell_id": "bs", "bandwidth_mhz": True, "scs_khz": 30},
    ]:
        with pytest.raises(ValueError):
            NRCellCapacity.from_config(config)


def test_explicit_commag_confirmation_is_mandatory(tmp_path):
    path = tmp_path / "config.json"
    config = {"cells": {"bs1": {
        "cell_id": "bs1", "bandwidth_mhz": 20, "scs_khz": 30,
    }}}
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="explicitly confirm"):
        commag_physical_budget_config(str(path))
    config["confirmed_commag_prb_units"] = True
    path.write_text(json.dumps(config))
    report = commag_physical_budget_config(str(path))
    assert report["budget_by_base_station"] == {"bs1": 51}
    assert report["physical_capacity_confirmed"]


def test_newer_bwp_scope_retained():
    budget = CellBudget.from_handshake({
        "carrier_id": "c", "bwp_id": "b", "total_prbs": 52,
        "reserved_prbs": 4, "numerology": 1, "source": "nr_phy_observed",
    }, "ns3-5glena")
    assert budget.usable_prbs == 48
    JointAction((16, 16, 16)).validate(budget)
    with pytest.raises(ValueError, match="exceeds"):
        JointAction((30, 20, 10)).validate(budget)
