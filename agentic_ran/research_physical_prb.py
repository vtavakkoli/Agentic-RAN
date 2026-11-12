"""Physical, BWP-scoped PRB accounting; never infer cell capacity from COMMAG labels."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SLICES = ("eMBB", "mMTC", "URLLC")
SCHEDULERS = ("round_robin", "waterfilling", "proportional_fair")


@dataclass(frozen=True)
class CellBudget:
    carrier_id: str
    bwp_id: str
    total_prbs: int
    reserved_prbs: int
    numerology: int
    source: str
    backend: str

    @classmethod
    def from_handshake(cls, data: dict[str, Any], backend: str) -> "CellBudget":
        if not isinstance(data, dict):
            raise ValueError("simulator must report a physical BWP capacity object")
        fields = ("carrier_id", "bwp_id", "total_prbs", "reserved_prbs", "numerology", "source")
        missing = sorted(set(fields) - set(data))
        if missing:
            raise ValueError(f"missing BWP capacity fields: {missing}")
        for key in ("total_prbs", "reserved_prbs", "numerology"):
            if type(data[key]) is not int:
                raise ValueError(f"{key} must be an integer physical unit")
        budget = cls(*(data[key] for key in fields), backend=str(backend))
        if not all((budget.carrier_id, budget.bwp_id, budget.source, budget.backend)):
            raise ValueError("empty carrier/BWP/source/backend identifier")
        if not (1 <= budget.total_prbs <= 275 and 0 <= budget.reserved_prbs < budget.total_prbs):
            raise ValueError("invalid NR BWP PRB capacity/reservation")
        if not 0 <= budget.numerology <= 5:
            raise ValueError("invalid NR numerology")
        if budget.source in ("COMMAG_trace", "historical_allocation_quantile", "unknown"):
            raise ValueError("historical slice allocations cannot establish physical cell budget")
        return budget

    @property
    def usable_prbs(self) -> int:
        return self.total_prbs - self.reserved_prbs

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "usable_prbs": self.usable_prbs,
                "backend_reports_phy_observed_capacity":
                self.backend in ("ns3-5glena", "colosseum") and self.source == "nr_phy_observed"}


@dataclass(frozen=True)
class JointAction:
    prbs: tuple[int, int, int]
    schedulers: tuple[str, str, str] = ("proportional_fair",) * 3

    def validate(self, budget: CellBudget, minimum: tuple[int, int, int] = (1, 1, 1)) -> None:
        if len(self.prbs) != 3 or len(self.schedulers) != 3:
            raise ValueError("exactly three aligned slice actions required")
        if any(type(x) is not int for x in self.prbs):
            raise ValueError("PRB allocations must use integer PHY resource units")
        if any(x < lo for x, lo in zip(self.prbs, minimum, strict=True)):
            raise ValueError("slice is below physical PRB minimum")
        if sum(self.prbs) > budget.usable_prbs:
            raise ValueError("joint PRB allocation exceeds BWP available capacity")
        if any(s not in SCHEDULERS for s in self.schedulers):
            raise ValueError("unknown scheduler")

    def to_wire(self, budget: CellBudget) -> dict[str, Any]:
        self.validate(budget)
        return {"carrier_id": budget.carrier_id, "bwp_id": budget.bwp_id,
                "slices": {s: {"prbs": self.prbs[i], "scheduler": self.schedulers[i]}
                           for i, s in enumerate(SLICES)}}

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> "JointAction":
        return cls(tuple(data["slices"][s]["prbs"] for s in SLICES),
                   tuple(data["slices"][s]["scheduler"] for s in SLICES))


def project_reference_action(
    desired_prbs: tuple[int, int, int], budget: CellBudget,
    schedulers: tuple[str, str, str] = ("proportional_fair",) * 3,
    minimum: tuple[int, int, int] = (1, 1, 1),
) -> tuple[JointAction, bool]:
    """Exact nearest-squared-error projection; label as capacity-projected baseline."""
    if len(desired_prbs) != 3 or any(type(x) is not int or x < 0 for x in desired_prbs):
        raise ValueError("invalid legacy PRB requests")
    if sum(minimum) > budget.usable_prbs:
        raise ValueError("insufficient BWP capacity for minimum slice allocations")
    states = {0: (0, ())}
    for j in range(3):
        new = {}
        for used, (cost, prior) in states.items():
            for n in range(minimum[j], budget.usable_prbs - used + 1):
                key = used + n
                item = (cost + (n - desired_prbs[j]) ** 2, prior + (n,))
                if key not in new or item < new[key]:
                    new[key] = item
        states = new
    _, (_, chosen) = min(states.items(), key=lambda x: (x[1][0], -x[0]))
    action = JointAction(chosen, schedulers)
    action.validate(budget, minimum)
    return action, chosen != desired_prbs


def discrete_joint_catalog(
    budget: CellBudget, *, minimum: tuple[int, int, int] = (1, 1, 1),
) -> list[JointAction]:
    """Fixed candidate designs in actual integer PRBs, not historical proxy units."""
    fractions = (
        (.34, .33, .33), (.50, .20, .30), (.35, .20, .45),
        (.60, .20, .20), (.40, .35, .25), (.40, .15, .45),
        (.25, .25, .50), (.50, .30, .20), (.20, .40, .40),
        (.20, .20, .60), (.65, .15, .20), (.30, .45, .25),
        (.40, .40, .20), (.30, .20, .50), (.45, .30, .25),
        (.25, .45, .30), (.25, .15, .60), (.45, .15, .40),
    )
    remaining = budget.usable_prbs - sum(minimum)
    if remaining < 0:
        raise ValueError("insufficient physical BWP capacity")
    seen = set()
    options = []
    for weights in fractions:
        raw = [remaining * w for w in weights]
        counts = [int(x) for x in raw]
        n_extra = remaining - sum(counts)
        for i in sorted(range(3), key=lambda j: -(raw[j] - counts[j]))[:n_extra]:
            counts[i] += 1
        prbs = tuple(counts[i] + minimum[i] for i in range(3))
        if prbs not in seen:
            choice = JointAction(prbs)
            choice.validate(budget, minimum)
            options.append(choice)
            seen.add(prbs)
    return options


# Compatibility interface for the original physical-experiment research runner.
# This is a DISTINCT protocol from the newer BWP-scoped CellBudget/JointAction
# runner. Both share the same strict accounting principle: the total allocation
# must fit within an explicitly specified physical NR cell, never a COMMAG label.
#
# Keep these names until the original 'agentic-ran-physical-prb-v1' JSON-lines
# bridge and its independently tested evaluation CLI are fully migrated.
_FR1_PRBS: dict[tuple[int, int], int] = {
    # FR1 transmission bandwidth configurations (MHz, subcarrier spacing kHz),
    # from 3GPP TS 38.104. Values count the full configured carrier grid.
    (5, 15): 25, (5, 30): 11,
    (10, 15): 52, (10, 30): 24, (10, 60): 11,
    (15, 15): 79, (15, 30): 38, (15, 60): 18,
    (20, 15): 106, (20, 30): 51, (20, 60): 24,
    (25, 15): 133, (25, 30): 65, (25, 60): 31,
    (30, 15): 160, (30, 30): 78, (30, 60): 38,
    (40, 15): 216, (40, 30): 106, (40, 60): 51,
    (50, 15): 270, (50, 30): 133, (50, 60): 65,
    (60, 30): 162, (60, 60): 79,
    (70, 30): 189, (70, 60): 93,
    (80, 30): 217, (80, 60): 107,
    (90, 30): 245, (90, 60): 121,
    (100, 30): 273, (100, 60): 135,
}


@dataclass(frozen=True)
class NRCellCapacity:
    """Explicit FR1 carrier grid for the legacy executed-action protocol.

    A known carrier grid is not proof of usable, BWP-scoped, NR-PHY-observed
    capacity. The newer CellBudget protocol should be preferred for 5G-LENA
    evidence; this legacy class only restores its documented independent API.
    """

    cell_id: str
    bandwidth_mhz: int
    scs_khz: int
    reserved_prbs: int = 0

    @classmethod
    def from_config(cls, data: dict[str, Any]) -> "NRCellCapacity":
        if not isinstance(data, dict):
            raise ValueError("physical cell configuration must be an object")
        keys = ("cell_id", "bandwidth_mhz", "scs_khz")
        if any(k not in data for k in keys):
            raise ValueError("cell_id, bandwidth_mhz, scs_khz are required")
        obj = cls(*(data[k] for k in keys), reserved_prbs=data.get("reserved_prbs", 0))
        _ = obj.available_prbs
        return obj

    @property
    def carrier_prbs(self) -> int:
        if not isinstance(self.cell_id, str) or not self.cell_id.strip():
            raise ValueError("cell_id must be a nonempty string")
        if type(self.bandwidth_mhz) is not int or type(self.scs_khz) is not int:
            raise ValueError("FR1 bandwidth and SCS must be integer MHz/kHz")
        try:
            return _FR1_PRBS[(self.bandwidth_mhz, self.scs_khz)]
        except KeyError as exc:
            raise ValueError("unsupported FR1 bandwidth/SCS configuration") from exc

    @property
    def available_prbs(self) -> int:
        total = self.carrier_prbs
        if type(self.reserved_prbs) is not int or not (0 <= self.reserved_prbs < total):
            raise ValueError("reserved_prbs must be smaller than carrier_prbs")
        return total - self.reserved_prbs


@dataclass(frozen=True)
class SliceAllocation:
    """The original bridge uses a shared scheduler and integer PRB tuple."""

    prbs: tuple[int, int, int]
    scheduler: str = "proportional_fair"

    def validate(
        self, cell: NRCellCapacity, minimums: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        if len(self.prbs) != 3 or len(minimums) != 3:
            raise ValueError("exactly three slice PRB allocations are required")
        if any(type(x) is not int or x < 0 for x in self.prbs):
            raise ValueError("physical slice PRBs must be nonnegative integers")
        if any(type(x) is not int or x < 0 for x in minimums):
            raise ValueError("minimum PRB allocations must be nonnegative integers")
        if any(a < m for a, m in zip(self.prbs, minimums, strict=True)):
            raise ValueError("slice allocation below required minimum")
        if sum(self.prbs) > cell.available_prbs:
            raise ValueError("physical PRB overallocation exceeds available cell PRBs")
        if self.scheduler not in SCHEDULERS:
            raise ValueError("unsupported shared physical scheduler")

    def to_wire(self, cell: NRCellCapacity) -> dict[str, Any]:
        self.validate(cell)
        return {
            "cell_id": cell.cell_id,
            "available_prbs": cell.available_prbs,
            "slice_prbs": dict(zip(SLICES, self.prbs, strict=True)),
            "scheduler": self.scheduler,
        }


def allocate_shares(
    cell: NRCellCapacity,
    shares: tuple[float, float, float],
    *,
    minimums: tuple[int, int, int] = (1, 1, 1),
    scheduler: str = "proportional_fair",
) -> SliceAllocation:
    """Deterministically divide remaining physical PRBs by largest remainder."""
    import math

    if len(shares) != 3 or any(
        isinstance(x, bool) or not isinstance(x, (int, float))
        or not math.isfinite(x) or x < 0 for x in shares
    ) or not math.isclose(sum(shares), 1.0, abs_tol=1e-9, rel_tol=0):
        raise ValueError("physical allocation shares must sum to one")
    if len(minimums) != 3 or any(type(x) is not int or x < 0 for x in minimums):
        raise ValueError("invalid per-slice minimums")
    remaining = cell.available_prbs - sum(minimums)
    if remaining < 0:
        raise ValueError("physical cell cannot accommodate minimum slice shares")
    raw = [remaining * float(x) for x in shares]
    extra = [math.floor(x) for x in raw]
    remainder = remaining - sum(extra)
    for i in sorted(range(3), key=lambda j: (-(raw[j] - extra[j]), j))[:remainder]:
        extra[i] += 1
    result = SliceAllocation(
        tuple(minimums[i] + extra[i] for i in range(3)), scheduler,
    )
    result.validate(cell, minimums)
    return result


def commag_physical_budget_config(path: str) -> dict[str, Any]:
    """Reject unconfirmed COMMAG PRB units; do not estimate from historical logs."""
    import json
    from pathlib import Path

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("confirmed_commag_prb_units") is not True:
        raise ValueError(
            "operator must explicitly confirm physical COMMAG PRB units and carrier configuration"
        )
    cells = raw.get("cells")
    if not isinstance(cells, dict) or not cells:
        raise ValueError("at least one explicitly configured physical cell is required")
    budgets = {}
    for name, item in cells.items():
        if not isinstance(item, dict) or name != item.get("cell_id"):
            raise ValueError("cell identifiers do not match physical configurations")
        physical = NRCellCapacity.from_config(item)
        budgets[name] = physical.available_prbs
    return {
        "budget_by_base_station": budgets,
        "shared_scheduler_required": True,
        "physical_capacity_confirmed": True,
    }
