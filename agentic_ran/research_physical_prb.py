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
