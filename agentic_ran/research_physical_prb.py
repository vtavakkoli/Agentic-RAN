"""Physical NR cell PRB capacity and explicit three-slice allocation actions.

No capacities are inferred from historical COMMAG allocations. The user must
provide a deployment-level numerology and carrier bandwidth (or explicit
validated physical NRB). Table 5.3.2-1, TS 38.104, FR1. All values refer
to *one cell, one carrier, one numerology and one scheduling interval*.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

SLICES = ("eMBB", "mMTC", "URLLC")

# NR FR1 transmission bandwidth configuration (N_RB), TS 38.104 Table 5.3.2-1.
# Explicitly reject unknown combinations rather than estimating.
FR1_NRB = {
    15: {5: 25, 10: 52, 15: 79, 20: 106, 25: 133, 30: 160, 40: 216, 50: 270},
    30: {5: 11, 10: 24, 15: 38, 20: 51, 25: 65, 30: 78, 40: 106,
         50: 133, 60: 162, 70: 189, 80: 217, 90: 245, 100: 273},
    60: {10: 11, 15: 18, 20: 24, 25: 31, 30: 38, 40: 51, 50: 65,
         60: 79, 70: 93, 80: 107, 90: 121, 100: 135},
}


@dataclass(frozen=True)
class NRCellCapacity:
    cell_id: str
    bandwidth_mhz: int
    scs_khz: int
    reserved_prbs: int = 0
    max_scheduled_prbs: int | None = None

    @property
    def carrier_prbs(self) -> int:
        value = FR1_NRB.get(self.scs_khz, {}).get(self.bandwidth_mhz)
        if value is None:
            raise ValueError("unsupported FR1 carrier bandwidth/SCS combination")
        return value

    @property
    def available_prbs(self) -> int:
        capacity = self.carrier_prbs
        if not isinstance(self.reserved_prbs, int) or not 0 <= self.reserved_prbs < capacity:
            raise ValueError("reserved_prbs must be a physical nonnegative count less than carrier NRBs")
        free = capacity - self.reserved_prbs
        if self.max_scheduled_prbs is not None:
            if not isinstance(self.max_scheduled_prbs, int) or not 0 < self.max_scheduled_prbs <= capacity:
                raise ValueError("max_scheduled_prbs must be within physical carrier capacity")
            free = min(free, self.max_scheduled_prbs)
        return free

    @classmethod
    def from_config(cls, data: Mapping[str, object]) -> "NRCellCapacity":
        required = {"cell_id", "bandwidth_mhz", "scs_khz"}
        if not required.issubset(data):
            raise ValueError(f"missing physical cell configuration {sorted(required - set(data))}")
        obj = cls(
            cell_id=str(data["cell_id"]),
            bandwidth_mhz=int(data["bandwidth_mhz"]),
            scs_khz=int(data["scs_khz"]),
            reserved_prbs=int(data.get("reserved_prbs", 0)),
            max_scheduled_prbs=(
                int(data["max_scheduled_prbs"])
                if data.get("max_scheduled_prbs") is not None else None
            ),
        )
        if not obj.cell_id.strip():
            raise ValueError("empty cell ID")
        _ = obj.available_prbs
        return obj


@dataclass(frozen=True)
class SliceAllocation:
    """Exactly one scheduler and one physical PRB count for each slice.

    A *single* common scheduler is used across slices unless a documented
    adapter supports per-slice scheduler selection. The explicit action is
    not a COMMAG logged-action token: no unsupported action reinterpretation.
    """
    prbs: tuple[int, int, int]
    scheduler: str = "proportional_fair"

    def __post_init__(self) -> None:
        if len(self.prbs) != 3 or any(type(x) is not int or x < 0 for x in self.prbs):
            raise ValueError("allocation must have three nonnegative integer physical PRB counts")
        if self.scheduler not in ("round_robin", "waterfilling", "proportional_fair"):
            raise ValueError("unsupported scheduler name")

    def validate(self, cell: NRCellCapacity, *, minimums: tuple[int, int, int] = (0, 0, 0)) -> None:
        if len(minimums) != 3 or any(type(x) is not int or x < 0 for x in minimums):
            raise ValueError("invalid slice minimums")
        if any(x < bound for x, bound in zip(self.prbs, minimums, strict=True)):
            raise ValueError("allocation violates per-slice minimum PRBs")
        if sum(self.prbs) > cell.available_prbs:
            raise ValueError(
                f"PRB overallocation: {sum(self.prbs)} requested, {cell.available_prbs} available"
            )

    def to_wire(self, cell: NRCellCapacity) -> dict[str, object]:
        self.validate(cell)
        return {
            "cell_id": cell.cell_id, "unit": "physical_NR_PRB_per_slot",
            "scheduler": self.scheduler,
            "slice_prbs": dict(zip(SLICES, self.prbs, strict=True)),
            "total_prbs": sum(self.prbs),
            "carrier_prbs": cell.carrier_prbs,
            "available_prbs": cell.available_prbs,
        }


def allocate_shares(
    cell: NRCellCapacity,
    shares: tuple[float, float, float],
    *,
    minimums: tuple[int, int, int] = (0, 0, 0),
    scheduler: str = "proportional_fair",
) -> SliceAllocation:
    """Largest remainder method with deterministic ties and exact physical bound."""
    if len(shares) != 3:
        raise ValueError("need three slice shares")
    import math
    if any(not math.isfinite(float(x)) or x < 0 for x in shares):
        raise ValueError("shares must be finite and nonnegative")
    if not math.isclose(sum(shares), 1.0, abs_tol=1e-7):
        raise ValueError("slice shares must sum to one")
    if len(minimums) != 3 or any(type(x) is not int or x < 0 for x in minimums):
        raise ValueError("invalid slice minimums")
    remaining = cell.available_prbs - sum(minimums)
    if remaining < 0:
        raise ValueError("slice minimums exceed physical cell capacity")
    fractional = [x * remaining for x in shares]
    base = [math.floor(x) for x in fractional]
    residual = remaining - sum(base)
    order = sorted(range(3), key=lambda k: (-(fractional[k] - base[k]), k))
    for index in order[:residual]:
        base[index] += 1
    result = SliceAllocation(tuple(int(a + b) for a, b in zip(base, minimums, strict=True)), scheduler)
    result.validate(cell, minimums=minimums)
    return result
