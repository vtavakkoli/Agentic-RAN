"""Offline training and matched external closed-loop evaluation CLI.

Requires a separately installed external ns-3/5G-LENA or Colosseum
bridge; this repository does NOT bundle or provision either platform.
The baseline is an explicit importable frozen policy factory, never an
unlabeled replacement for the original Agentic-RAN implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import shlex
from typing import Any

import joblib
import numpy as np

from agentic_ran.research_physical_prb import NRCellCapacity, SliceAllocation
from agentic_ran.research_realized_advantage import (
    action_library, compare_matched_external, fit_advantages, logged_from_external,
)


def load_frozen_policy(spec: str, cell: NRCellCapacity) -> tuple[Any, dict[str, str]]:
    if ":" not in spec:
        raise ValueError("frozen baseline must be an importable module:function factory")
    module_name, function_name = spec.split(":", 1)
    if not module_name or not function_name or function_name.startswith("_"):
        raise ValueError("invalid frozen baseline entrypoint")
    factory = getattr(importlib.import_module(module_name), function_name)
    if not callable(factory):
        raise TypeError("baseline entrypoint must be callable")
    policy = factory(cell)
    if not callable(policy):
        raise TypeError("baseline factory must return policy(state,previous,t)")
    module = inspect.getmodule(factory)
    source = Path(module.__file__) if module is not None and module.__file__ else None
    if source is None or not source.is_file():
        raise ValueError("frozen baseline needs a readable module source file")
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    return policy, {"entrypoint": spec, "source_file": str(source), "source_sha256": sha}


def _read_config(path: Path) -> tuple[NRCellCapacity, list[str], tuple[str, ...],
                                      tuple[int, ...], tuple[int, ...], tuple[int, ...], int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cell = NRCellCapacity.from_config(raw["cell"])
    command = raw.get("bridge_command", [])
    if not isinstance(command, list) or not command:
        raise ValueError("explicit local bridge_command argv required; ns-3 is not auto-installed")
    if not all(isinstance(x, str) and x for x in command):
        raise ValueError("invalid bridge command arguments")
    scenarios = tuple(str(x) for x in raw["scenarios"])
    if not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("scenarios must be nonempty and unique")
    train = tuple(map(int, raw["train_seeds"]))
    validation = tuple(map(int, raw["validation_seeds"]))
    test = tuple(map(int, raw["test_seeds"]))
    if len(train) < 4 or len(validation) < 2 or len(test) < 2:
        raise ValueError("need at least four training and two each validation/test seeds")
    if len(set(train)) != len(train) or len(set(validation)) != len(validation) or len(set(test)) != len(test):
        raise ValueError("duplicate seeds within a split")
    if set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test):
        raise ValueError("train/validation/final-test seeds must be disjoint")
    steps = int(raw.get("steps", 24))
    if not 3 <= steps <= 10000:
        raise ValueError("steps outside allowed range")
    return cell, command, scenarios, train, validation, test, steps


def run(config: Path, original_factory: str, destination: Path) -> dict[str, Any]:
    cell, command, scenarios, train, validation, test, steps = _read_config(config)
    original, baseline_provenance = load_frozen_policy(original_factory, cell)
    # TRAIN ONLY: randomized performed actions, explicit receipt and measured
    # next state and utility. Fails if only static COMMAG is provided.
    records = logged_from_external(
        command, cell, scenarios=scenarios, seeds=train, steps=steps,
    )
    choices = action_library(cell)
    model = fit_advantages(records, choices)
    destination.mkdir(parents=True, exist_ok=True)
    # Freeze training and original baseline BEFORE observing final-test data.
    joblib.dump(model, destination / "advantage_models.joblib")
    validation_rows = []
    best_margin = None
    for margin in (0.0, 0.005, 0.01, 0.02):
        policy = lambda state, previous, t, m=margin: model.choose(
            state, original(state, previous, t), minimum_advantage=m,
        )
        rows, stats = compare_matched_external(
            command, cell, original, policy,
            scenarios=scenarios, seeds=validation, steps=steps,
        )
        sla_ok = all(
            sum(row[f"candidate_sla_{slice_name}"] for row in rows) <=
            sum(row[f"baseline_sla_{slice_name}"] for row in rows) + 1e-9
            for slice_name in ("eMBB", "mMTC", "URLLC")
        )
        validation_rows.append({
            "margin": margin, "sla_non_regression": sla_ok,
            **stats,
        })
    viable = [
        row for row in validation_rows
        if row["sla_non_regression"] and row["paired_mean_realized_delta"] >= .002
    ]
    if viable:
        chosen = max(viable, key=lambda row: (row["paired_mean_realized_delta"], row["margin"]))
        best_margin = float(chosen["margin"])
        selection_reason = "validation_gain_with_nonregressing_realized_slice_SLAs"
    else:
        selection_reason = "retain_frozen_original_no_validation_gain"
    selected = (
        original if best_margin is None
        else lambda state, previous, t: model.choose(
            state, original(state, previous, t),
            minimum_advantage=best_margin,
        )
    )
    # Use identical held-out seeds and scenarios for original and chosen.
    final_rows, final = compare_matched_external(
        command, cell, original, selected, scenarios=scenarios, seeds=test, steps=steps,
    )
    result = {
        "protocol": "agentic-ran-physical-prb-v1",
        "external_backend_command": command,
        "backend_identity_verified_by_repository": False,
        "external_backend_executed_actions_acknowledged": True,
        "physical_cell": {
            "id": cell.cell_id, "bandwidth_mhz": cell.bandwidth_mhz,
            "scs_khz": cell.scs_khz, "carrier_prbs": cell.carrier_prbs,
            "available_prbs": cell.available_prbs,
        },
        "baseline_identity": baseline_provenance,
        "baseline_factory_explicitly_supplied": True,
        "randomized_training_records": len(records),
        "randomized_training_seeds": list(train),
        "validation_seeds": list(validation),
        "heldout_final_test_seeds": list(test),
        "validation_profiles": validation_rows,
        "selected_margin": best_margin,
        "selection_reason": selection_reason,
        "final_test": final,
        "final_pairs": final_rows,
        "claims": [
            "Only externally acknowledged and observed transitions enter training/evaluation.",
            "External backend identity and fidelity require independent verification.",
            "The baseline is whatever frozen policy the caller supplied; inspect SHA before naming it original Agentic-RAN.",
            "No claims of superior throughput or service without positive realized held-out effects and uncertainty analysis.",
        ],
    }
    (destination / "physical_joint_advantage_summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    (destination / "physical_joint_training_records.json").write_text(
        json.dumps(records, allow_nan=False) + "\n", encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--original-factory", required=True,
        help="module:function -> frozen original RAN controller with physical action semantics",
    )
    parser.add_argument("--output", type=Path, default=Path("results/physical-jagc"))
    args = parser.parse_args()
    result = run(args.config, args.original_factory, args.output)
    print(json.dumps({
        "selection_reason": result["selection_reason"],
        "selected_margin": result["selected_margin"],
        "final_test": result["final_test"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
