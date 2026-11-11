"""Action-dependent ns-3/5G-LENA or Colosseum JSON-lines bridge.

This is a tested *protocol client*, not an ns-3 build or Colosseum account.
A separately configured external process must implement reset/step and
actually apply physical per-slot PRBs before returning next observations.
No synthetic outcomes or static COMMAG replay are used by this runner.

Wire framing: exactly one JSON request and one JSON response per line.
reset -> {type:"observation",state:[9 finite numbers],...}
step  -> {type:"transition",next_state:[9...],reward:finite,
          sla_violations:[3 bools],done:bool,applied_allocation:{...}}
The adapter rejects missing actuation receipts and non-changing simulator
step sequence numbers. With no configured external executable it fails closed.
"""
from __future__ import annotations

import json
import math
import subprocess
import threading
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from agentic_ran.research_physical_prb import NRCellCapacity, SliceAllocation

SCHEMA_VERSION = "agentic-ran-physical-prb-v1"


def _state(values: Any) -> np.ndarray:
    if not isinstance(values, list) or len(values) != 9:
        raise ValueError("external environment must return 9-element RAN state")
    state = np.asarray(values, dtype=float)
    if not np.isfinite(state).all():
        raise ValueError("nonfinite external RAN state")
    if np.any(state[:3] < 0) or np.any(state[3:6] < 0) or np.any(state[6:9] <= 0):
        raise ValueError("invalid queue/demand/channel observations")
    return state


def _require_bool_list(value: Any) -> tuple[bool, bool, bool]:
    if not isinstance(value, list) or len(value) != 3 or any(type(x) is not bool for x in value):
        raise ValueError("expected three explicit boolean SLA indicators")
    return tuple(value)


@dataclass(frozen=True)
class StepResult:
    next_state: np.ndarray
    reward: float
    sla_violations: tuple[bool, bool, bool]
    done: bool
    kpis: dict[str, Any]


class ExternalRANSession:
    """Explicit command allowlisting is the caller's responsibility.

    Calls a locally installed ns-3/Colosseum bridge via argv (shell=False);
    the subprocess is persistent across steps so its state evolves under
    the actual chosen actions. Hard timeouts terminate stalled processes.
    """
    def __init__(
        self, command: Sequence[str], cell: NRCellCapacity, *,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not command or any(not isinstance(arg, str) or not arg for arg in command):
            raise ValueError("explicit executable/argv required")
        if not 0.1 <= timeout_seconds <= 300:
            raise ValueError("timeout outside allowed range")
        self.command = list(command)
        self.cell = cell
        self.timeout_seconds = timeout_seconds
        self.process: subprocess.Popen[str] | None = None
        self._sequence = 0

    def __enter__(self) -> "ExternalRANSession":
        self.process = subprocess.Popen(
            self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1,
            shell=False,
        )
        return self

    def __exit__(self, *_: object) -> None:
        if self.process is not None:
            self.process.terminate() if self.process.poll() is None else None
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
            self.process = None

    def _exchange(self, payload: dict[str, Any]) -> dict[str, Any]:
        process = self.process
        if process is None or process.stdin is None or process.stdout is None:
            raise RuntimeError("external RAN session must be started")
        if process.poll() is not None:
            raise RuntimeError("external RAN simulator exited before command")
        message = json.dumps(payload, allow_nan=False, separators=(",", ":"))
        if len(message) > 65536:
            raise ValueError("external simulator request too large")
        process.stdin.write(message + "\n")
        process.stdin.flush()
        response: list[str] = []
        error: list[BaseException] = []
        def read_line() -> None:
            try:
                response.append(process.stdout.readline(65537))
            except BaseException as exc:
                error.append(exc)
        worker = threading.Thread(target=read_line, daemon=True)
        worker.start()
        worker.join(timeout=self.timeout_seconds)
        if worker.is_alive():
            process.kill()
            process.wait(timeout=2)
            raise TimeoutError("external RAN bridge response timed out")
        if error:
            raise RuntimeError("external RAN bridge read failed") from error[0]
        if not response or not response[0] or len(response[0]) > 65536:
            raise RuntimeError("empty, truncated or oversized bridge response")
        result = json.loads(response[0])
        if not isinstance(result, dict) or result.get("schema") != SCHEMA_VERSION:
            raise ValueError("unsupported external bridge schema")
        if result.get("error"):
            raise RuntimeError(f"external RAN bridge rejected request: {result['error']}")
        return result

    def reset(self, *, scenario: str, seed: int, steps: int) -> np.ndarray:
        if not scenario or seed < 0 or steps <= 0:
            raise ValueError("invalid reset parameters")
        self._sequence = 0
        result = self._exchange({
            "schema": SCHEMA_VERSION, "op": "reset",
            "scenario": scenario, "seed": seed, "steps": steps,
            "physical_cell": {
                "cell_id": self.cell.cell_id,
                "bandwidth_mhz": self.cell.bandwidth_mhz,
                "scs_khz": self.cell.scs_khz,
                "reserved_prbs": self.cell.reserved_prbs,
                "available_prbs": self.cell.available_prbs,
            },
        })
        if result.get("type") != "observation" or result.get("step_index") != 0:
            raise ValueError("reset did not acknowledge fresh episode")
        return _state(result.get("state"))

    def step(self, allocation: SliceAllocation) -> StepResult:
        allocation.validate(self.cell)
        self._sequence += 1
        wire = allocation.to_wire(self.cell)
        result = self._exchange({
            "schema": SCHEMA_VERSION, "op": "step",
            "step_index": self._sequence, "allocation": wire,
        })
        if result.get("type") != "transition" or result.get("step_index") != self._sequence:
            raise ValueError("simulator returned stale/misordered step response")
        # Actuation acknowledgement is required: no silently simulated
        # action-effect estimates are accepted as executed outcomes.
        receipt = result.get("applied_allocation")
        if not isinstance(receipt, dict) or receipt != wire:
            raise ValueError("simulator did not confirm exact applied physical action")
        if type(result.get("done")) is not bool:
            raise ValueError("step requires an explicit done boolean")
        reward = result.get("reward")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
            raise ValueError("missing or nonfinite realized reward")
        kpis = result.get("kpis")
        if not isinstance(kpis, dict) or not all(
            key in kpis for key in ("throughput_mbps", "latency_ms")
        ):
            raise ValueError("missing measured throughput/latency fields")
        for key in ("throughput_mbps", "latency_ms"):
            values = kpis[key]
            if not isinstance(values, list) or len(values) != 3 or not all(
                isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x >= 0
                for x in values
            ):
                raise ValueError(f"invalid per-slice measured {key}")
        return StepResult(
            _state(result.get("next_state")), float(reward),
            _require_bool_list(result.get("sla_violations")),
            result["done"], kpis,
        )


Policy = Callable[[np.ndarray, SliceAllocation, int], SliceAllocation]


def run_realized_episode(
    command: Sequence[str], cell: NRCellCapacity,
    policy: Policy, *, scenario: str, seed: int,
    steps: int = 24, initial: SliceAllocation | None = None,
) -> tuple[dict[str, float | int | str], list[dict[str, Any]]]:
    """Invoke one external bridge per policy/seed, with a fresh reset.

    Call twice with the SAME seed/scenario and frozen original and candidate
    policies to obtain a matched pair of realized outcomes. The external
    backend must ensure deterministic exogenous disturbances under a seed.
    """
    initial = initial or SliceAllocation((0, 0, 0))
    initial.validate(cell)
    evidence: list[dict[str, Any]] = []
    total = 0.0
    violations = np.zeros(3, dtype=float)
    throughput = np.zeros(3)
    latency = np.zeros(3)
    switches = 0
    with ExternalRANSession(command, cell) as session:
        state = session.reset(scenario=scenario, seed=seed, steps=steps)
        previous = initial
        for index in range(steps):
            selected = policy(state.copy(), previous, index)
            if not isinstance(selected, SliceAllocation):
                raise TypeError("controller must return physical SliceAllocation")
            selected.validate(cell)
            observed = session.step(selected)
            switches += int(selected != previous)
            total += observed.reward
            violations += observed.sla_violations
            throughput += np.asarray(observed.kpis["throughput_mbps"])
            latency += np.asarray(observed.kpis["latency_ms"])
            evidence.append({
                "scenario": scenario, "seed": seed, "t": index,
                "state": state.tolist(), "action": selected.to_wire(cell),
                "next_state": observed.next_state.tolist(),
                "reward": observed.reward,
                "sla_violations": list(observed.sla_violations),
                "kpis": observed.kpis, "done": observed.done,
                "backend": "external_executed_action_acknowledged",
            })
            state, previous = observed.next_state, selected
            if observed.done:
                break
    n = len(evidence)
    return {
        "scenario": scenario, "seed": seed, "steps": n,
        "mean_realized_reward": total / n,
        "sla_eMBB": float(violations[0] / n),
        "sla_mMTC": float(violations[1] / n),
        "sla_URLLC": float(violations[2] / n),
        "mean_throughput_mbps": float(throughput.sum() / n),
        "mean_latency_ms": float(latency.mean() / n),
        "switch_rate": switches / n,
        "backend": "external_bridge_not_verified_as_ns3_or_colosseum_by_client",
    }, evidence
