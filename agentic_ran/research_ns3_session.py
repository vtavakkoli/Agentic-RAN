"""Fail-closed JSON-lines bridge to an EXECUTING external ns-3/NR or Colosseum process.

The simulator itself owns state, PHY capacity and realized post-action outcomes.
The bundled event-demo does not claim 5G-LENA or Colosseum network validation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
from typing import Any, Sequence

from agentic_ran.research_physical_prb import CellBudget, JointAction, SLICES


class ExternalClosedLoopSession:
    """Persistent process: hello, reset, observation, step, transition."""

    def __init__(
        self, command: Sequence[str], *, require_real: bool = True,
        cwd: str | Path | None = None,
    ):
        if not command or any(not isinstance(x, str) or not x for x in command):
            raise ValueError("explicit trusted simulator executable argv required")
        self.process = subprocess.Popen(
            list(command), cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self.closed = False
        self.last_observation = None
        try:
            hello = self._read()
            if hello.get("type") != "hello" or hello.get("protocol") != "jagc-closed-loop-v1":
                raise RuntimeError("simulator did not advertise JAGC closed-loop protocol")
            self.backend = str(hello.get("backend", ""))
            self.real_actuation = hello.get("real_actuation") is True
            self.budget = CellBudget.from_handshake(hello.get("cell", {}), self.backend)
            if require_real and (
                self.backend not in ("ns3-5glena", "colosseum")
                or self.budget.source != "nr_phy_observed"
                or not self.real_actuation
            ):
                raise RuntimeError(
                    "refusing synthetic or unverified simulator in real mode; "
                    "require NR-PHY-derived BWP capacity and executed actions"
                )
        except BaseException:
            self.close()
            raise

    def _read(self) -> dict[str, Any]:
        assert self.process.stdout is not None
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("simulator closed protocol output or terminated")
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError("simulator stdout must contain only one JSON object per line") from exc
        if not isinstance(obj, dict):
            raise RuntimeError("simulator protocol response must be a JSON object")
        if obj.get("type") == "error":
            raise RuntimeError(f"external simulator error: {obj.get('reason')}")
        return obj

    def _send(self, payload: dict[str, Any]) -> None:
        if self.closed or self.process.poll() is not None:
            raise RuntimeError("simulator process unavailable")
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n")
        self.process.stdin.flush()

    @staticmethod
    def _validate_observation(obj: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(obj, dict) or obj.get("type") != "observation":
            raise RuntimeError("post-reset or post-action observation missing")
        if type(obj.get("step")) is not int or obj["step"] < 0:
            raise RuntimeError("invalid simulator step counter")
        slices = obj.get("slices")
        if not isinstance(slices, dict) or set(slices) != set(SLICES):
            raise RuntimeError("missing aligned eMBB/mMTC/URLLC slice observations")
        for name in SLICES:
            fields = slices[name].get("metrics")
            if not isinstance(fields, dict) or not fields:
                raise RuntimeError("missing measured slice KPI fields")
            if any(
                isinstance(v, bool) or not isinstance(v, (int, float))
                or not math.isfinite(v) for v in fields.values()
            ):
                raise RuntimeError("nonfinite simulator KPI field")
        return obj

    def reset(self, seed: int, scenario: str) -> dict[str, Any]:
        if type(seed) is not int or seed < 0 or not scenario:
            raise ValueError("reset requires deterministic seed and scenario")
        self._send({"type": "reset", "seed": seed, "scenario": scenario})
        obs = self._validate_observation(self._read())
        if obs["step"] != 0:
            raise RuntimeError("simulator failed to reset step counter")
        self.last_observation = obs
        return obs

    def step(self, action: JointAction) -> tuple[dict[str, Any], float, bool]:
        if self.last_observation is None:
            raise RuntimeError("reset simulator before issuing actions")
        self._send({"type": "step", "action": action.to_wire(self.budget)})
        response = self._read()
        if response.get("type") != "transition":
            raise RuntimeError("external simulator must respond with realized transition")
        reward = response.get("reward")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
            raise RuntimeError("missing or invalid realized reward")
        next_obs = self._validate_observation(response.get("observation"))
        if next_obs["step"] != self.last_observation["step"] + 1:
            raise RuntimeError("external simulator did not advance exactly one action step")
        if type(response.get("done")) is not bool:
            raise RuntimeError("missing explicit episode completion")
        self.last_observation = next_obs
        return next_obs, float(reward), response["done"]

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        p = self.process
        if p.poll() is None:
            try:
                if p.stdin:
                    p.stdin.close()
                p.terminate()
                p.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                p.kill()
                p.wait(timeout=3)
        if p.stdout:
            p.stdout.close()

    def __enter__(self) -> "ExternalClosedLoopSession":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
