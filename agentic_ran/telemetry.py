"""Telemetry providers for synthetic, replay, srsRAN JSON metrics, E2 KPM, and Prometheus sources."""

from __future__ import annotations

import asyncio
import csv
import json
import math
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentic_ran.domain import NetworkObservation


class TelemetryProvider(ABC):
    @abstractmethod
    async def observe(self) -> NetworkObservation:
        raise NotImplementedError


class SyntheticTelemetryProvider(TelemetryProvider):
    """Deterministic live-like source for CI, demos, and offline control-loop tests."""

    def __init__(self, cell_id: str = "sim-cell-001", slice_type: str = "eMBB", period_seconds: float = 1.0):
        self.cell_id = cell_id
        self.slice_type = slice_type
        self.period_seconds = period_seconds
        self._step = 0

    async def observe(self) -> NetworkObservation:
        if self._step:
            await asyncio.sleep(self.period_seconds)
        phase = self._step / 8.0
        load = 0.62 + 0.25 * math.sin(phase)
        users = int(150 + 85 * (1 + math.sin(phase * 0.8)) / 2)
        latency = 24 + 50 * max(0.0, load - 0.72)
        downlink = max(20.0, 150.0 * (1.12 - load * 0.35))
        demand = 110 + 85 * (1 + math.sin(phase * 0.9)) / 2
        observation = NetworkObservation(
            timestamp=datetime.now(timezone.utc),
            cell_id=self.cell_id,
            slice_type=self.slice_type,
            prb_utilization=min(1.15, max(0.08, load)),
            active_users=users,
            downlink_mbps=downlink,
            uplink_mbps=downlink * 0.18,
            latency_ms=latency,
            jitter_ms=max(1.0, latency * 0.16),
            packet_loss_pct=max(0.02, (load - 0.68) * 4.0),
            throughput_demand_mbps=demand,
            energy_load=min(1.1, 0.22 + load * 0.72),
            handover_failure_pct=max(0.2, 0.6 + (load - 0.8) * 2.0),
            rsrp_dbm=-94.0,
            sinr_db=15.0 - max(0.0, load - 0.75) * 20.0,
            source="synthetic",
            sequence=self._step,
        )
        self._step += 1
        return observation


class CSVReplayTelemetryProvider(TelemetryProvider):
    def __init__(self, path: Path | str, loop: bool = True, period_seconds: float = 0.0):
        self.path = Path(path)
        self.loop = loop
        self.period_seconds = period_seconds
        with self.path.open(newline="", encoding="utf-8") as handle:
            self._rows = list(csv.DictReader(handle))
        if not self._rows:
            raise ValueError("telemetry replay CSV is empty")
        self._index = 0

    async def observe(self) -> NetworkObservation:
        if self._index and self.period_seconds:
            await asyncio.sleep(self.period_seconds)
        if self._index >= len(self._rows):
            if not self.loop:
                raise StopAsyncIteration
            self._index = 0
        row = dict(self._rows[self._index])
        self._index += 1
        row.pop("policy_label", None)
        row["source"] = "csv-replay"
        row["sequence"] = self._index - 1
        return NetworkObservation(**row)


def _collect_numeric(payload: Any, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(payload, dict):
        name = payload.get("name") or payload.get("metric") or payload.get("metric_name")
        value = payload.get("value")
        if name is not None and isinstance(value, (int, float)):
            result[str(name)] = float(value)
        for key, item in payload.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(item, (dict, list)):
                result.update(_collect_numeric(item, child_prefix))
            elif isinstance(item, (int, float)):
                result[child_prefix] = float(item)
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            result.update(_collect_numeric(item, f"{prefix}[{index}]"))
    return result


class SrsRANMetricMapper:
    """Maps current srsRAN JSON/E2 metric names into the stable Agentic-RAN schema.

    JSON layouts evolve; matching is therefore alias-based and tolerant of nested keys.
    """

    ALIASES: dict[str, tuple[str, ...]] = {
        "downlink_mbps": ("DRB.UEThpDl", "dl_brate", "dl_bitrate", "downlink_mbps"),
        "uplink_mbps": ("DRB.UEThpUl", "ul_brate", "ul_bitrate", "uplink_mbps"),
        "packet_loss_pct": ("DRB.RlcPacketDropRateDl", "rlc_drop_rate", "packet_loss_pct"),
        "rsrp_dbm": ("RSRP", "rsrp", "rsrp_dbm"),
        "sinr_db": ("SINR", "sinr", "sinr_db", "cqi"),
        "prb_utilization": ("prb_utilization", "dl_prb_usage", "dl_prb", "prb"),
        "active_users": ("active_users", "nof_ues", "ue_count"),
        "latency_ms": ("latency_ms", "rlc_latency_ms", "delay_ms"),
        "jitter_ms": ("jitter_ms",),
        "energy_load": ("energy_load", "cpu_load", "ru_power_normalized"),
        "handover_failure_pct": ("handover_failure_pct", "ho_failure_rate"),
        "throughput_demand_mbps": ("throughput_demand_mbps", "demand_mbps"),
    }

    def __init__(self, defaults: dict[str, float] | None = None):
        self.defaults = {
            "downlink_mbps": 0.0,
            "uplink_mbps": 0.0,
            "packet_loss_pct": 0.0,
            "rsrp_dbm": -95.0,
            "sinr_db": 12.0,
            "prb_utilization": 0.5,
            "active_users": 1.0,
            "latency_ms": 30.0,
            "jitter_ms": 5.0,
            "energy_load": 0.55,
            "handover_failure_pct": 0.5,
            "throughput_demand_mbps": 1.0,
            **(defaults or {}),
        }

    @staticmethod
    def _find(values: dict[str, float], aliases: tuple[str, ...]) -> float | None:
        for alias in aliases:
            alias_lower = alias.lower()
            for key, value in values.items():
                key_lower = key.lower()
                if key_lower == alias_lower or key_lower.endswith(f".{alias_lower}") or alias_lower in key_lower:
                    return float(value)
        return None

    def map(self, payload: dict[str, Any], cell_id: str = "srsran-cell", slice_type: str = "eMBB") -> NetworkObservation:
        values = _collect_numeric(payload)
        mapped: dict[str, float] = {}
        matched = 0
        for target, aliases in self.ALIASES.items():
            value = self._find(values, aliases)
            if value is not None:
                matched += 1
                mapped[target] = value
            else:
                mapped[target] = self.defaults[target]

        for key in ("downlink_mbps", "uplink_mbps"):
            if mapped[key] > 10_000:
                mapped[key] /= 1_000.0
        if mapped["prb_utilization"] > 1.5:
            mapped["prb_utilization"] /= 100.0
        if mapped["sinr_db"] > 40.0:
            mapped["sinr_db"] = min(30.0, mapped["sinr_db"] * 1.5 - 10.0)

        demand = max(mapped["throughput_demand_mbps"], mapped["downlink_mbps"] * 1.08)
        return NetworkObservation(
            timestamp=datetime.now(timezone.utc),
            cell_id=cell_id,
            slice_type=slice_type,
            prb_utilization=min(1.5, max(0.0, mapped["prb_utilization"])),
            active_users=max(0, int(mapped["active_users"])),
            downlink_mbps=max(0.0, mapped["downlink_mbps"]),
            uplink_mbps=max(0.0, mapped["uplink_mbps"]),
            latency_ms=max(0.0, mapped["latency_ms"]),
            jitter_ms=max(0.0, mapped["jitter_ms"]),
            packet_loss_pct=min(100.0, max(0.0, mapped["packet_loss_pct"])),
            throughput_demand_mbps=max(0.0, demand),
            energy_load=min(1.5, max(0.0, mapped["energy_load"])),
            handover_failure_pct=min(100.0, max(0.0, mapped["handover_failure_pct"])),
            rsrp_dbm=min(-20.0, max(-160.0, mapped["rsrp_dbm"])),
            sinr_db=min(80.0, max(-30.0, mapped["sinr_db"])),
            source="srsran",
            telemetry_completeness=min(1.0, matched / len(self.ALIASES)),
        )


class SrsRANWebSocketProvider(TelemetryProvider):
    """Live srsRAN JSON metrics provider using the documented metrics subscription command."""

    def __init__(self, url: str, mapper: SrsRANMetricMapper | None = None, cell_id: str = "srsran-cell"):
        self.url = url
        self.mapper = mapper or SrsRANMetricMapper()
        self.cell_id = cell_id
        self._ws: Any = None

    def _receive_blocking(self) -> dict[str, Any]:
        try:
            import websocket  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("Install agentic-ran[oran] to use live srsRAN WebSocket telemetry") from exc
        if self._ws is None:
            self._ws = websocket.create_connection(self.url, timeout=5)
            self._ws.send(json.dumps({"cmd": "metrics_subscribe"}))
        message = self._ws.recv()
        return json.loads(message)

    async def observe(self) -> NetworkObservation:
        payload = await asyncio.to_thread(self._receive_blocking)
        return self.mapper.map(payload, cell_id=self.cell_id)


class E2KPMProvider(TelemetryProvider):
    """Transport-neutral provider for already decoded E2SM-KPM indication dictionaries."""

    def __init__(self, source: AsyncIterator[dict[str, Any]], mapper: SrsRANMetricMapper | None = None):
        self.source = source
        self.mapper = mapper or SrsRANMetricMapper()

    async def observe(self) -> NetworkObservation:
        payload = await anext(self.source)
        cell_id = str(payload.get("cell_id", payload.get("ran_node", "e2-cell")))
        slice_type = str(payload.get("slice_type", "eMBB"))
        return self.mapper.map(payload, cell_id=cell_id, slice_type=slice_type)


class PrometheusTelemetryProvider(TelemetryProvider):
    def __init__(self, url: str, mapping: dict[str, str], base: Callable[[dict[str, float]], NetworkObservation]):
        self.url = url
        self.mapping = mapping
        self.base = base

    @staticmethod
    def _parse(text: str) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            name = parts[0].split("{")[0]
            try:
                metrics[name] = float(parts[-1])
            except ValueError:
                continue
        return metrics

    async def observe(self) -> NetworkObservation:
        def _fetch() -> str:
            with urllib.request.urlopen(self.url, timeout=3) as response:
                return response.read().decode("utf-8")

        values = self._parse(await asyncio.to_thread(_fetch))
        projected = {target: values[source] for target, source in self.mapping.items() if source in values}
        return self.base(projected)
