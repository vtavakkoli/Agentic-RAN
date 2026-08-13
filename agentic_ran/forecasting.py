"""Lightweight demand forecasting used as advisory context for planning."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from agentic_ran.domain import NetworkObservation


@dataclass(frozen=True, slots=True)
class Forecast:
    horizon_steps: int
    demand_mbps: tuple[float, ...]
    prb_utilization: tuple[float, ...]
    latency_ms: tuple[float, ...]
    confidence: float
    method: str


class TelemetryHistory:
    def __init__(self, maxlen: int = 120):
        self.values: deque[NetworkObservation] = deque(maxlen=maxlen)

    def append(self, observation: NetworkObservation) -> None:
        self.values.append(observation)


class PersistenceForecaster:
    def forecast(self, history: TelemetryHistory, horizon: int = 5) -> Forecast:
        if not history.values:
            raise ValueError("forecast history is empty")
        last = history.values[-1]
        return Forecast(horizon, tuple([last.throughput_demand_mbps] * horizon), tuple([last.prb_utilization] * horizon), tuple([last.latency_ms] * horizon), 0.55, "persistence")


class EWMAForecaster:
    def __init__(self, alpha: float = 0.45): self.alpha = alpha
    def _ewma(self, values: list[float]) -> float:
        estimate = values[0]
        for value in values[1:]: estimate = self.alpha * value + (1 - self.alpha) * estimate
        return estimate
    def forecast(self, history: TelemetryHistory, horizon: int = 5) -> Forecast:
        if not history.values: raise ValueError("forecast history is empty")
        rows = list(history.values)[-24:]
        demand = self._ewma([row.throughput_demand_mbps for row in rows]); prb = self._ewma([row.prb_utilization for row in rows]); latency = self._ewma([row.latency_ms for row in rows])
        return Forecast(horizon, tuple([demand] * horizon), tuple([prb] * horizon), tuple([latency] * horizon), min(0.88, 0.55 + len(rows) / 100.0), "ewma")


class LinearTrendForecaster:
    @staticmethod
    def _project(values: list[float], horizon: int) -> tuple[float, ...]:
        if len(values) < 3: return tuple([values[-1]] * horizon)
        x = np.arange(len(values), dtype=float); slope, intercept = np.polyfit(x, np.asarray(values, dtype=float), 1)
        return tuple(max(0.0, float(intercept + slope * (len(values) + step))) for step in range(horizon))
    def forecast(self, history: TelemetryHistory, horizon: int = 5) -> Forecast:
        if not history.values: raise ValueError("forecast history is empty")
        rows = list(history.values)[-24:]
        return Forecast(horizon, self._project([row.throughput_demand_mbps for row in rows], horizon), tuple(min(1.5, value) for value in self._project([row.prb_utilization for row in rows], horizon)), self._project([row.latency_ms for row in rows], horizon), min(0.82, 0.45 + len(rows) / 90.0), "linear-trend")


class ForecastEnsemble:
    def __init__(self): self.models = (EWMAForecaster(), LinearTrendForecaster())
    def forecast(self, history: TelemetryHistory, horizon: int = 5) -> Forecast:
        forecasts = [model.forecast(history, horizon) for model in self.models]
        demand = tuple(float(np.mean([item.demand_mbps[step] for item in forecasts])) for step in range(horizon)); prb = tuple(float(np.mean([item.prb_utilization[step] for item in forecasts])) for step in range(horizon)); latency = tuple(float(np.mean([item.latency_ms[step] for item in forecasts])) for step in range(horizon))
        spread = float(np.mean([np.std([item.demand_mbps[step] for item in forecasts]) for step in range(horizon)])); confidence = max(0.25, min(0.90, float(np.mean([item.confidence for item in forecasts])) - spread / 500.0))
        return Forecast(horizon, demand, prb, latency, confidence, "ensemble")
