"""Transparent feature completion for partially observed real-world RAN measurements."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from agentic_ran.data import expert_policy_label, validate_dataset
from agentic_ran.data_adapters import MEASURED_FIELDS


def _median(frame: pd.DataFrame, name: str, fallback: float) -> float:
    values = frame[name].dropna()
    return fallback if values.empty else float(values.median())


def build_model_ready(measurements: pd.DataFrame) -> pd.DataFrame:
    """Complete the model schema while explicitly marking measured vs derived inputs."""

    if measurements.empty:
        raise ValueError("No usable real-world measurements were extracted")
    dl_default = _median(measurements, "downlink_mbps", 80.0)
    ul_default = _median(measurements, "uplink_mbps", max(8.0, dl_default * 0.18))
    latency_default = _median(measurements, "latency_ms", 35.0)
    rsrp_default = _median(measurements, "rsrp_dbm", -95.0)
    p95_dl = max(
        float(measurements["downlink_mbps"].dropna().quantile(0.95))
        if measurements["downlink_mbps"].notna().any()
        else 100.0,
        10.0,
    )

    records = []
    for index, row in measurements.iterrows():
        dl = float(row["downlink_mbps"]) if pd.notna(row["downlink_mbps"]) else dl_default
        ul = float(row["uplink_mbps"]) if pd.notna(row["uplink_mbps"]) else ul_default
        latency = float(row["latency_ms"]) if pd.notna(row["latency_ms"]) else latency_default
        rsrp = float(row["rsrp_dbm"]) if pd.notna(row["rsrp_dbm"]) else rsrp_default
        sinr = (
            float(row["sinr_db"])
            if pd.notna(row["sinr_db"])
            else float(np.clip((rsrp + 105.0) * 0.78 + 3.0, -8.0, 35.0))
        )
        jitter = (
            float(row["jitter_ms"])
            if pd.notna(row["jitter_ms"])
            else float(np.clip(latency * 0.14, 0.2, 35.0))
        )
        load = float(np.clip(dl / p95_dl, 0.0, 1.25))
        radio_pressure = float(np.clip((-rsrp - 88.0) / 42.0, 0.0, 1.0))
        prb = float(
            np.clip(0.18 + 0.66 * load + np.clip((latency - 25.0) / 100.0, 0.0, 0.35), 0.05, 1.18)
        )
        demand = float(max(1.0, dl * (0.88 + 0.38 * min(prb, 1.0))))
        measured = {value for value in str(row["measured_fields"]).split(",") if value}
        observed = len(set(MEASURED_FIELDS).intersection(measured))
        timestamp = str(row.get("timestamp", "")).strip()
        if not timestamp or timestamp.lower() == "nan":
            timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat()
        parts = (str(row.get("provider", "")).strip(), str(row.get("location", "")).strip())
        suffix = "-".join(value for value in parts if value and value.lower() != "nan")

        record = {
            "timestamp": timestamp,
            "cell_id": f"{row['source_id']}-{suffix or index}"[:64],
            "slice_type": "eMBB",
            "prb_utilization": round(prb, 5),
            "active_users": 1,
            "downlink_mbps": round(max(0.0, dl), 4),
            "uplink_mbps": round(max(0.0, ul), 4),
            "latency_ms": round(max(0.0, latency), 4),
            "jitter_ms": round(max(0.0, jitter), 4),
            "packet_loss_pct": round(
                float(np.clip(max(0.0, latency - 38.0) / 55.0 + 0.85 * radio_pressure, 0.0, 8.0)), 5
            ),
            "throughput_demand_mbps": round(demand, 4),
            "energy_load": round(float(np.clip(0.18 + 0.68 * min(prb, 1.0), 0.08, 1.18)), 5),
            "handover_failure_pct": round(float(np.clip(0.25 + 4.5 * radio_pressure, 0.0, 8.0)), 5),
            "rsrp_dbm": round(float(np.clip(rsrp, -160.0, -20.0)), 4),
            "sinr_db": round(float(np.clip(sinr, -30.0, 80.0)), 4),
            "source_id": row["source_id"],
            "source_sheet": row["source_sheet"],
            "measured_fields": row["measured_fields"],
            "observed_model_features": observed,
            "derived_model_features": 12 - observed,
            "realism_score": round(observed / 12.0, 4),
        }
        record["policy_label"] = expert_policy_label(record)
        records.append(record)
    output = pd.DataFrame.from_records(records)
    validate_dataset(output)
    return output
