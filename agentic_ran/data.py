"""Small, reproducible RAN-policy dataset utilities."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from agentic_ran.config import FEATURES, TARGET_COLUMN


class DatasetIntegrityError(ValueError):
    """Raised when downloaded or generated data fails an integrity check."""


POLICY_LABELS = [
    "balanced",
    "latency_guard",
    "throughput_boost",
    "congestion_relief",
    "energy_saver",
    "coverage_recovery",
    "massive_iot_access",
]


def expert_policy_label(row: dict[str, float | int | str]) -> str:
    """Create transparent expert labels for the compact bootstrap dataset."""

    slice_type = str(row["slice_type"])
    prb = float(row["prb_utilization"])
    latency = float(row["latency_ms"])
    loss = float(row["packet_loss_pct"])
    demand = float(row["throughput_demand_mbps"])
    downlink = max(float(row["downlink_mbps"]), 1.0)
    users = int(row["active_users"])
    rsrp = float(row["rsrp_dbm"])
    sinr = float(row["sinr_db"])
    energy = float(row["energy_load"])
    handover = float(row["handover_failure_pct"])

    if rsrp < -112 or sinr < 1.5 or handover > 5.0:
        return "coverage_recovery"
    if slice_type == "URLLC" and (latency > 12.0 or loss > 0.35 or prb > 0.88):
        return "latency_guard"
    if prb > 0.93 or loss > 2.2 or demand / downlink > 1.48:
        return "congestion_relief"
    if slice_type == "mMTC" and (users > 700 or loss > 1.1):
        return "massive_iot_access"
    if slice_type == "eMBB" and (demand / downlink > 1.15 or demand > 180.0):
        return "throughput_boost"
    if prb < 0.38 and energy < 0.48 and demand / downlink < 0.90:
        return "energy_saver"
    return "balanced"


def generate_dataset(rows: int = 1_200, seed: int = 42) -> pd.DataFrame:
    """Generate a small KPI dataset with realistic correlations and expert labels."""

    if rows < 140:
        raise ValueError("rows must be at least 140 so every policy is represented")

    rng = np.random.default_rng(seed)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    slices = rng.choice(["eMBB", "URLLC", "mMTC"], size=rows, p=[0.46, 0.28, 0.26])
    records: list[dict[str, float | int | str]] = []

    for index, slice_type in enumerate(slices):
        hour = (index // 12) % 24
        peak = 1.0 if 8 <= hour <= 11 or 17 <= hour <= 21 else 0.0
        cell_bias = (index % 8) / 28.0

        if slice_type == "eMBB":
            users = int(np.clip(rng.normal(155 + 80 * peak, 55), 12, 520))
            demand = float(np.clip(rng.normal(150 + 105 * peak, 52), 30, 420))
            latency_base = 27.0
        elif slice_type == "URLLC":
            users = int(np.clip(rng.normal(52 + 20 * peak, 18), 4, 170))
            demand = float(np.clip(rng.normal(48 + 25 * peak, 17), 8, 125))
            latency_base = 7.0
        else:
            users = int(np.clip(rng.normal(560 + 230 * peak, 190), 80, 1_350))
            demand = float(np.clip(rng.normal(32 + 15 * peak, 12), 5, 90))
            latency_base = 58.0

        prb = float(np.clip(0.20 + demand / 480.0 + users / 2_900.0 + peak * 0.12 + rng.normal(0, 0.09), 0.05, 1.18))
        radio_degradation = max(0.0, rng.normal(0.0, 1.0)) if index % 11 == 0 else max(0.0, rng.normal(-0.4, 0.55))
        rsrp = float(np.clip(rng.normal(-91.0 - 13.0 * radio_degradation - 4.0 * cell_bias, 7.5), -128, -62))
        sinr = float(np.clip(21.0 - 0.23 * (-90.0 - rsrp) - 9.0 * max(prb - 0.75, 0) + rng.normal(0, 4.0), -8, 38))

        capacity = max(10.0, 330.0 * max(0.18, 1.0 - 0.56 * prb) * max(0.35, min(1.15, (sinr + 12.0) / 30.0)))
        downlink = float(np.clip(min(demand * rng.uniform(0.70, 1.10), capacity), 1.0, 500.0))
        uplink = float(np.clip(downlink * rng.uniform(0.08, 0.34), 0.2, 130.0))
        congestion = max(0.0, prb - 0.72)
        latency = float(np.clip(latency_base * (1.0 + congestion * 3.2) + radio_degradation * 9.0 + rng.normal(0, 4.5), 1.0, 240.0))
        jitter = float(np.clip(latency * rng.uniform(0.08, 0.27) + congestion * 13.0, 0.1, 80.0))
        loss = float(np.clip(0.05 + congestion * 7.0 + radio_degradation * 0.9 + rng.normal(0, 0.18), 0.0, 12.0))
        energy = float(np.clip(0.18 + 0.68 * min(prb, 1.0) + 0.08 * peak + rng.normal(0, 0.05), 0.08, 1.18))
        handover = float(np.clip(0.25 + radio_degradation * 2.8 + max(0.0, -sinr) * 0.18 + rng.normal(0, 0.35), 0.0, 14.0))

        # Inject recurring, realistic operating regimes so the compact dataset has
        # enough examples of every policy for stable train/test splits.
        regime = index % 14
        if regime in {0, 1}:  # URLLC SLA pressure
            slice_type = "URLLC"
            prb = float(rng.uniform(0.76, 0.91))
            users = int(rng.integers(35, 120))
            downlink = float(rng.uniform(38, 76))
            demand = float(downlink * rng.uniform(1.02, 1.25))
            latency = float(rng.uniform(16, 36))
            jitter = float(rng.uniform(4, 12))
            loss = float(rng.uniform(0.40, 1.10))
            energy = float(rng.uniform(0.66, 0.92))
            handover, rsrp, sinr = 1.0, -97.0, 12.0
        elif regime in {2, 3}:  # eMBB demand pressure with remaining headroom
            slice_type = "eMBB"
            prb = float(rng.uniform(0.58, 0.84))
            users = int(rng.integers(140, 360))
            downlink = float(rng.uniform(85, 170))
            demand = float(downlink * rng.uniform(1.20, 1.42))
            latency = float(rng.uniform(24, 45))
            jitter = float(rng.uniform(3, 9))
            loss = float(rng.uniform(0.08, 0.70))
            energy = float(rng.uniform(0.60, 0.88))
            handover, rsrp, sinr = 0.8, -94.0, 16.0
        elif regime in {4, 5}:  # critical congestion
            slice_type = "eMBB"
            prb = float(rng.uniform(0.96, 1.12))
            users = int(rng.integers(280, 520))
            downlink = float(rng.uniform(55, 120))
            demand = float(downlink * rng.uniform(1.35, 1.80))
            latency = float(rng.uniform(58, 125))
            jitter = float(rng.uniform(12, 32))
            loss = float(rng.uniform(2.6, 6.5))
            energy = float(rng.uniform(0.88, 1.10))
            handover, rsrp, sinr = 1.6, -101.0, 7.0
        elif regime == 6:  # low-load energy opportunity
            slice_type = "eMBB"
            prb = float(rng.uniform(0.14, 0.34))
            users = int(rng.integers(20, 90))
            downlink = float(rng.uniform(55, 105))
            demand = float(downlink * rng.uniform(0.58, 0.84))
            latency = float(rng.uniform(18, 34))
            jitter = float(rng.uniform(2, 6))
            loss = float(rng.uniform(0.01, 0.25))
            energy = float(rng.uniform(0.20, 0.42))
            handover, rsrp, sinr = 0.4, -91.0, 20.0
        elif regime == 7:  # weak coverage / mobility instability
            slice_type = "eMBB"
            prb = float(rng.uniform(0.46, 0.76))
            users = int(rng.integers(90, 240))
            downlink = float(rng.uniform(35, 90))
            demand = float(downlink * rng.uniform(0.90, 1.18))
            latency = float(rng.uniform(44, 95))
            jitter = float(rng.uniform(8, 21))
            loss = float(rng.uniform(0.8, 2.0))
            energy = float(rng.uniform(0.62, 0.90))
            handover, rsrp, sinr = float(rng.uniform(5.5, 9.5)), float(rng.uniform(-124, -114)), float(rng.uniform(-4, 1))
        elif regime == 8:  # dense massive-IoT access
            slice_type = "mMTC"
            prb = float(rng.uniform(0.58, 0.84))
            users = int(rng.integers(780, 1_250))
            downlink = float(rng.uniform(24, 58))
            demand = float(downlink * rng.uniform(0.92, 1.30))
            latency = float(rng.uniform(62, 105))
            jitter = float(rng.uniform(8, 20))
            loss = float(rng.uniform(1.2, 2.0))
            energy = float(rng.uniform(0.62, 0.90))
            handover, rsrp, sinr = 0.9, -96.0, 13.0
        elif regime == 9:  # explicit balanced operation
            slice_type = "eMBB"
            prb = float(rng.uniform(0.42, 0.65))
            users = int(rng.integers(80, 190))
            downlink = float(rng.uniform(105, 175))
            demand = float(downlink * rng.uniform(0.88, 1.08))
            latency = float(rng.uniform(20, 35))
            jitter = float(rng.uniform(2, 7))
            loss = float(rng.uniform(0.03, 0.45))
            energy = float(rng.uniform(0.50, 0.72))
            handover, rsrp, sinr = 0.6, -93.0, 18.0

        record: dict[str, float | int | str] = {
            "timestamp": (start + timedelta(minutes=5 * index)).isoformat(),
            "cell_id": f"cell-{index % 8 + 1:03d}",
            "slice_type": str(slice_type),
            "prb_utilization": round(prb, 5),
            "active_users": users,
            "downlink_mbps": round(downlink, 4),
            "uplink_mbps": round(uplink, 4),
            "latency_ms": round(latency, 4),
            "jitter_ms": round(jitter, 4),
            "packet_loss_pct": round(loss, 5),
            "throughput_demand_mbps": round(demand, 4),
            "energy_load": round(energy, 5),
            "handover_failure_pct": round(handover, 5),
            "rsrp_dbm": round(rsrp, 4),
            "sinr_db": round(sinr, 4),
        }
        record[TARGET_COLUMN] = expert_policy_label(record)
        records.append(record)

    frame = pd.DataFrame.from_records(records)

    # Guarantee complete policy coverage with a handful of deterministic anchors.
    anchors = _policy_anchors()
    for offset, anchor in enumerate(anchors):
        anchor["timestamp"] = (start + timedelta(minutes=5 * (rows + offset))).isoformat()
        anchor["cell_id"] = f"anchor-{offset + 1:02d}"
        anchor[TARGET_COLUMN] = expert_policy_label(anchor)
    frame = pd.concat([frame, pd.DataFrame(anchors)], ignore_index=True)
    validate_dataset(frame)
    return frame


def _policy_anchors() -> list[dict[str, float | int | str]]:
    common = {"uplink_mbps": 12.0, "jitter_ms": 3.0, "handover_failure_pct": 0.5, "rsrp_dbm": -90.0, "sinr_db": 18.0}
    return [
        {**common, "slice_type": "eMBB", "prb_utilization": 0.55, "active_users": 120, "downlink_mbps": 140.0, "latency_ms": 25.0, "packet_loss_pct": 0.1, "throughput_demand_mbps": 145.0, "energy_load": 0.6},
        {**common, "slice_type": "URLLC", "prb_utilization": 0.82, "active_users": 80, "downlink_mbps": 50.0, "latency_ms": 22.0, "packet_loss_pct": 0.6, "throughput_demand_mbps": 60.0, "energy_load": 0.8},
        {**common, "slice_type": "eMBB", "prb_utilization": 0.76, "active_users": 260, "downlink_mbps": 120.0, "latency_ms": 35.0, "packet_loss_pct": 0.6, "throughput_demand_mbps": 260.0, "energy_load": 0.85},
        {**common, "slice_type": "eMBB", "prb_utilization": 1.03, "active_users": 390, "downlink_mbps": 80.0, "latency_ms": 70.0, "packet_loss_pct": 4.0, "throughput_demand_mbps": 260.0, "energy_load": 1.0},
        {**common, "slice_type": "mMTC", "prb_utilization": 0.22, "active_users": 150, "downlink_mbps": 25.0, "latency_ms": 70.0, "packet_loss_pct": 0.3, "throughput_demand_mbps": 18.0, "energy_load": 0.25},
        {**common, "slice_type": "eMBB", "prb_utilization": 0.60, "active_users": 180, "downlink_mbps": 70.0, "latency_ms": 50.0, "packet_loss_pct": 1.5, "throughput_demand_mbps": 90.0, "energy_load": 0.7, "rsrp_dbm": -119.0, "sinr_db": -1.0, "handover_failure_pct": 7.0},
        {**common, "slice_type": "mMTC", "prb_utilization": 0.72, "active_users": 980, "downlink_mbps": 35.0, "latency_ms": 85.0, "packet_loss_pct": 1.6, "throughput_demand_mbps": 48.0, "energy_load": 0.75},
    ]


def validate_dataset(frame: pd.DataFrame) -> None:
    required = set(FEATURES + [TARGET_COLUMN, "timestamp", "cell_id"])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Dataset is empty")
    if frame[FEATURES].isnull().any().any():
        raise ValueError("Dataset contains null feature values")
    invalid = sorted(set(frame[TARGET_COLUMN].astype(str)).difference(POLICY_LABELS))
    if invalid:
        raise ValueError(f"Dataset contains unknown policy labels: {invalid}")


def write_dataset(frame: pd.DataFrame, destination: Path | str) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(
    url: str,
    destination: Path | str,
    expected_sha256: str | None = None,
    allow_fallback: bool = True,
    fallback_rows: int = 1_200,
    seed: int = 42,
) -> tuple[Path, str]:
    """Download the small dataset atomically; generate it locally when offline."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    source = "download"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Agentic-RAN/1.0"})
        timeout = float(os.getenv("AGENTIC_RAN_DOWNLOAD_TIMEOUT", "20"))
        with urllib.request.urlopen(request, timeout=timeout) as response, tempfile.NamedTemporaryFile(delete=False) as temp:
            shutil.copyfileobj(response, temp)
            temp_path = Path(temp.name)
        if expected_sha256 and sha256_file(temp_path) != expected_sha256.lower():
            temp_path.unlink(missing_ok=True)
            raise DatasetIntegrityError("Downloaded dataset checksum does not match AGENTIC_RAN_DATASET_SHA256")
        frame = pd.read_csv(temp_path)
        validate_dataset(frame)
        temp_path.replace(path)
    except DatasetIntegrityError:
        raise
    except Exception:
        if not allow_fallback:
            raise
        source = "deterministic-fallback"
        write_dataset(generate_dataset(rows=fallback_rows, seed=seed), path)
        if expected_sha256 and sha256_file(path) != expected_sha256.lower():
            path.unlink(missing_ok=True)
            raise DatasetIntegrityError("Fallback dataset checksum does not match AGENTIC_RAN_DATASET_SHA256")
    return path, source


def load_dataset(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_dataset(frame)
    return frame
