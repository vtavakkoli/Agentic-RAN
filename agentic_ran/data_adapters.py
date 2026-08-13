"""Normalize heterogeneous public 5G measurement tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ALIASES = {
    "timestamp": ("timestamp", "time", "datetime", "date"),
    "rsrp_dbm": ("rsrpdbm", "rsrp", "ssrsrp", "signaldbm", "signalstrengthdbm", "signal"),
    "rsrq_db": ("rsrqdb", "rsrq", "ssrsrq"),
    "sinr_db": ("sinrdb", "sinr", "sssinr", "snrdb", "snr"),
    "downlink_mbps": ("throughputdlmbps", "dlthroughputmbps", "downloadmbps", "downloadspeedmbps", "download", "throughputdl"),
    "uplink_mbps": ("throughputulmbps", "ulthroughputmbps", "uploadmbps", "uploadspeedmbps", "upload", "throughputul"),
    "latency_ms": ("latencyms", "latency", "pingms", "ping", "roundtriptimems", "roundtriptime", "rttms", "rtt"),
    "jitter_ms": ("jitterms", "jitter", "stdroundtriptimems", "rttstdms"),
    "latitude": ("latitude", "lat"),
    "longitude": ("longitude", "lon", "lng"),
    "location": ("location", "area", "site", "route"),
    "provider": ("provider", "operator", "network"),
    "device": ("device", "handset", "ue"),
}
MEASURED_FIELDS = ("rsrp_dbm", "sinr_db", "downlink_mbps", "uplink_mbps", "latency_ms", "jitter_ms")


def _key(value: object) -> str:
    return "".join(char for char in str(value).lower() if char.isalnum())


def _find(columns: list[object], aliases: tuple[str, ...]) -> object | None:
    names = {_key(column): column for column in columns}
    for alias in aliases:
        if alias in names:
            return names[alias]
    for name, original in names.items():
        if any(alias in name or name in alias for alias in aliases if len(alias) >= 5):
            return original
    return None


def read_tables(path: Path, source_format: str) -> list[tuple[str, pd.DataFrame]]:
    if source_format.lower() in {"xlsx", "xls"}:
        sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
        return [(str(name), frame) for name, frame in sheets.items()]
    return [("data", pd.read_csv(path))]


def normalize_table(frame: pd.DataFrame, source_id: str, sheet: str) -> pd.DataFrame:
    found = {name: _find(list(frame.columns), aliases) for name, aliases in ALIASES.items()}
    if sum(found[name] is not None for name in MEASURED_FIELDS) < 2:
        return pd.DataFrame()

    out = pd.DataFrame(index=frame.index)
    out["source_id"] = source_id
    out["source_sheet"] = sheet
    out["source_row"] = frame.index.astype(int)
    for name in ("timestamp", "location", "provider", "device"):
        out[name] = "" if found[name] is None else frame[found[name]].fillna("").astype(str)
    for name in ("rsrp_dbm", "rsrq_db", "sinr_db", "downlink_mbps", "uplink_mbps", "latency_ms", "jitter_ms", "latitude", "longitude"):
        out[name] = np.nan if found[name] is None else pd.to_numeric(frame[found[name]], errors="coerce")
    out["measured_fields"] = out.apply(
        lambda row: ",".join(name for name in MEASURED_FIELDS if pd.notna(row[name])), axis=1
    )
    return out.dropna(subset=list(MEASURED_FIELDS), how="all").reset_index(drop=True)
