"""Tamper-evident decision audit log and deterministic replay helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class HashChainAuditLog:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _canonical(payload: dict[str, Any]) -> bytes:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")

    @staticmethod
    def _hash(payload: dict[str, Any]) -> str:
        return hashlib.sha256(HashChainAuditLog._canonical(payload)).hexdigest()

    def _last_hash(self) -> str:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return "0" * 64
        last = ""
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    last = line
        if not last:
            return "0" * 64
        return str(json.loads(last)["record_hash"])

    def append(self, event_type: str, payload: dict[str, Any]) -> str:
        previous_hash = self._last_hash()
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), "event_type": event_type, "previous_hash": previous_hash, "payload": payload}
        record_hash = self._hash(record)
        record["record_hash"] = record_hash
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        return record_hash

    def verify(self) -> tuple[bool, list[str]]:
        if not self.path.exists():
            return True, []
        expected_previous = "0" * 64
        errors: list[str] = []
        with self.path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                record_hash = record.pop("record_hash")
                if record.get("previous_hash") != expected_previous:
                    errors.append(f"line {index}: previous hash mismatch")
                calculated = self._hash(record)
                if calculated != record_hash:
                    errors.append(f"line {index}: record hash mismatch")
                expected_previous = record_hash
        return not errors, errors

    def events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        events = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if event_type:
            events = [event for event in events if event.get("event_type") == event_type]
        return events
