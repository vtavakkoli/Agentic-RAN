"""Catalog and download utilities for public real-world RAN datasets."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import yaml


def load_source_catalog(path: Path | str) -> list[dict[str, Any]]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    sources = payload.get("sources", [])
    if not isinstance(sources, list) or not sources:
        raise ValueError("Real-data catalog must contain a non-empty 'sources' list")
    return [dict(item) for item in sources if isinstance(item, dict) and item.get("enabled", True)]


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_source(source: dict[str, Any], raw_root: Path, retries: int = 3) -> Path:
    source_id = str(source["id"])
    destination = raw_root / source_id / str(source["filename"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = str(source.get("md5") or "").lower()
    if destination.exists() and (not expected or md5_file(destination) == expected):
        return destination

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                str(source["download_url"]),
                headers={"User-Agent": "Agentic-RAN/2.1 real-data benchmark"},
            )
            with urllib.request.urlopen(request, timeout=90) as response, tempfile.NamedTemporaryFile(delete=False) as stream:
                shutil.copyfileobj(response, stream)
                temporary = Path(stream.name)
            if expected and md5_file(temporary) != expected:
                temporary.unlink(missing_ok=True)
                raise ValueError(f"Checksum mismatch for {source_id}")
            temporary.replace(destination)
            return destination
        except Exception as exc:  # pragma: no cover - network boundary
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Unable to download {source_id}: {last_error}") from last_error
