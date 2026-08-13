"""Orchestrate real-data download, normalization, compression, and provenance."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from agentic_ran.data import sha256_file
from agentic_ran.data_adapters import MEASURED_FIELDS, normalize_table, read_tables
from agentic_ran.data_sources import download_source, load_source_catalog
from agentic_ran.real_features import build_model_ready


def prepare_real_data(
    catalog_path: Path | str = "configs/data_sources.yaml",
    raw_dir: Path | str = "data/raw",
    output_dir: Path | str = "data/prepared",
    max_rows_per_source: int = 20_000,
) -> dict[str, Any]:
    """Prepare compressed benchmark data from public 5G measurements."""

    sources = load_source_catalog(catalog_path)
    raw_root = Path(raw_dir)
    destination = Path(output_dir)
    raw_root.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)

    extracted: list[pd.DataFrame] = []
    reports: list[dict[str, Any]] = []
    for source in sources:
        source_id = str(source["id"])
        path = download_source(source, raw_root)
        normalized_tables = []
        for sheet, frame in read_tables(path, str(source.get("format", "csv"))):
            normalized = normalize_table(frame, source_id, sheet)
            if not normalized.empty:
                normalized_tables.append(normalized)
        if not normalized_tables:
            raise ValueError(f"No compatible KPI table found for {source_id}")
        combined = pd.concat(normalized_tables, ignore_index=True).head(max_rows_per_source)
        extracted.append(combined)
        reports.append(
            {
                "id": source_id,
                "title": source.get("title"),
                "doi": source.get("doi"),
                "record_url": source.get("record_url"),
                "license": source.get("license", "source-record"),
                "raw_file": str(path),
                "raw_sha256": sha256_file(path),
                "rows": len(combined),
                "measured_field_counts": {name: int(combined[name].notna().sum()) for name in MEASURED_FIELDS},
            }
        )

    measurements = pd.concat(extracted, ignore_index=True)
    model_ready = build_model_ready(measurements)
    normalized_path = destination / "real_measurements.csv.gz"
    model_path = destination / "real_policy_eval.csv.gz"
    measurements.to_csv(normalized_path, index=False, compression="gzip")
    model_ready.to_csv(model_path, index=False, compression="gzip")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "catalog": str(catalog_path),
        "raw_files_are_redistributed": False,
        "normalized_dataset": str(normalized_path),
        "model_ready_dataset": str(model_path),
        "normalized_sha256": sha256_file(normalized_path),
        "model_ready_sha256": sha256_file(model_path),
        "rows": len(model_ready),
        "mean_realism_score": float(model_ready["realism_score"].mean()),
        "sources": reports,
        "important_note": (
            "Real measurements do not contain operator policy actions. policy_label and missing RAN features are "
            "transparent expert-derived benchmark references, not deployment ground truth."
        ),
    }
    provenance = destination / "provenance.json"
    provenance.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
