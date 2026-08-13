from __future__ import annotations

from pathlib import Path

import pandas as pd

from agentic_ran.config import FEATURES, TARGET_COLUMN
from agentic_ran.data import POLICY_LABELS, download_dataset, generate_dataset, sha256_file, validate_dataset, write_dataset
from agentic_ran.data_adapters import normalize_table
from agentic_ran.real_features import build_model_ready


def test_generation_is_deterministic() -> None:
    first = generate_dataset(rows=210, seed=7)
    second = generate_dataset(rows=210, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_generated_dataset_has_complete_schema_and_labels() -> None:
    frame = generate_dataset(rows=240, seed=3)
    validate_dataset(frame)
    assert set(FEATURES).issubset(frame.columns)
    assert set(frame[TARGET_COLUMN]) == set(POLICY_LABELS)
    assert frame[FEATURES].isna().sum().sum() == 0


def test_dataset_write_and_checksum(tmp_path: Path) -> None:
    path = write_dataset(generate_dataset(rows=160, seed=1), tmp_path / "sample.csv")
    assert path.exists()
    assert len(sha256_file(path)) == 64


def test_download_uses_deterministic_fallback_when_offline(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AGENTIC_RAN_DOWNLOAD_TIMEOUT", "0.1")
    path, source = download_dataset(
        "http://127.0.0.1:9/not-available.csv",
        tmp_path / "downloaded.csv",
        allow_fallback=True,
        fallback_rows=160,
        seed=9,
    )
    assert source == "deterministic-fallback"
    validate_dataset(pd.read_csv(path))


def test_checksum_mismatch_never_falls_back_silently(tmp_path: Path) -> None:
    from agentic_ran.data import DatasetIntegrityError

    source = write_dataset(generate_dataset(rows=160, seed=2), tmp_path / "source.csv")
    try:
        download_dataset(
            source.as_uri(),
            tmp_path / "target.csv",
            expected_sha256="0" * 64,
            allow_fallback=True,
            fallback_rows=160,
            seed=2,
        )
    except DatasetIntegrityError:
        pass
    else:  # pragma: no cover - assertion branch
        raise AssertionError("checksum mismatch must raise DatasetIntegrityError")


def test_real_measurement_adapter_preserves_measured_fields() -> None:
    raw = pd.DataFrame(
        {
            "Signal (dBm)": [-82, -101, -94],
            "Download (Mbps)": [420.0, 55.0, 170.0],
            "Upload (Mbps)": [80.0, 12.0, 35.0],
            "Ping (ms)": [18.0, 72.0, 34.0],
            "Location": ["A", "B", "C"],
        }
    )
    normalized = normalize_table(raw, "fixture", "measurements")
    ready = build_model_ready(normalized)
    validate_dataset(ready)
    assert normalized["rsrp_dbm"].tolist() == [-82, -101, -94]
    assert ready["observed_model_features"].min() >= 4
    assert ready["realism_score"].between(0, 1).all()
