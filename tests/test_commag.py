from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agentic_ran.cli import main
from agentic_ran.commag import (
    COMMAG_REVISION,
    commag_core_paths,
    prepare_commag_data,
    train_commag_fitted_q,
    validate_commag_benchmark,
)


def _commag_fixture(root: Path, samples: int = 125) -> list[str]:
    paths: list[str] = []
    for experiment in (1, 2):
        for training, (slice_id, scheduler, prbs) in enumerate(((0, 0, 4), (1, 1, 8), (2, 2, 12))):
            relative = (
                f"slice_traffic/rome_slow_close/tr{training}/exp{experiment}/bs1/slices_bs1/"
                f"101012345600{training + 2}_metrics.csv"
            )
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            index = np.arange(samples, dtype=float)
            bitrate_scale = (slice_id + 1) * (1.0 + 0.03 * experiment)
            frame = pd.DataFrame(
                {
                    "Timestamp": (1_700_000_000_000 + experiment * 1_000_000 + index * 1_000).astype(int),
                    "num_ues": 10 + slice_id,
                    "slice_id": slice_id,
                    "slice_prb": prbs,
                    "power_multiplier": 1,
                    "scheduling_policy": scheduler,
                    "dl_mcs": 8 + slice_id + np.sin(index / 9),
                    "dl_buffer [bytes]": 120 + index % 17,
                    "tx_brate downlink [Mbps]": bitrate_scale * (0.2 + (index % 11) / 20),
                    "tx_errors downlink (%)": (index + slice_id) % 5,
                    "dl_cqi": 7 + slice_id,
                    "ul_mcs": 5 + slice_id,
                    "ul_buffer [bytes]": 20 + index % 7,
                    "rx_brate uplink [Mbps]": 0.05 + (index % 5) / 100,
                    "rx_errors uplink (%)": (index + 1) % 4,
                    "ul_sinr": 10 + slice_id + np.cos(index / 10),
                    "sum_requested_prbs": 20 + index % 9,
                    "sum_granted_prbs": 18 + index % 8,
                }
            )
            frame.to_csv(path, index=False)
            paths.append(relative)
    return paths


def test_core_profile_is_pinned_and_compact() -> None:
    paths = commag_core_paths(train_configs=(0,), experiments=(1, 2), base_stations=(1,))
    assert len(paths) == 6
    assert all("rome_slow_close" in path for path in paths)
    assert len(COMMAG_REVISION) == 40
    with pytest.raises(ValueError, match="between 0 and 17"):
        commag_core_paths(train_configs=(18,))
    with pytest.raises(ValueError, match="base stations 1 and 4"):
        commag_core_paths(base_stations=(2,))


def test_prepare_train_and_validate_commag(tmp_path: Path) -> None:
    raw = tmp_path / "upstream"
    prepared = tmp_path / "prepared"
    paths = _commag_fixture(raw)
    manifest = prepare_commag_data(
        output_dir=prepared,
        source_dir=raw,
        source_paths=paths,
        revision=COMMAG_REVISION,
    )
    dataset = prepared / "commag_transitions.csv.gz"
    manifest_path = prepared / "commag_manifest.json"
    frame = pd.read_csv(dataset)

    assert manifest["source_license"] == "GPL-3.0"
    assert manifest["episode_overlap"] == 0
    assert manifest["train_rows"] == manifest["test_rows"]
    assert set(frame["split"]) == {"train", "test"}
    assert frame["reward"].between(-0.25, 1.0).all()
    assert dataset.stat().st_size < manifest["raw_bytes"]

    model = tmp_path / "commag.joblib"
    metrics_path = tmp_path / "metrics.json"
    report = tmp_path / "report.html"
    metrics = train_commag_fitted_q(
        dataset,
        model,
        metrics_path,
        report,
        manifest_path,
        iterations=2,
        seed=7,
    )
    valid, errors = validate_commag_benchmark(metrics_path)

    assert valid, errors
    assert metrics["verdict"] == "BENCHMARK-READY"
    assert len(metrics["actions"]) == 3
    assert model.exists()
    assert "COMMAG benchmark report" in report.read_text(encoding="utf-8")
    assert main(["validate-commag", "--metrics", str(metrics_path)]) == 0


def test_validation_rejects_incomplete_metrics(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"structural_checks": {"data split": False}}), encoding="utf-8")
    valid, errors = validate_commag_benchmark(path)
    assert not valid
    assert "data split" in errors
    assert main(["validate-commag", "--metrics", str(path)]) == 2
