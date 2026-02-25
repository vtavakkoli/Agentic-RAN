from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from oran_sim.data import _load_bs_metrics, _load_enb_metrics, _load_ue_metrics, load_timeseries_from_kpm

FIX = Path(__file__).parent / "fixtures"


def _build_kpm_tree(tmp_path: Path) -> Path:
    root = tmp_path / "shared_data" / "dataset-kpm" / "cluster_1" / "slicing_1" / "scheduling_0" / "RESERVATION-142634"
    (root / "bs").mkdir(parents=True, exist_ok=True)
    (root / "ue_001010123456002").mkdir(parents=True, exist_ok=True)

    shutil.copy(FIX / "bs" / "1010123456002_metrics.csv", root / "bs" / "1010123456002_metrics.csv")
    shutil.copy(FIX / "bs" / "enb_metrics.csv", root / "enb_metrics.csv")
    shutil.copy(FIX / "bs" / "enb_metrics.csv", root / "bs" / "enb_metrics.csv")
    shutil.copy(FIX / "ue_001010123456002" / "ue_metrics.csv", root / "ue_001010123456002" / "ue_metrics.csv")
    shutil.copy(FIX / "ue_001010123456002" / "mgen.csv", root / "ue_001010123456002" / "flow_log.csv")

    return tmp_path / "shared_data" / "dataset-kpm"


def test_bs_metrics_parsing_normalization(tmp_path: Path) -> None:
    ds_root = _build_kpm_tree(tmp_path)
    reservation = ds_root / "cluster_1" / "slicing_1" / "scheduling_0" / "RESERVATION-142634"
    df = _load_bs_metrics(reservation)

    assert len(df) == 5
    for col in [
        "num_ues",
        "dl_mcs",
        "tx_brate_dl_mbps",
        "dl_buffer_bytes",
        "ul_sinr",
        "sum_granted_prbs",
        "sum_requested_prbs",
        "tx_errors_dl_pct",
        "rx_errors_ul_pct",
    ]:
        assert col in df.columns
        assert pd.api.types.is_numeric_dtype(df[col])


def test_enb_metrics_parsing_and_time_alignment(tmp_path: Path) -> None:
    ds_root = _build_kpm_tree(tmp_path)
    reservation = ds_root / "cluster_1" / "slicing_1" / "scheduling_0" / "RESERVATION-142634"
    enb = _load_enb_metrics(reservation)

    assert len(enb) == 17
    assert "time_ms" in enb.columns
    assert enb["time_ms"].is_monotonic_increasing
    assert pd.api.types.is_numeric_dtype(enb["enb_dl_brate"])


def test_ue_metrics_semicolon_parsing(tmp_path: Path) -> None:
    ds_root = _build_kpm_tree(tmp_path)
    reservation = ds_root / "cluster_1" / "slicing_1" / "scheduling_0" / "RESERVATION-142634"
    ue_path = reservation / "ue_001010123456002" / "ue_metrics.csv"

    ue = pd.read_csv(ue_path, sep=";")
    for col in ["rsrp", "dl_snr", "dl_brate", "ul_buff", "is_attached"]:
        assert col in ue.columns

    assert float(ue.iloc[-1]["is_attached"]) == 1.0
    assert float(ue.loc[ue["time"] == 1743, "rsrp"].iloc[0]) == -66

    merged_ue = _load_ue_metrics(reservation, start_time_ms=1708463142910)
    assert "time_ms" in merged_ue.columns
    assert merged_ue["time_ms"].min() >= 1708463142910


def test_load_timeseries_from_kpm_merging(tmp_path: Path) -> None:
    ds_root = _build_kpm_tree(tmp_path)
    df = load_timeseries_from_kpm(ds_root, n_steps=100, verbose=False)

    assert not df.empty
    assert df["time_ms"].is_monotonic_increasing
    assert pd.api.types.is_numeric_dtype(df["traffic_load"])

    must_have = ["dl_cqi", "ul_sinr", "sum_granted_prbs", "sum_requested_prbs", "latency_ms"]
    for col in must_have:
        assert col in df.columns

    assert df.shape[1] >= 10
    assert int(df.isna().sum().sum()) == 0
