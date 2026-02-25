from __future__ import annotations

from pathlib import Path

from oran_sim.data import load_timeseries_from_kpm


def _make_reservation(root: Path, name: str = "RESERVATION-1") -> Path:
    res = root / "cluster_a" / "slicing_on_20" / "scheduling_rr" / name
    (res / "bs").mkdir(parents=True, exist_ok=True)
    (res / "ue_1").mkdir(parents=True, exist_ok=True)
    return res


def test_bs_metrics_header_with_double_commas(tmp_path: Path) -> None:
    res = _make_reservation(tmp_path)
    (res / "bs" / "cell_metrics.csv").write_text(
        "time,,tx_brate downlink [Mbps],sum_requested_prbs,sum_granted_prbs\n"
        "1000,,10.0,20,10\n"
        "2000,,12.0,22,11\n",
        encoding="utf-8",
    )
    (res / "bs" / "enb_metrics.csv").write_text("time,nof_ue,dl_brate\n1000,3,10\n2000,4,12\n", encoding="utf-8")
    (res / "ue_1" / "ue_metrics.csv").write_text("time;dl_mcs;ul_mcs;dl_cqi;ul_sinr\n0;5;4;10;20\n1000;6;5;11;21\n", encoding="utf-8")

    df = load_timeseries_from_kpm(tmp_path)
    assert len(df) >= 2
    assert "traffic_load" in df.columns
    assert "sum_requested_prbs" in df.columns


def test_ue_metrics_semicolon_delimiter(tmp_path: Path) -> None:
    res = _make_reservation(tmp_path, "RESERVATION-2")
    (res / "bs" / "enb_metrics.csv").write_text("time,nof_ue,dl_brate\n1000,2,5\n2000,2,6\n", encoding="utf-8")
    (res / "bs" / "cell_metrics.csv").write_text("time,tx_brate downlink [Mbps]\n1000,5\n2000,6\n", encoding="utf-8")
    (res / "ue_1" / "ue_metrics.csv").write_text("time;dl_mcs;ul_mcs;dl_cqi;ul_sinr;ul_rssi\n0;1;2;3;4;5\n1000;2;3;4;5;6\n", encoding="utf-8")

    df = load_timeseries_from_kpm(tmp_path)
    assert "ul_rssi" in df.columns
    assert df["dl_mcs"].sum() > 0
