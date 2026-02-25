import tempfile
import unittest
from pathlib import Path

from oran_sim.data import load_timeseries_from_kpm


class KpmParserTests(unittest.TestCase):
    def test_parser_handles_bs_enb_ue(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset-kpm" / "cluster_1" / "slicing_on" / "scheduling_rr" / "RESERVATION-1"
            (root / "bs").mkdir(parents=True)
            (root / "ue_1").mkdir(parents=True)

            (root / "bs" / "enb_metrics.csv").write_text(
                "time,dl_brate,nof_ue,dl_buffer [bytes],sum_granted_prbs\n"
                "1000,10,5,1000,50\n"
                "1100,12,6,1200,60\n",
                encoding="utf-8",
            )
            (root / "bs" / "cell_metrics.csv").write_text(
                "time,tx_brate downlink [Mbps],sum_requested_prbs\n"
                "1000,10,55\n"
                "1100,13,65\n",
                encoding="utf-8",
            )
            (root / "ue_1" / "ue_metrics.csv").write_text(
                "time;dl_cqi;ul_sinr;dl_mcs;ul_mcs\n"
                "0;10;12;18;16\n"
                "100;11;13;19;17\n",
                encoding="utf-8",
            )

            df = load_timeseries_from_kpm(root.parent.parent.parent.parent, n_steps=2, verbose=False)
            self.assertEqual(len(df), 2)
            self.assertIn("traffic_load", df.columns)
            self.assertIn("dl_cqi", df.columns)
            self.assertIn("sum_requested_prbs", df.columns)


if __name__ == "__main__":
    unittest.main()
