import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class TrainE2ETests(unittest.TestCase):
    def test_training_pipeline_on_synthetic(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            data = td_path / "traffic_data.csv"
            model_dir = td_path / "model"

            subprocess.run(
                ["python", "generate_data.py", "--steps", "5000", "--output", str(data)],
                check=True,
            )
            subprocess.run(
                [
                    "python",
                    "-m",
                    "scripts.train",
                    "--data_root",
                    str(data),
                    "--out_dir",
                    str(model_dir),
                    "--seq_len",
                    "32",
                    "--feature_count",
                    "10",
                    "--model",
                    "ridge",
                ],
                check=True,
            )

            metrics_path = model_dir / "metrics.json"
            self.assertTrue(metrics_path.exists())
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertIn("metrics", metrics)
            self.assertIn("test", metrics["metrics"])


if __name__ == "__main__":
    unittest.main()
