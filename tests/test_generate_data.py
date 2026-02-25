import unittest

from generate_data import generate_synthetic_dataset


class GenerateDataTests(unittest.TestCase):
    def test_row_count_and_split(self):
        df = generate_synthetic_dataset(5000, seed=42)
        self.assertEqual(len(df), 5000)
        counts = df["split"].value_counts().to_dict()
        self.assertEqual(counts.get("train"), 3000)
        self.assertEqual(counts.get("val"), 1500)
        self.assertEqual(counts.get("test"), 500)


if __name__ == "__main__":
    unittest.main()
