from __future__ import annotations

from scripts.generate_data import main as generate_data
from scripts.train import main as train
from scripts.test import main as test


def main() -> None:
    print("[run-all] Step 1/3: generate_data")
    generate_data()
    print("[run-all] Step 2/3: train")
    train()
    print("[run-all] Step 3/3: test")
    test()
    print("[run-all] Pipeline complete")


if __name__ == "__main__":
    main()
