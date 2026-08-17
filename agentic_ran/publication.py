"""Python 3.13 entrypoint for the final COMMAG publication benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agentic_ran.publication_data import prepare
from agentic_ran.publication_science import load_science_config, run_final_benchmark
from agentic_ran.publication_v2 import PubConfig, load_config

PAPER_REFERENCE: dict[str, Any] = {
    "type": "literature_reference_only",
    "method": "Original COMMAG PPO/DRL scheduler control",
    "paper": {
        "authors": "L. Bonati, S. D'Oro, M. Polese, S. Basagni, T. Melodia",
        "title": "Intelligence and Learning in O-RAN for Data-driven NextG Cellular Networks",
        "venue": "IEEE Communications Magazine",
        "volume": 59,
        "issue": 10,
        "pages": "21-27",
        "year": 2021,
        "doi": "10.1109/MCOM.101.2001120",
        "arxiv": "2012.01263",
    },
    "reported_results": {
        "embb_spectral_efficiency_gain": {
            "value_percent": 20,
            "qualifier": "up to",
            "comparison": "best-performing static scheduling policy",
        },
        "urllc_average_buffer_reduction_percent": {
            "vs_round_robin": 37,
            "vs_waterfilling": 5,
            "vs_proportional_fair": 17,
        },
    },
    "comparison_rule": (
        "Reference values are copied from the published experimental results. They use different metrics, "
        "test conditions and action semantics and therefore must not be inserted into direct-method utility "
        "tables, paired statistical tests, or described as a reproduced baseline."
    ),
}


def evaluate(
    data_path: str | Path,
    output: str | Path,
    pub_cfg: PubConfig,
    science_config: str | Path,
    seed: int | None = None,
) -> dict[str, Any]:
    science = load_science_config(science_config)
    return run_final_benchmark(
        data_path,
        output,
        pub_cfg,
        science,
        seed=seed,
        literature_reference=PAPER_REFERENCE,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Full COMMAG publication benchmark")
    sub = parser.add_subparsers(dest="cmd", required=True)

    prepare_cmd = sub.add_parser("prepare")
    prepare_cmd.add_argument("--raw-dir", default="data/raw/commag")
    prepare_cmd.add_argument("--output", default="data/prepared/commag-publication")
    prepare_cmd.add_argument("--config", default="configs/full_commag_publication.yaml")
    prepare_cmd.add_argument("--workers", type=int, default=4)
    prepare_cmd.add_argument("--max-rows-per-file", type=int, default=0)

    evaluate_cmd = sub.add_parser("evaluate")
    evaluate_cmd.add_argument(
        "--data",
        default="data/prepared/commag-publication/commag_publication_transitions.csv.gz",
    )
    evaluate_cmd.add_argument("--output", default="results/publication")
    evaluate_cmd.add_argument("--config", default="configs/full_commag_publication.yaml")
    evaluate_cmd.add_argument("--science-config", default="configs/final_publication.yaml")
    evaluate_cmd.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)
    pub_cfg = load_config(args.config)
    if args.cmd == "prepare":
        result = prepare(
            args.raw_dir,
            args.output,
            pub_cfg,
            args.workers,
            args.max_rows_per_file or None,
        )
    else:
        result = evaluate(
            args.data,
            args.output,
            pub_cfg,
            args.science_config,
            args.seed or None,
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
