"""Python 3.13 publication benchmark entrypoint.

The executable benchmark evaluates only methods that are reproducible in the current
runtime. The original COMMAG PPO result is retained as a literature reference from
the published paper and is never mixed with direct-method utility comparisons.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agentic_ran.publication_v2 import PubConfig, evaluate as evaluate_reproducible, load_config, prepare


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
    cfg: PubConfig,
    seed: int | None = None,
) -> dict[str, Any]:
    result = evaluate_reproducible(data_path, output, cfg, seed, ppo_export=None)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)

    result["verdict"] = "PUBLICATION-BENCHMARK-READY"
    result.pop("original_ppo_available", None)
    result["literature_reference"] = PAPER_REFERENCE
    result["limitations"] = [
        item
        for item in result.get("limitations", [])
        if "original COMMAG PPO" not in item
    ]
    result["limitations"].append(
        "The original COMMAG PPO numbers are literature-reference results and are not reproduced or used in paired tests."
    )

    (destination / "literature_reference.json").write_text(
        json.dumps(PAPER_REFERENCE, indent=2), encoding="utf-8"
    )
    (destination / "publication_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


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
        "--data", default="data/prepared/commag-publication/commag_publication_transitions.csv.gz"
    )
    evaluate_cmd.add_argument("--output", default="results/publication")
    evaluate_cmd.add_argument("--config", default="configs/full_commag_publication.yaml")
    evaluate_cmd.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.cmd == "prepare":
        result = prepare(args.raw_dir, args.output, cfg, args.workers, args.max_rows_per_file or None)
    else:
        result = evaluate(args.data, args.output, cfg, args.seed or None)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
