"""Command-line interface for dataset, training, decisions, and benchmarks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import uvicorn

from agentic_ran.config import Settings
from agentic_ran.data import download_dataset, generate_dataset, load_dataset, sha256_file, write_dataset
from agentic_ran.domain import NetworkObservation
from agentic_ran.model import train_policy_proposer, write_training_metrics
from agentic_ran.reporting import benchmark
from agentic_ran.service import PolicyService


DEFAULT_DATASET_URL = "https://raw.githubusercontent.com/vtavakkoli/Agentic-RAN/main/data/bootstrap/ran_policy_sample.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentic-ran", description="Safe agentic network-policy selection")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-data", help="Generate the deterministic small RAN dataset")
    generate.add_argument("--output", default="data/runtime/ran_policy_sample.csv")
    generate.add_argument("--rows", type=int, default=1_200)
    generate.add_argument("--seed", type=int, default=42)

    download = subparsers.add_parser("download-data", help="Download the small dataset with an offline fallback")
    download.add_argument("--url", default=os.getenv("AGENTIC_RAN_DATASET_URL", DEFAULT_DATASET_URL))
    download.add_argument("--output", default=os.getenv("AGENTIC_RAN_DATASET", "data/runtime/ran_policy_sample.csv"))
    download.add_argument("--sha256", default=os.getenv("AGENTIC_RAN_DATASET_SHA256"))
    download.add_argument("--no-fallback", action="store_true")
    download.add_argument("--fallback-rows", type=int, default=140)

    train = subparsers.add_parser("train", help="Train the lightweight policy proposer")
    train.add_argument("--data", default=os.getenv("AGENTIC_RAN_DATASET", "data/runtime/ran_policy_sample.csv"))
    train.add_argument("--model", default=os.getenv("AGENTIC_RAN_MODEL", "artifacts/policy_selector.joblib"))
    train.add_argument("--metrics", default="results/training_metrics.json")
    train.add_argument("--seed", type=int, default=42)

    decide = subparsers.add_parser("decide", help="Select a policy for one JSON KPI snapshot")
    source = decide.add_mutually_exclusive_group(required=True)
    source.add_argument("--json", help="Inline JSON observation")
    source.add_argument("--file", help="Path to JSON observation")

    serve = subparsers.add_parser("serve", help="Run the API and web demonstration")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--workers", type=int, default=1)

    bench = subparsers.add_parser("benchmark", help="Run end-to-end policy-selection benchmark")
    bench.add_argument("--data", default=os.getenv("AGENTIC_RAN_DATASET", "data/runtime/ran_policy_sample.csv"))
    bench.add_argument("--output", default="results")
    bench.add_argument("--limit", type=int, default=400)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate-data":
        path = write_dataset(generate_dataset(rows=args.rows, seed=args.seed), args.output)
        print(json.dumps({"dataset": str(path), "sha256": sha256_file(path)}, indent=2))
        return 0
    if args.command == "download-data":
        path, source = download_dataset(
            args.url,
            args.output,
            expected_sha256=args.sha256,
            allow_fallback=not args.no_fallback,
            fallback_rows=args.fallback_rows,
        )
        print(json.dumps({"dataset": str(path), "source": source, "sha256": sha256_file(path)}, indent=2))
        return 0
    if args.command == "train":
        frame = load_dataset(args.data)
        proposer, metrics = train_policy_proposer(frame, seed=args.seed)
        proposer.save(args.model)
        write_training_metrics(metrics, args.metrics)
        print(json.dumps({"model": args.model, **{key: value for key, value in metrics.items() if key != "classification_report"}}, indent=2))
        return 0
    if args.command == "decide":
        payload = json.loads(args.json if args.json is not None else Path(args.file).read_text(encoding="utf-8"))
        decision = PolicyService.load().decide(NetworkObservation(**payload))
        print(decision.model_dump_json(indent=2))
        return 0
    if args.command == "serve":
        uvicorn.run("agentic_ran.api:app", host=args.host, port=args.port, workers=args.workers)
        return 0
    if args.command == "benchmark":
        metrics = benchmark(PolicyService.load(), args.data, args.output, limit=args.limit)
        print(json.dumps(metrics, indent=2))
        return 0
    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
