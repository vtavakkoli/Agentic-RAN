"""Command-line interface for training, decisions, control loops, audit, and resilience benchmarks."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path

import uvicorn

from agentic_ran.config import Settings
from agentic_ran.data import download_dataset, generate_dataset, load_dataset, sha256_file, write_dataset
from agentic_ran.domain import ExecutionMode, NetworkObservation
from agentic_ran.model import train_policy_proposer, write_training_metrics
from agentic_ran.reporting import benchmark
from agentic_ran.scenarios import ResilienceBenchmark
from agentic_ran.service import PolicyService
from agentic_ran.telemetry import SrsRANWebSocketProvider, SyntheticTelemetryProvider


DEFAULT_DATASET_URL = "https://raw.githubusercontent.com/vtavakkoli/Agentic-RAN/main/data/bootstrap/ran_policy_sample.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentic-ran", description="Safety-governed O-RAN intelligence")
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

    decide = subparsers.add_parser("decide", help="Select a governed policy for one JSON KPI snapshot")
    source = decide.add_mutually_exclusive_group(required=True)
    source.add_argument("--json", help="Inline JSON observation")
    source.add_argument("--file", help="Path to JSON observation")
    decide.add_argument("--intent", default=None)

    serve = subparsers.add_parser("serve", help="Run the API and operations dashboard")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--workers", type=int, default=1)

    bridge = subparsers.add_parser("serve-bridge", help="Run the local development E2/twin bridge")
    bridge.add_argument("--host", default="0.0.0.0")
    bridge.add_argument("--port", type=int, default=8090)

    control = subparsers.add_parser("control-step", help="Execute one synthetic/replay-safe control-loop step")
    control.add_argument("--mode", choices=[item.value for item in ExecutionMode], default="simulated")
    control.add_argument("--intent", default="balanced")

    live = subparsers.add_parser("xapp-shadow", help="Run shadow decisions from live srsRAN JSON metrics")
    live.add_argument("--url", default=os.getenv("SRSRAN_WS_URL", "ws://127.0.0.1:8001"))
    live.add_argument("--steps", type=int, default=10)
    live.add_argument("--intent", default="balanced")

    bench = subparsers.add_parser("benchmark", help="Run the legacy end-to-end policy-selection benchmark")
    bench.add_argument("--data", default=os.getenv("AGENTIC_RAN_DATASET", "data/runtime/ran_policy_sample.csv"))
    bench.add_argument("--output", default="results")
    bench.add_argument("--limit", type=int, default=400)

    resilience = subparsers.add_parser("resilience", help="Run all built-in fault scenarios for one observation")
    source2 = resilience.add_mutually_exclusive_group(required=True)
    source2.add_argument("--json", help="Inline JSON observation")
    source2.add_argument("--file", help="Path to JSON observation")

    subparsers.add_parser("audit-verify", help="Verify the tamper-evident decision audit chain")
    return parser


def _payload(args: argparse.Namespace) -> dict:
    return json.loads(args.json if args.json is not None else Path(args.file).read_text(encoding="utf-8"))


async def _control_step(mode: ExecutionMode, intent: str):
    service = PolicyService.load()
    loop = service.control_loop(SyntheticTelemetryProvider(period_seconds=0), mode=mode)
    return await loop.step(intent=intent)


async def _shadow(url: str, steps: int, intent: str):
    service = PolicyService.load()
    loop = service.control_loop(SrsRANWebSocketProvider(url), mode=ExecutionMode.SHADOW)
    output = []
    for _ in range(steps):
        result = await loop.step(intent=intent)
        output.append({"cell": result.observation.cell_id, "policy": result.decision.selected_policy, "confidence": result.decision.confidence})
    return output


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate-data":
        path = write_dataset(generate_dataset(rows=args.rows, seed=args.seed), args.output)
        print(json.dumps({"dataset": str(path), "sha256": sha256_file(path)}, indent=2))
        return 0
    if args.command == "download-data":
        path, source = download_dataset(args.url, args.output, expected_sha256=args.sha256, allow_fallback=not args.no_fallback, fallback_rows=args.fallback_rows)
        print(json.dumps({"dataset": str(path), "source": source, "sha256": sha256_file(path)}, indent=2))
        return 0
    if args.command == "train":
        frame = load_dataset(args.data)
        proposer, metrics = train_policy_proposer(frame, seed=args.seed)
        proposer.save(args.model)
        write_training_metrics(metrics, args.metrics)
        summary = {key: value for key, value in metrics.items() if key != "classification_report"}
        print(json.dumps({"model": args.model, **summary}, indent=2))
        return 0
    if args.command == "decide":
        decision = PolicyService.load().decide(NetworkObservation(**_payload(args)), intent=args.intent)
        print(decision.model_dump_json(indent=2))
        return 0
    if args.command == "serve":
        uvicorn.run("agentic_ran.api:app", host=args.host, port=args.port, workers=args.workers)
        return 0
    if args.command == "serve-bridge":
        uvicorn.run("agentic_ran.bridge:app", host=args.host, port=args.port, workers=1)
        return 0
    if args.command == "benchmark":
        metrics = benchmark(PolicyService.load(), args.data, args.output, limit=args.limit)
        print(json.dumps(metrics, indent=2))
        return 0
    if args.command == "control-step":
        result = asyncio.run(_control_step(ExecutionMode(args.mode), args.intent))
        payload = {
            "observation": result.observation.model_dump(mode="json"),
            "decision": result.decision.model_dump(mode="json"),
            "action_result": result.action_result.model_dump(mode="json"),
        }
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.command == "xapp-shadow":
        print(json.dumps(asyncio.run(_shadow(args.url, args.steps, args.intent)), indent=2))
        return 0
    if args.command == "resilience":
        results, summary = ResilienceBenchmark(PolicyService.load()).run(NetworkObservation(**_payload(args)))
        print(json.dumps({"summary": summary, "scenarios": [asdict(item) for item in results]}, indent=2))
        return 0
    if args.command == "audit-verify":
        valid, errors = PolicyService.load().audit.verify()
        print(json.dumps({"valid": valid, "errors": errors}, indent=2))
        return 0 if valid else 2
    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
