"""Pinned legacy ns-O-RAN/Waf build metadata regression checks.

Tests intentionally do not claim the resource-heavy native C++ source build
was executed in CI. That is the optional Docker native smoke job.
"""
from __future__ import annotations

from pathlib import Path
import sys

from agentic_ran.research_ran_gym_adapters import nsoran_build_status_path


def test_pinned_waf_build_uses_optimized_status_file(tmp_path: Path):
    (tmp_path / "waf").write_text("#!/usr/bin/env python3\n")
    path = nsoran_build_status_path(tmp_path, optimized=True)
    assert path == tmp_path / "build/optimized/build-status.py"
    assert nsoran_build_status_path(
        tmp_path, optimized=False
    ) == tmp_path / "build/build-status.py"


def test_modern_ns3_backend_uses_platform_lock_file(tmp_path: Path):
    (tmp_path / "ns3").write_text("#!/usr/bin/env python3\n")
    assert nsoran_build_status_path(tmp_path) == (
        tmp_path / f".lock-ns3_{sys.platform}_build"
    )


def test_native_dockerfile_uses_pinned_waf_and_rejects_old_ns3_launcher():
    dockerfile = (Path(__file__).resolve().parent.parent / "Dockerfile.ran-gym").read_text()
    native = dockerfile.split("FROM python:3.11-bullseye AS nsoran", 1)[1].split(
        "FROM runtime AS final-algorithms", 1
    )[0]
    assert "python3 ./waf configure --enable-examples" in native
    assert "python3 ./waf build -j" in native
    assert "test -s build/optimized/build-status.py" in native
    assert "grep -q 'scenario-one' build/optimized/build-status.py" in native
    assert "./ns3 configure" not in native
    assert "--disable-werror" not in native.split("RUN test -f ./waf", 1)[1]
