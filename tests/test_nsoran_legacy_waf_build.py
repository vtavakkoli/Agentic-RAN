"""Pinned legacy ns-O-RAN/Waf build metadata regression checks.

Tests intentionally do not claim the resource-heavy native C++ source build
was executed in CI. That is the optional Docker native smoke job.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

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
    path = Path(__file__).resolve().parent.parent / "Dockerfile.ran-gym"
    if not path.exists():
        pytest.skip("Dockerfile not copied into installed-package-only test image")
    dockerfile = path.read_text()
    native = dockerfile.split("FROM python:3.13-bookworm AS nsoran", 1)[1].split(
        "FROM runtime AS final-algorithms", 1
    )[0]
    assert "bullseye-security" not in native
    assert "apt-get -o Acquire::Retries=3 update" in native
    # Waf's pinned wscript appends CXXFLAGS_EXTRA after the flags from
    # its own build profile. Do not globally disable -Werror here.
    assert 'ARG NSORAN_WAF_CXXFLAGS_EXTRA="-Wno-error=use-after-free"' in native
    assert 'ENV CXXFLAGS_EXTRA="${NSORAN_WAF_CXXFLAGS_EXTRA}"' in native
    assert "--disable-werror" not in native.split("RUN test -f ./waf", 1)[1].split("WORKDIR /workspace", 1)[0]
    assert "-Wno-use-after-free" not in native
    assert "python3.11 python3.11-dev python3-distutils" in native
    assert "/usr/bin/python3.11 -c 'import pipes'" in native
    assert "/usr/bin/python3.11 ./waf configure --enable-examples" in native
    assert '/usr/bin/python3.11 ./waf build -j' in native
    assert '&& python3 ./waf configure' not in native
    assert '&& python3 ./waf build' not in native
    assert 'RUN python -m pip install --no-cache-dir ".[ran-gym,dev]" "nsoran==0.0.2"' in native
    assert "test -s build/optimized/build-status.py" in native
    assert "grep -q 'scenario-one' build/optimized/build-status.py" in native
    assert "./ns3 configure" not in native
    assert "/usr/bin/python3.11 ./waf configure --enable-examples --disable-werror" not in native
