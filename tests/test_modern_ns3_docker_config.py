"""Pinned independently verified native backend regression checks.

The regular Python CI checks the dependency and fail-closed configuration.
The separate native CI MUST compile ns-3 and run actual packet transmissions.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agentic_ran.research_modern_ns3_adapter import (
    VERIFIED_NS3_GYM_SHA,
    VERIFIED_NS3_SHA,
)


def test_native_docker_uses_only_verified_modern_source_and_cmake():
    path = Path(__file__).resolve().parent.parent / "Dockerfile.ran-gym"
    if not path.exists():
        pytest.skip("Dockerfile not included in package-only image")
    dockerfile = path.read_text(encoding="utf-8")
    assert "FROM python:3.13-bookworm AS modern-ns3" in dockerfile
    assert f"ARG MODERN_NS3_GYM_SHA={VERIFIED_NS3_GYM_SHA}" in dockerfile
    assert f"ARG MODERN_NS3_SHA={VERIFIED_NS3_SHA}" in dockerfile
    assert "https://github.com/vtavakkoli/ns3-gym.git" in dockerfile
    assert "https://github.com/nsnam/ns-3-dev-git.git" in dockerfile
    assert "cmake -S /opt/ns3 -B /opt/ns3/cmake-build" in dockerfile
    assert "CMAKE_CXX_STANDARD=23" in dockerfile
    assert "python3.13 -m pip install --no-cache-dir ." in dockerfile
    assert "RUN python3.13 -m pytest -q tests/test_modern_ns3_integration.py" in dockerfile
    assert "wineslab" not in dockerfile
    assert "nsoran==0.0.2" not in dockerfile
    assert "tkn-tub/ns3-gym.git" not in dockerfile
    assert "python3.11 ./waf" not in dockerfile


def test_obsolete_adapter_fails_closed():
    from agentic_ran.research_ran_gym_adapters import make_nsoran_traffic_steering

    with pytest.raises(NotImplementedError, match="Legacy"):
        make_nsoran_traffic_steering("scenario-one", 2026, {})


def test_compose_keeps_real_native_and_external_research_distinct():
    path = Path(__file__).resolve().parent.parent / "docker-compose.ran-gym.yml"
    if not path.exists():
        pytest.skip("Compose not present in package-only image")
    compose = path.read_text(encoding="utf-8")
    assert "ran-gym-modern-ns3-smoke:" in compose
    assert "target: modern-ns3" in compose
    assert "ran-gym-nsoran-smoke:" not in compose
    assert "target: nsoran" not in compose
    assert "ran-gym-active:" in compose
    assert "publication-final-gate:" in compose
