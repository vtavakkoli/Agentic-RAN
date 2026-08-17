"""Resilient data preparation for the full COMMAG publication benchmark.

Tree discovery intentionally uses the Git protocol instead of GitHub's recursive
REST tree endpoint. Only commit/tree metadata are fetched (`--filter=blob:none`);
the existing COMMAG downloader still downloads individual CSV blobs on demand and
reuses non-empty files already present under ``data/raw/commag``.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from agentic_ran.commag import (
    COMMAG_REPOSITORY,
    COMMAG_REVISION,
    _read_commag_trace,
    _sha256,
    _to_transitions,
    download_commag_core,
)
from agentic_ran.publication_v2 import PubConfig, _split, filter_paths


def _git(
    args: list[str],
    *,
    check: bool = True,
    timeout: int = 240,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    process = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    if check and process.returncode != 0:
        stderr = process.stderr.strip()
        stdout = process.stdout.strip()
        detail = stderr or stdout or f"exit code {process.returncode}"
        raise RuntimeError(f"Git COMMAG tree discovery failed: {detail[-3000:]}")
    return process


def discover_tree(raw_dir: str | Path) -> list[str]:
    """Return all paths at the pinned COMMAG revision without fetching file blobs."""

    root = Path(raw_dir)
    root.mkdir(parents=True, exist_ok=True)
    cache = root / f".tree-git-{COMMAG_REVISION}.txt"
    if cache.exists() and cache.stat().st_size > 0:
        paths = [line.strip() for line in cache.read_text(encoding="utf-8").splitlines() if line.strip()]
        if paths:
            return paths

    git_dir = root / ".commag-tree.git"
    if not (git_dir / "HEAD").exists():
        _git(["init", "--bare", str(git_dir)])

    repository = COMMAG_REPOSITORY.rstrip("/")
    if not repository.endswith(".git"):
        repository += ".git"

    remote = _git(["--git-dir", str(git_dir), "remote", "get-url", "origin"], check=False)
    if remote.returncode == 0:
        if remote.stdout.strip() != repository:
            _git(["--git-dir", str(git_dir), "remote", "set-url", "origin", repository])
    else:
        _git(["--git-dir", str(git_dir), "remote", "add", "origin", repository])

    has_revision = _git(
        ["--git-dir", str(git_dir), "cat-file", "-e", f"{COMMAG_REVISION}^{{commit}}"],
        check=False,
    )
    if has_revision.returncode != 0:
        _git(
            [
                "--git-dir",
                str(git_dir),
                "-c",
                "protocol.version=2",
                "fetch",
                "--no-tags",
                "--depth=1",
                "--filter=blob:none",
                "origin",
                COMMAG_REVISION,
            ],
            timeout=600,
        )

    listing = _git(
        ["--git-dir", str(git_dir), "ls-tree", "-r", "--name-only", COMMAG_REVISION],
        timeout=240,
    )
    paths = [line.strip() for line in listing.stdout.splitlines() if line.strip()]
    if not paths:
        raise RuntimeError("Pinned COMMAG Git tree is empty")

    tmp = cache.with_suffix(cache.suffix + ".tmp")
    tmp.write_text("\n".join(paths) + "\n", encoding="utf-8")
    tmp.replace(cache)
    return paths


def prepare(
    raw_dir: str | Path,
    output: str | Path,
    cfg: PubConfig,
    workers: int = 4,
    max_rows: int | None = None,
) -> dict[str, Any]:
    paths = filter_paths(discover_tree(raw_dir), cfg)
    files = download_commag_core(raw_dir, paths, revision=COMMAG_REVISION, workers=workers)
    obs = pd.concat(
        [_read_commag_trace(file, path, max_rows=max_rows) for file, path in zip(files, paths, strict=True)],
        ignore_index=True,
    )
    data = _split(_to_transitions(obs), cfg)

    required = {"train", "validation", "test_seen", "test_unseen"}
    if not required.issubset(set(data.publication_split)):
        raise ValueError("one or more publication splits are empty")

    episode_sets = {
        split: set(group.episode_id.astype(str))
        for split, group in data.groupby("publication_split")
    }
    overlap = {
        f"{left}__{right}": len(episode_sets[left] & episode_sets[right])
        for index, left in enumerate(sorted(episode_sets))
        for right in sorted(episode_sets)[index + 1 :]
    }
    if any(overlap.values()):
        raise ValueError(f"episode leakage: {overlap}")

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    dataset = destination / "commag_publication_transitions.csv.gz"
    data.to_csv(
        dataset,
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )

    manifest = {
        "source_repository": COMMAG_REPOSITORY,
        "source_revision": COMMAG_REVISION,
        "tree_discovery": "git-protocol-v2-blobless",
        "profile": "full-slice-traffic-publication",
        "raw_files": len(files),
        "raw_bytes": int(sum(file.stat().st_size for file in files)),
        "rows": len(data),
        "scenarios": sorted(data.scenario.unique()),
        "training_configs": sorted(data.training_config.unique()),
        "base_stations": sorted(data.base_station.unique()),
        "experiments": sorted(data.experiment.unique()),
        "split_rows": data.publication_split.value_counts().sort_index().to_dict(),
        "split_episodes": data.groupby("publication_split").episode_id.nunique().to_dict(),
        "episode_overlap": overlap,
        "prepared_sha256": _sha256(dataset),
    }
    (destination / "commag_publication_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest
