"""Pinned Colosseum O-RAN COMMAG preparation and offline-RL benchmark."""

from __future__ import annotations

import hashlib
import html
import json
import math
import shutil
import tempfile
import time
import urllib.request
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score

COMMAG_REPOSITORY = "https://github.com/wineslab/colosseum-oran-commag-dataset"
COMMAG_REVISION = "7331cd725fe42b5b9930fdc8acee3006cea00bd7"
COMMAG_LICENSE = "GPL-3.0"
COMMAG_CITATION = (
    "L. Bonati, S. D'Oro, M. Polese, S. Basagni, and T. Melodia, "
    '"Intelligence and Learning in O-RAN for Data-driven NextG Cellular Networks," '
    "IEEE Communications Magazine, 59(10), 21-27, 2021."
)

SCHEDULERS = {0: "round_robin", 1: "waterfilling", 2: "proportional_fair"}
SLICE_TYPES = {0: "eMBB", 1: "mMTC", 2: "URLLC"}
CORE_UE_FILES = {
    1: ("1010123456002", "1010123456003", "1010123456004"),
    4: ("1010123456035", "1010123456036", "1010123456037"),
}

STATE_COLUMNS = [
    "num_ues",
    "slice_prb",
    "power_multiplier",
    "scheduler_code",
    "dl_mcs",
    "dl_buffer_bytes",
    "dl_bitrate_mbps",
    "dl_errors_pct",
    "dl_cqi",
    "ul_mcs",
    "ul_buffer_bytes",
    "ul_bitrate_mbps",
    "ul_errors_pct",
    "ul_sinr",
    "requested_prbs",
    "granted_prbs",
    "grant_ratio",
]

_COLUMN_MAP = {
    "Timestamp": "timestamp_ms",
    "num_ues": "num_ues",
    "slice_id": "slice_id",
    "slice_prb": "slice_prb",
    "power_multiplier": "power_multiplier",
    "scheduling_policy": "scheduler_code",
    "dl_mcs": "dl_mcs",
    "dl_buffer [bytes]": "dl_buffer_bytes",
    "tx_brate downlink [Mbps]": "dl_bitrate_mbps",
    "tx_errors downlink (%)": "dl_errors_pct",
    "dl_cqi": "dl_cqi",
    "ul_mcs": "ul_mcs",
    "ul_buffer [bytes]": "ul_buffer_bytes",
    "rx_brate uplink [Mbps]": "ul_bitrate_mbps",
    "rx_errors uplink (%)": "ul_errors_pct",
    "ul_sinr": "ul_sinr",
    "sum_requested_prbs": "requested_prbs",
    "sum_granted_prbs": "granted_prbs",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def commag_core_paths(
    train_configs: Sequence[int] = (0, 1, 2),
    experiments: Sequence[int] = (1, 2),
    base_stations: Sequence[int] = (1, 4),
) -> list[str]:
    """Return the pinned compact profile without cloning the 1.2 GB repository."""

    paths: list[str] = []
    for training in train_configs:
        if not 0 <= int(training) <= 17:
            raise ValueError("COMMAG training configurations must be between 0 and 17")
        for experiment in experiments:
            if int(experiment) < 1:
                raise ValueError("COMMAG experiment numbers must be positive")
            for base_station in base_stations:
                ue_files = CORE_UE_FILES.get(int(base_station))
                if ue_files is None:
                    raise ValueError("The compact profile supports base stations 1 and 4")
                for imsi in ue_files:
                    paths.append(
                        "slice_traffic/rome_slow_close/"
                        f"tr{training}/exp{experiment}/bs{base_station}/slices_bs{base_station}/"
                        f"{imsi}_metrics.csv"
                    )
    return paths


def _download_one(relative_path: str, raw_root: Path, revision: str, retries: int = 3) -> Path:
    destination = raw_root / relative_path
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://raw.githubusercontent.com/wineslab/colosseum-oran-commag-dataset/{revision}/{relative_path}"
    last_error: Exception | None = None
    for attempt in range(retries):
        temporary: Path | None = None
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Agentic-RAN/2.2 COMMAG benchmark"})
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                tempfile.NamedTemporaryFile(delete=False) as stream,
            ):
                shutil.copyfileobj(response, stream)
                temporary = Path(stream.name)
            if temporary.stat().st_size == 0:
                raise ValueError(f"Downloaded an empty COMMAG file: {relative_path}")
            shutil.copy2(temporary, destination)
            temporary.unlink(missing_ok=True)
            return destination
        except Exception as exc:  # pragma: no cover - network boundary
            last_error = exc
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Unable to download COMMAG file {relative_path}: {last_error}") from last_error


def download_commag_core(
    raw_dir: Path | str,
    paths: Sequence[str],
    revision: str = COMMAG_REVISION,
    workers: int = 4,
) -> list[Path]:
    """Download only selected immutable blobs from the pinned upstream commit."""

    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision.lower()):
        raise ValueError("COMMAG revision must be a full 40-character commit SHA")
    root = Path(raw_dir)
    root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), 8))) as pool:
        return list(pool.map(lambda item: _download_one(item, root, revision), paths))


def _parse_identifiers(relative_path: str) -> dict[str, str]:
    parts = Path(relative_path).parts
    if len(parts) < 7:
        raise ValueError(f"Unexpected COMMAG path: {relative_path}")
    return {
        "scenario": parts[1],
        "training_config": parts[2],
        "experiment": parts[3],
        "base_station": parts[4],
        "imsi_file": Path(parts[-1]).stem.removesuffix("_metrics"),
    }


def _mode(series: pd.Series) -> float:
    values = series.dropna().mode()
    return float(values.iloc[0]) if not values.empty else 0.0


def _read_commag_trace(path: Path, relative_path: str, max_rows: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, nrows=max_rows or None, low_memory=False)
    missing = [name for name in _COLUMN_MAP if name not in frame.columns]
    if missing:
        raise ValueError(f"COMMAG file {relative_path} is missing columns: {', '.join(missing)}")
    frame = frame[list(_COLUMN_MAP)].rename(columns=_COLUMN_MAP)
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["timestamp_ms", "slice_id", "slice_prb", "scheduler_code"])
    frame = frame[frame["slice_id"].isin(SLICE_TYPES) & frame["scheduler_code"].isin(SCHEDULERS)]
    if frame.empty:
        raise ValueError(f"COMMAG file {relative_path} contains no usable sliced observations")

    frame["timestamp_s"] = (frame["timestamp_ms"] // 1000).astype("int64")
    mean_columns = [
        column
        for column in STATE_COLUMNS
        if column not in {"slice_prb", "scheduler_code", "requested_prbs", "granted_prbs", "grant_ratio"}
    ]
    aggregations: dict[str, str | Any] = {column: "mean" for column in mean_columns}
    aggregations.update(
        {
            "slice_id": _mode,
            "slice_prb": _mode,
            "scheduler_code": _mode,
            "requested_prbs": "sum",
            "granted_prbs": "sum",
        }
    )
    grouped = frame.groupby("timestamp_s", sort=True, as_index=False).agg(aggregations)
    grouped = grouped.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    grouped["grant_ratio"] = np.where(
        grouped["requested_prbs"] > 0,
        grouped["granted_prbs"] / grouped["requested_prbs"],
        1.0,
    ).clip(0.0, 1.0)
    identifiers = _parse_identifiers(relative_path)
    for name, value in identifiers.items():
        grouped[name] = value
    grouped["slice_type"] = grouped["slice_id"].round().astype(int).map(SLICE_TYPES)
    grouped["episode_id"] = (
        grouped["scenario"]
        + "/"
        + grouped["training_config"]
        + "/"
        + grouped["experiment"]
        + "/"
        + grouped["base_station"]
        + "/"
        + grouped["imsi_file"]
    )
    grouped["source_path"] = relative_path
    return grouped


def _action(scheduler_code: float, slice_prb: float) -> str:
    scheduler_number = round(scheduler_code)
    scheduler = SCHEDULERS.get(scheduler_number, f"scheduler_{scheduler_number}")
    prbs = round(slice_prb)
    return f"{scheduler}:prb={prbs}"


def _reward(slice_type: str, downlink_mbps: float, grant_ratio: float, dl_errors_pct: float) -> float:
    if slice_type == "eMBB":
        objective = min(max(downlink_mbps / 1.0, 0.0), 1.0)
    elif slice_type == "mMTC":
        objective = min(max(downlink_mbps / 0.03, 0.0), 1.0)
    else:
        objective = min(max(grant_ratio, 0.0), 1.0)
    error_penalty = min(max(dl_errors_pct / 100.0, 0.0), 1.0)
    return float(np.clip(objective - 0.25 * error_penalty, -0.25, 1.0))


def _to_transitions(measurements: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, episode in measurements.groupby("episode_id", sort=True):
        episode = episode.sort_values("timestamp_s").reset_index(drop=True)
        if len(episode) < 2:
            continue
        current = episode.iloc[:-1].copy()
        following = episode.iloc[1:].reset_index(drop=True)
        for column in STATE_COLUMNS:
            current[f"next_{column}"] = following[column].to_numpy()
        current["action"] = [
            _action(scheduler, prbs)
            for scheduler, prbs in zip(following["scheduler_code"], following["slice_prb"], strict=True)
        ]
        current["reward"] = [
            _reward(slice_type, downlink, ratio, errors)
            for slice_type, downlink, ratio, errors in zip(
                following["slice_type"],
                following["dl_bitrate_mbps"],
                following["grant_ratio"],
                following["dl_errors_pct"],
                strict=True,
            )
        ]
        current["done"] = False
        current.loc[current.index[-1], "done"] = True
        current["split"] = np.where(current["experiment"].eq("exp1"), "train", "test")
        rows.append(current)
    if not rows:
        raise ValueError("COMMAG preparation produced no sequential transitions")
    transitions = pd.concat(rows, ignore_index=True)
    ordered = [
        "episode_id",
        "timestamp_s",
        "scenario",
        "training_config",
        "experiment",
        "base_station",
        "imsi_file",
        "source_path",
        "slice_type",
        *STATE_COLUMNS,
        "action",
        "reward",
        *[f"next_{column}" for column in STATE_COLUMNS],
        "done",
        "split",
    ]
    return transitions[ordered]


def prepare_commag_data(
    raw_dir: Path | str = "data/raw/commag",
    output_dir: Path | str = "data/prepared/commag",
    revision: str = COMMAG_REVISION,
    train_configs: Sequence[int] = (0, 1, 2),
    experiments: Sequence[int] = (1, 2),
    base_stations: Sequence[int] = (1, 4),
    workers: int = 4,
    source_dir: Path | str | None = None,
    source_paths: Sequence[str] | None = None,
    max_rows_per_file: int | None = None,
) -> dict[str, Any]:
    """Prepare a compressed, experiment-split transition table from COMMAG."""

    paths = list(source_paths or commag_core_paths(train_configs, experiments, base_stations))
    raw_root = Path(source_dir) if source_dir is not None else Path(raw_dir)
    files = (
        [raw_root / relative_path for relative_path in paths]
        if source_dir is not None
        else download_commag_core(raw_root, paths, revision, workers)
    )
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing COMMAG source files: {', '.join(missing)}")

    measurements = pd.concat(
        [
            _read_commag_trace(path, relative_path, max_rows=max_rows_per_file)
            for path, relative_path in zip(files, paths, strict=True)
        ],
        ignore_index=True,
    )
    transitions = _to_transitions(measurements)
    if set(transitions["split"]) != {"train", "test"}:
        raise ValueError("COMMAG core requires both exp1 training and exp2 held-out transitions")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    dataset_path = destination / "commag_transitions.csv.gz"
    transitions.to_csv(
        dataset_path,
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    source_records = [
        {
            "path": relative_path,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path, relative_path in zip(files, paths, strict=True)
    ]
    train_episodes = set(transitions.loc[transitions["split"].eq("train"), "episode_id"])
    test_episodes = set(transitions.loc[transitions["split"].eq("test"), "episode_id"])
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_repository": COMMAG_REPOSITORY,
        "source_revision": revision,
        "source_license": COMMAG_LICENSE,
        "citation": COMMAG_CITATION,
        "raw_files_redistributed": False,
        "profile": "core",
        "raw_files": len(files),
        "raw_bytes": sum(item["bytes"] for item in source_records),
        "prepared_file": str(dataset_path),
        "prepared_bytes": dataset_path.stat().st_size,
        "prepared_sha256": _sha256(dataset_path),
        "rows": len(transitions),
        "train_rows": int(transitions["split"].eq("train").sum()),
        "test_rows": int(transitions["split"].eq("test").sum()),
        "train_episodes": len(train_episodes),
        "test_episodes": len(test_episodes),
        "episode_overlap": len(train_episodes & test_episodes),
        "actions": transitions["action"].value_counts().sort_index().to_dict(),
        "slices": transitions["slice_type"].value_counts().sort_index().to_dict(),
        "source_files": source_records,
        "important_note": (
            "This is an observational offline-RL benchmark. Experiment-separated testing measures value-model "
            "generalization; it is not causal evidence of live-network improvement."
        ),
    }
    (destination / "commag_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _validate_transitions(frame: pd.DataFrame) -> None:
    required = {
        "episode_id",
        "timestamp_s",
        "slice_type",
        "action",
        "reward",
        "done",
        "split",
        *STATE_COLUMNS,
        *[f"next_{column}" for column in STATE_COLUMNS],
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"COMMAG transition dataset is missing columns: {', '.join(missing)}")
    if not set(frame["split"]).issuperset({"train", "test"}):
        raise ValueError("COMMAG transition dataset requires train and test splits")
    numeric = [*STATE_COLUMNS, *[f"next_{column}" for column in STATE_COLUMNS], "reward"]
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("COMMAG transition dataset contains non-finite values")


def _encode(
    frame: pd.DataFrame,
    actions: Sequence[str],
    action_values: Iterable[str],
    prefix: str = "",
) -> np.ndarray:
    numeric = frame[[f"{prefix}{column}" for column in STATE_COLUMNS]].to_numpy(dtype=float)
    slice_values = np.column_stack(
        [frame["slice_type"].eq(name).to_numpy(dtype=float) for name in SLICE_TYPES.values()]
    )
    action_series = pd.Series(list(action_values), index=frame.index)
    action_values_encoded = np.column_stack([action_series.eq(action).to_numpy(dtype=float) for action in actions])
    return np.column_stack([numeric, slice_values, action_values_encoded])


def _prediction_matrix(
    model: ExtraTreesRegressor,
    frame: pd.DataFrame,
    actions: Sequence[str],
    prefix: str = "",
    actions_by_slice: dict[str, set[str]] | None = None,
) -> np.ndarray:
    predictions = np.column_stack(
        [model.predict(_encode(frame, actions, [action] * len(frame), prefix)) for action in actions]
    )
    if actions_by_slice is None:
        return predictions
    allowed = np.column_stack(
        [
            frame["slice_type"]
            .map(lambda name, candidate=action: candidate in actions_by_slice.get(str(name), set()))
            .to_numpy()
            for action in actions
        ]
    )
    return np.where(allowed, predictions, -np.inf)


def _max_q(
    model: ExtraTreesRegressor,
    frame: pd.DataFrame,
    actions: Sequence[str],
    prefix: str,
    actions_by_slice: dict[str, set[str]],
) -> np.ndarray:
    return np.max(_prediction_matrix(model, frame, actions, prefix, actions_by_slice), axis=1)


def _discounted_returns(frame: pd.DataFrame, gamma: float, horizon: int) -> np.ndarray:
    output = np.zeros(len(frame), dtype=float)
    for _, indices in frame.groupby("episode_id", sort=False).groups.items():
        ordered = list(indices)
        rewards = frame.loc[ordered, "reward"].to_numpy(dtype=float)
        for position, index in enumerate(ordered):
            available = rewards[position : position + max(1, horizon)]
            discounts = gamma ** np.arange(len(available))
            output[index] = float(np.dot(available, discounts))
    return output


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def train_commag_fitted_q(
    data_path: Path | str = "data/prepared/commag/commag_transitions.csv.gz",
    model_path: Path | str = "artifacts/commag_fitted_q.joblib",
    metrics_path: Path | str = "results/commag_benchmark.json",
    report_path: Path | str = "results/commag_report.html",
    manifest_path: Path | str | None = "data/prepared/commag/commag_manifest.json",
    iterations: int = 6,
    gamma: float = 0.97,
    seed: int = 42,
) -> dict[str, Any]:
    """Train Fitted-Q and test value generalization on held-out experiments."""

    frame = pd.read_csv(data_path)
    _validate_transitions(frame)
    manifest = None
    if manifest_path is not None and Path(manifest_path).exists():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    train = frame[frame["split"].eq("train")].reset_index(drop=True)
    test = frame[frame["split"].eq("test")].reset_index(drop=True)
    actions = sorted(train["action"].unique())
    if len(train) < 100 or len(test) < 100 or len(actions) < 3:
        raise ValueError("COMMAG benchmark needs at least 100 rows per split and three supported actions")
    unsupported = sorted(set(test["action"]) - set(actions))
    if unsupported:
        raise ValueError(f"Held-out COMMAG split contains unsupported actions: {', '.join(unsupported)}")
    actions_by_slice = {
        str(slice_type): set(group["action"].astype(str))
        for slice_type, group in train.groupby("slice_type", sort=True)
    }

    x_train = _encode(train, actions, train["action"])
    rewards = train["reward"].to_numpy(dtype=float)
    targets = rewards.copy()
    q_model: ExtraTreesRegressor | None = None
    for _ in range(max(1, int(iterations))):
        q_model = ExtraTreesRegressor(
            n_estimators=140,
            min_samples_leaf=3,
            max_features=0.85,
            random_state=seed,
            n_jobs=-1,
        )
        q_model.fit(x_train, targets)
        continuation = _max_q(q_model, train, actions, "next_", actions_by_slice)
        targets = rewards + gamma * continuation * (~train["done"].astype(bool)).to_numpy(dtype=float)
    assert q_model is not None

    reward_model = ExtraTreesRegressor(
        n_estimators=140,
        min_samples_leaf=3,
        max_features=0.85,
        random_state=seed + 1,
        n_jobs=-1,
    )
    reward_model.fit(x_train, rewards)

    x_test = _encode(test, actions, test["action"])
    logged_q = q_model.predict(x_test)
    evaluation_horizon = max(1, int(iterations))
    returns = _discounted_returns(test, gamma, evaluation_horizon)
    next_q = _max_q(q_model, test, actions, "next_", actions_by_slice)
    bellman_target = test["reward"].to_numpy(dtype=float) + gamma * next_q * (~test["done"].astype(bool)).to_numpy(
        dtype=float
    )
    action_q = _prediction_matrix(q_model, test, actions, actions_by_slice=actions_by_slice)
    selected_index = np.argmax(action_q, axis=1)
    selected_actions = np.asarray(actions, dtype=object)[selected_index]
    reward_predictions = _prediction_matrix(reward_model, test, actions, actions_by_slice=actions_by_slice)
    selected_reward = reward_predictions[np.arange(len(test)), selected_index]
    logged_reward_estimate = reward_model.predict(x_test)

    train_episodes = set(train["episode_id"])
    test_episodes = set(test["episode_id"])
    structural_checks = {
        "at least 100 training transitions": len(train) >= 100,
        "at least 100 held-out transitions": len(test) >= 100,
        "at least three discrete actions": len(actions) >= 3,
        "training and test episodes are disjoint": not bool(train_episodes & test_episodes),
        "all held-out actions have training support": not unsupported,
        "all rewards and predictions are finite": bool(
            np.isfinite(np.concatenate([returns, logged_q, bellman_target, selected_reward])).all()
        ),
    }
    metrics = {
        "benchmark": "commag-offline-fitted-q",
        "verdict": "BENCHMARK-READY" if all(structural_checks.values()) else "INVALID",
        "algorithm": "fitted_q_iteration_extra_trees",
        "source_repository": (manifest or {}).get("source_repository", COMMAG_REPOSITORY),
        "source_revision": (manifest or {}).get("source_revision", COMMAG_REVISION),
        "source_license": (manifest or {}).get("source_license", COMMAG_LICENSE),
        "data_path": str(data_path),
        "data_sha256": _sha256(Path(data_path)),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_episodes": len(train_episodes),
        "test_episodes": len(test_episodes),
        "actions": actions,
        "action_support": train["action"].value_counts().sort_index().to_dict(),
        "gamma": gamma,
        "iterations": max(1, int(iterations)),
        "seed": seed,
        "held_out": {
            "reward_model_mae": float(mean_absolute_error(test["reward"], logged_reward_estimate)),
            "reward_model_r2": _finite(r2_score(test["reward"], logged_reward_estimate)),
            "q_truncated_return_horizon": evaluation_horizon,
            "q_truncated_return_mae": float(mean_absolute_error(returns, logged_q)),
            "q_truncated_return_r2": _finite(r2_score(returns, logged_q)),
            "bellman_mae": float(mean_absolute_error(bellman_target, logged_q)),
            "logged_action_agreement": float(np.mean(selected_actions == test["action"].to_numpy())),
            "logged_mean_reward": float(test["reward"].mean()),
            "direct_method_selected_mean_reward": float(np.mean(selected_reward)),
            "direct_method_estimated_uplift": float(np.mean(selected_reward - logged_reward_estimate)),
        },
        "selected_action_distribution": pd.Series(selected_actions).value_counts().sort_index().to_dict(),
        "structural_checks": structural_checks,
        "limitations": [
            "The compact profile covers one RF scenario and is not a production network sample.",
            "Direct-method uplift is model-estimated and must not be presented as causal online improvement.",
            "PPO is not trained from fixed logs; it requires an interactive validated environment.",
        ],
    }
    model_destination = Path(model_path)
    model_destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "q_model": q_model,
            "reward_model": reward_model,
            "actions": actions,
            "actions_by_slice": {name: sorted(values) for name, values in actions_by_slice.items()},
            "state_columns": STATE_COLUMNS,
            "metrics": metrics,
        },
        model_destination,
        compress=3,
    )
    metrics["model_path"] = str(model_destination)
    metrics["model_sha256"] = _sha256(model_destination)
    metrics_destination = Path(metrics_path)
    metrics_destination.parent.mkdir(parents=True, exist_ok=True)
    metrics_destination.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report_destination = Path(report_path)
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    report_destination.write_text(_render_commag_report(metrics, manifest), encoding="utf-8")
    return metrics


def validate_commag_benchmark(metrics_path: Path | str) -> tuple[bool, list[str]]:
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    checks = metrics.get("structural_checks", {})
    errors = [name for name, passed in checks.items() if not passed]
    for required in ("model_sha256", "data_sha256", "held_out"):
        if required not in metrics:
            errors.append(f"missing metrics field: {required}")
    return not errors, errors


def _render_commag_report(metrics: dict[str, Any], manifest: dict[str, Any] | None) -> str:
    held_out = metrics["held_out"]
    checks = "".join(
        f"<li>{'PASS' if passed else 'FAIL'} — {html.escape(name)}</li>"
        for name, passed in metrics["structural_checks"].items()
    )
    actions = "".join(
        f"<tr><td>{html.escape(action)}</td><td>{count}</td>"
        f"<td>{metrics['selected_action_distribution'].get(action, 0)}</td></tr>"
        for action, count in metrics["action_support"].items()
    )
    provenance = ""
    if manifest:
        ratio = manifest["prepared_bytes"] / max(manifest["raw_bytes"], 1)
        provenance = (
            f"<p>Downloaded core: {manifest['raw_files']} files / {manifest['raw_bytes'] / 1048576:.2f} MiB. "
            f"Compressed transitions: {manifest['prepared_bytes'] / 1048576:.2f} MiB "
            f"({ratio * 100:.1f}% of selected raw bytes).</p>"
        )
    limitations = "".join(f"<li>{html.escape(item)}</li>" for item in metrics["limitations"])
    cards = "".join(
        [
            "<div class='card'><div>Algorithm</div><div class='value'>Fitted-Q</div></div>",
            f"<div class='card'><div>Train rows</div><div class='value'>{metrics['train_rows']}</div></div>",
            f"<div class='card'><div>Held-out rows</div><div class='value'>{metrics['test_rows']}</div></div>",
            f"<div class='card'><div>Actions</div><div class='value'>{len(metrics['actions'])}</div></div>",
            "<div class='card'><div>Reward-model MAE</div>"
            f"<div class='value'>{held_out['reward_model_mae']:.3f}</div></div>",
            f"<div class='card'><div>Bellman MAE</div><div class='value'>{held_out['bellman_mae']:.3f}</div></div>",
        ]
    )
    style = """
<style>
body { margin:0; background:#f3f6f9; color:#17212b; font:15px/1.55 Inter,system-ui,sans-serif; }
main { max-width:1120px; margin:auto; padding:38px 20px; }
h1 { font-size:40px; margin:4px 0; }
.eyebrow { text-transform:uppercase; letter-spacing:.12em; color:#52687b; font-weight:700; }
.verdict { display:inline-block; padding:9px 13px; border-radius:10px; background:#d1fae5;
  color:#065f46; font-weight:800; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:14px; margin:24px 0; }
.card, section { background:#fff; border:1px solid #dce4eb; border-radius:14px; padding:18px;
  margin:14px 0; box-shadow:0 8px 30px #2030400b; }
.value { font-size:27px; font-weight:800; color:#086b61; }
table { width:100%; border-collapse:collapse; }
th, td { padding:9px; border-bottom:1px solid #e7edf2; text-align:left; }
th { color:#536879; }
.warn { background:#fff7ed; border-left:4px solid #f59e0b; padding:12px; }
code { background:#edf2f6; padding:2px 5px; border-radius:5px; }
</style>
"""
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Agentic-RAN COMMAG Offline-RL Report</title>{style}</head><body><main>"
        "<div class='eyebrow'>Pinned real-data profile · experiment-separated offline RL</div>"
        f"<h1>COMMAG benchmark report</h1><p class='verdict'>{metrics['verdict']}</p>"
        f"<div class='grid'>{cards}</div>"
        f"<section><h2>Validity gates</h2><ul>{checks}</ul></section>"
        "<section><h2>Held-out experiment evaluation</h2>"
        f"<pre>{html.escape(json.dumps(held_out, indent=2))}</pre>"
        "<p class='warn'><strong>Interpretation:</strong> Direct-method uplift is a reward-model estimate, "
        "not an observed intervention effect.</p></section>"
        "<section><h2>Action support</h2><table><thead><tr><th>Action</th>"
        "<th>Training rows</th><th>Selected on test</th></tr></thead>"
        f"<tbody>{actions}</tbody></table></section>"
        "<section><h2>Provenance and compression</h2>"
        f"<p>Source: <a href='{COMMAG_REPOSITORY}'>Colosseum O-RAN COMMAG</a> at commit "
        f"<code>{metrics['source_revision']}</code>, license {COMMAG_LICENSE}.</p>"
        f"{provenance}<p>{html.escape(COMMAG_CITATION)}</p></section>"
        f"<section><h2>Limitations</h2><ul>{limitations}</ul></section>"
        "</main></body></html>"
    )
