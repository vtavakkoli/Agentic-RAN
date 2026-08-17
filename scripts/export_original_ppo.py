from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tf_agents.trajectories import time_step as ts

SLICE_NAMES = {0: "eMBB", 1: "mMTC", 2: "URLLC"}
METRICS = [
    "Timestamp",
    "slice_id",
    "slice_prb",
    "dl_buffer [bytes]",
    "tx_brate downlink [Mbps]",
    "sum_requested_prbs",
    "sum_granted_prbs",
]


def _read_cell(raw_dir: Path, scenario: str, tr: str, exp: str, bs: str) -> pd.DataFrame:
    folder = raw_dir / "slice_traffic" / scenario / tr / exp / bs / f"slices_{bs}"
    files = sorted(folder.glob("*_metrics.csv"))
    if not files:
        raise FileNotFoundError(f"No COMMAG metrics found under {folder}")
    frames = []
    for path in files:
        frame = pd.read_csv(path, usecols=METRICS, low_memory=False)
        for column in METRICS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["Timestamp", "slice_id", "slice_prb"])
        frame = frame[frame["sum_requested_prbs"] > 0]
        if frame.empty:
            continue
        frame["timestamp_s"] = (frame["Timestamp"] // 1000).astype("int64")
        frame["ratio_granted_req"] = np.clip(
            np.nan_to_num(frame["sum_granted_prbs"] / frame["sum_requested_prbs"]), 0.0, 1.0
        )
        frame["dl_buffer_scaled"] = frame["dl_buffer [bytes]"] / 100000.0
        frames.append(frame)
    if not frames:
        raise ValueError(f"No usable COMMAG metrics found under {folder}")
    return pd.concat(frames, ignore_index=True)


def _timestep(observation: np.ndarray) -> ts.TimeStep:
    return ts.TimeStep(
        tf.convert_to_tensor([0], dtype=tf.int32),
        tf.convert_to_tensor([0], dtype=tf.float32),
        tf.convert_to_tensor([1], dtype=tf.float32),
        tf.convert_to_tensor([observation], dtype=tf.float32),
    )


def _ten_rows(group: pd.DataFrame) -> tuple[np.ndarray, float]:
    metrics = group[["dl_buffer_scaled", "tx_brate downlink [Mbps]", "ratio_granted_req"]].to_numpy(
        dtype="float32"
    )
    prb = float(group["slice_prb"].iloc[0])
    if len(metrics) >= 10:
        metrics = metrics[:10]
    else:
        metrics = np.vstack([metrics, np.zeros((10 - len(metrics), 3), dtype="float32")])
    return metrics, prb


def export_actions(raw_dir: Path, prepared: Path, output: Path, model_root: Path) -> None:
    transitions = pd.read_csv(prepared, low_memory=False)
    transitions = transitions[transitions["publication_split"].isin(["test_seen", "test_unseen"])]
    cells = transitions[["scenario", "training_config", "experiment", "base_station"]].drop_duplicates()

    encoder = tf.keras.models.load_model(model_root / "encoder.h5")
    agents = {
        0: tf.saved_model.load(str(model_root / "embb_policy")),
        1: tf.saved_model.load(str(model_root / "mtc_policy")),
        2: tf.saved_model.load(str(model_root / "urllc_policy")),
    }

    output_rows = []
    for cell in cells.itertuples(index=False):
        frame = _read_cell(raw_dir, cell.scenario, cell.training_config, cell.experiment, cell.base_station)
        grouped = frame.groupby(["timestamp_s", "slice_id"], sort=True)
        previous = {0: 0, 1: 0, 2: 0}
        timestamps = sorted(frame["timestamp_s"].unique())
        for timestamp_s in timestamps:
            for slice_id in (0, 1, 2):
                key = (timestamp_s, slice_id)
                if key in grouped.groups:
                    group = grouped.get_group(key).sort_values(["Timestamp", "slice_prb"]).reset_index(drop=True)
                    metrics, prb = _ten_rows(group)
                    encoded = encoder.predict(np.expand_dims(metrics, axis=0), verbose=0).astype("float32")
                    observation = np.append(encoded, prb).astype("float32")
                    action = agents[slice_id].action(_timestep(observation))
                    scheduler = int(action[0][0][0].numpy())
                    previous[slice_id] = scheduler
                else:
                    scheduler = previous[slice_id]
                output_rows.append(
                    {
                        "scenario": cell.scenario,
                        "training_config": cell.training_config,
                        "experiment": cell.experiment,
                        "base_station": cell.base_station,
                        "timestamp_s": int(timestamp_s),
                        "slice_type": SLICE_NAMES[slice_id],
                        "scheduler_code": scheduler,
                    }
                )

    result = pd.DataFrame(output_rows).drop_duplicates(
        ["scenario", "training_config", "experiment", "base_station", "timestamp_s", "slice_type"],
        keep="last",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, compression="gzip")
    print(f"wrote {len(result)} original-PPO scheduler decisions to {output}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="/workspace/data/raw/commag")
    parser.add_argument(
        "--prepared",
        default="/workspace/data/prepared/commag-publication/commag_publication_transitions.csv.gz",
    )
    parser.add_argument(
        "--output",
        default="/workspace/data/prepared/commag-publication/original_ppo_actions.csv.gz",
    )
    parser.add_argument("--model-root", default="/opt/commag/ml_models")
    args = parser.parse_args()
    export_actions(Path(args.raw_dir), Path(args.prepared), Path(args.output), Path(args.model_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
