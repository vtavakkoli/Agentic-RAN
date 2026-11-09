"""PULSE-RAN COMMAG dataset-backed *shadow* evaluation.

This reads prepared, real Colosseum COMMAG telemetry. Unlike the synthetic
twin, it does NOT simulate interventions or assign counterfactual rewards.

PULSE-RAN was defined for dimensionless simulated queue values (0.45 URLLC,
1.0 eMBB). Actual COMMAG downlink buffers are in BYTES. Hence this adapter
calibrates a fixed bytes-per-unit normalization on TRAIN ONLY and explicitly
labels the result a COMMAG adaptation, not evidence that the original
threshold values transferred directly to the live experiment.

It aligns the three slices by scenario, configuration, experiment, BS and
timestamp. Unaligned observations are omitted, reported, never fabricated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agentic_ran.commag import COMMAG_REPOSITORY, COMMAG_REVISION
from agentic_ran.research_twin_environment import ACTIONS

GROUP = (
    "publication_split", "scenario", "training_config", "experiment",
    "base_station", "timestamp_s",
)
SLICES = ("eMBB", "mMTC", "URLLC")
REQUIRED = (
    *GROUP, "slice_type", "dl_buffer_bytes", "slice_prb", "reward",
)
SPLITS = ("train", "validation", "test_seen", "test_unseen")
MODES = ("balanced", "urllc_priority", "embb_priority")


def _counts(series: pd.Series) -> dict[str, int]:
    values = series.value_counts()
    return {key: int(values.get(key, 0)) for key in MODES}


def build_shadow(
    frame: pd.DataFrame, *, train_quantile: float = 0.90,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Produce context-only recommendations from *recorded* COMMAG buffers.

    No future next_* fields or held-out outcomes are used to select actions.
    No attempt is made to estimate how changing an action alters reward.
    """
    if not 0.5 <= train_quantile <= 0.99:
        raise ValueError("train quantile must be in [0.5,0.99]")
    missing = sorted(set(REQUIRED) - set(frame))
    if missing:
        raise ValueError(f"COMMAG publication data missing columns: {missing}")
    data = frame.loc[:, list(REQUIRED)].copy()
    if data.empty:
        raise ValueError("COMMAG input has no data")
    if not set(data["publication_split"]).issubset(SPLITS):
        raise ValueError("unknown COMMAG publication split")
    if not set(data["slice_type"]).issubset(SLICES):
        raise ValueError("unrecognized COMMAG slice type")
    for name in ("dl_buffer_bytes", "slice_prb", "reward", "timestamp_s"):
        data[name] = pd.to_numeric(data[name], errors="coerce")
        if not np.isfinite(data[name].to_numpy(dtype=float)).all():
            raise ValueError(f"invalid or nonfinite COMMAG measurement: {name}")
    if (data["dl_buffer_bytes"] < 0).any() or (data["slice_prb"] < 0).any():
        raise ValueError("negative buffer or PRB telemetry")
    # Real traces are per-UE/per-slice. Aggregate only within the exact same
    # observed base-station, configuration, experiment and second.
    observed_keys = data.groupby(list(GROUP), observed=True).size().rename("rows")
    grouped = data.groupby([*GROUP, "slice_type"], observed=True, as_index=False).agg(
        queue_bytes=("dl_buffer_bytes", "mean"),
        recorded_rbg=("slice_prb", "mean"),
        recorded_reward=("reward", "mean"),
        ue_trace_rows=("reward", "size"),
    )
    queue = grouped.pivot(index=list(GROUP), columns="slice_type", values="queue_bytes")
    rbg = grouped.pivot(index=list(GROUP), columns="slice_type", values="recorded_rbg")
    reward = grouped.pivot(index=list(GROUP), columns="slice_type", values="recorded_reward")
    # No filling of missing slice measurements: that would invent context.
    joint = queue.reindex(columns=SLICES).dropna(subset=list(SLICES))
    if joint.empty:
        raise ValueError(
            "No aligned eMBB/mMTC/URLLC observations at the same BS/time; "
            "cannot evaluate a joint PULSE-RAN decision from these traces."
        )
    train = joint.index.get_level_values("publication_split") == "train"
    if not np.any(train):
        raise ValueError("No aligned COMMAG training observations for normalization")
    # Calibrate only on real train buffers; never on validation/test states.
    training = joint.loc[train, ["eMBB", "URLLC"]]
    scales = training.quantile(train_quantile)
    if not np.isfinite(scales.to_numpy()).all() or (scales <= 0).any():
        raise ValueError(
            "Cannot derive positive eMBB/URLLC byte scales from training data"
        )
    normalized_embb = joint["eMBB"].to_numpy() / float(scales["eMBB"])
    normalized_urllc = joint["URLLC"].to_numpy() / float(scales["URLLC"])
    mode = np.select(
        [normalized_urllc > 0.45, normalized_embb > 1.0],
        [1, 2], default=0,
    ).astype(int)
    # ACTIONS 0..2 preserve the PULSE simulated *target share* semantics.
    # They are not executable COMMAG RR/WF/PF scheduler/prb action labels.
    targets = ACTIONS[mode]
    output = joint.reset_index()[list(GROUP)].copy()
    output["embb_buffer_bytes"] = joint["eMBB"].to_numpy()
    output["mtc_buffer_bytes"] = joint["mMTC"].to_numpy()
    output["urllc_buffer_bytes"] = joint["URLLC"].to_numpy()
    output["embb_queue_normalized"] = normalized_embb
    output["urllc_queue_normalized"] = normalized_urllc
    output["pulse_mode"] = np.asarray(MODES)[mode]
    for i, slice_name in enumerate(SLICES):
        output[f"target_{slice_name}_share"] = targets[:, i]
    observed_prb = rbg.reindex(joint.index).reindex(columns=SLICES).to_numpy()
    denominator = observed_prb.sum(axis=1)
    valid_mix = denominator > 0
    normalized_logged = np.zeros_like(observed_prb)
    normalized_logged[valid_mix] = (
        observed_prb[valid_mix] / denominator[valid_mix, None]
    )
    difference = np.full(len(output), np.nan)
    difference[valid_mix] = np.abs(
        targets[valid_mix] - normalized_logged[valid_mix]
    ).sum(axis=1)
    output["recorded_allocation_l1_distance"] = difference
    for i, slice_name in enumerate(SLICES):
        output[f"recorded_{slice_name}_share"] = np.where(
            valid_mix, normalized_logged[:, i], np.nan
        )
        output[f"recorded_{slice_name}_reward"] = (
            reward.reindex(joint.index).reindex(columns=SLICES).to_numpy()[:, i]
        )
    # Recorded reward belongs to the logged configuration, never to PULSE.
    output["reward_source"] = "recorded_commag_actions_only_not_pulse"
    counts = observed_keys.reset_index()
    counts["complete"] = counts.set_index(list(GROUP)).index.isin(joint.index)
    coverage = {}
    modes = {}
    for split in SPLITS:
        source = counts[counts["publication_split"] == split]
        selected = output[output["publication_split"] == split]
        coverage[split] = {
            "all_base_station_seconds": int(len(source)),
            "aligned_three_slice_seconds": int(len(selected)),
            "alignment_rate": (
                float(len(selected) / len(source)) if len(source) else None
            ),
        }
        modes[split] = _counts(selected["pulse_mode"])
    if not coverage["test_seen"]["aligned_three_slice_seconds"] or (
        not coverage["test_unseen"]["aligned_three_slice_seconds"]
    ):
        raise ValueError("Both seen and unseen COMMAG test splits require aligned slice observations")
    output = output.sort_values(list(GROUP), kind="mergesort").reset_index(drop=True)
    summary = {
        "study": "PULSE-RAN-COMMAG-shadow",
        "source_kind": "real_recorded_colosseum_commag_traces",
        "source_repository": COMMAG_REPOSITORY,
        "source_revision": COMMAG_REVISION,
        "dataset_has_actuated_pulse_ran": False,
        "ns3_was_executed": False,
        "new_colosseum_experiment_was_executed": False,
        "reported_counterfactual_utility": False,
        "train_only_calibration": {
            "method": "per_slice_queue_bytes_quantile",
            "quantile": train_quantile,
            "embb_bytes_per_normalized_queue_unit": float(scales["eMBB"]),
            "urllc_bytes_per_normalized_queue_unit": float(scales["URLLC"]),
            "original_toy_thresholds": {"urllc": 0.45, "embb": 1.0},
            "note": (
                "The original toy thresholds are reused only after train-only "
                "byte normalization. The absolute decisions are not directly "
                "equivalent to the simulated policy without an external calibration."
            ),
        },
        "aligned_observation_coverage": coverage,
        "recommendation_counts": modes,
        "target_share_source": "PULSE-RAN target shares for modes 0/1/2; not real COMMAG scheduler commands",
        "recorded_share_distance_type": "descriptive_L1_not_action_value_or_observed_gain",
        "limitations": [
            "COMMAG was collected under historical RR/WF/PF and RBG configurations.",
            "Recommended PULSE target PRB fractions are not directly executable recorded COMMAG action labels.",
            "Missing joint slice timestamps are excluded rather than filled.",
            "Reward columns represent observed logged actions and CANNOT establish PULSE performance.",
            "COMMAG is a Colosseum-emulator dataset; it does not launch ns-3 or a fresh Colosseum control loop.",
        ],
    }
    return output, summary


def run(
    data_path: str | Path,
    output_dir: str | Path,
    *,
    train_quantile: float = 0.90,
) -> dict[str, Any]:
    source = Path(data_path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Prepared COMMAG dataset not found: {source}. "
            "Run the prepare-full-commag service first."
        )
    # Avoid loading the many unused next_* future variables, and keep
    # dataset-backed shadow selection independent of logged outcomes.
    frame = pd.read_csv(source, usecols=list(REQUIRED), low_memory=False)
    recommendations, summary = build_shadow(
        frame, train_quantile=train_quantile,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    recommendations.to_csv(
        destination / "pulse_commag_shadow_recommendations.csv.gz",
        index=False, compression="gzip",
    )
    (destination / "pulse_commag_shadow_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PULSE-RAN shadow recommendations on real COMMAG recordings (no control actuation)"
    )
    parser.add_argument(
        "--data", type=Path,
        default=Path("data/prepared/commag-publication/commag_publication_transitions.csv.gz"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/pulse-commag"),
    )
    parser.add_argument("--train-quantile", type=float, default=0.90)
    args = parser.parse_args()
    summary = run(args.data, args.output, train_quantile=args.train_quantile)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
