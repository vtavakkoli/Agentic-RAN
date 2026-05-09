from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from agentic_ran.drl_data import SLICE_OBSERVATION_COLS, split_data
from agentic_ran.drl_observation import SliceSequenceEncoder, build_drl_observation
from agentic_ran.drl_rewards import reward_embb, reward_mtc, reward_urllc

ACTION_MAP = {
    0: "keep_current",
    1: "switch_to_RR",
    2: "switch_to_WF",
    3: "switch_to_PF",
    4: "increase_eMBB_PRB",
    5: "increase_MTC_PRB",
    6: "increase_URLLC_PRB",
    7: "decrease_eMBB_PRB",
    8: "decrease_MTC_PRB",
    9: "decrease_URLLC_PRB",
    10: "safe_fallback_policy",
}


@dataclass
class Spec:
    shape: tuple[int, ...]
    dtype: str
    minimum: int | float
    maximum: int | float


class TransitionModel(Protocol):
    def predict_next_state(self, state: dict, action: int) -> dict: ...


class DatasetDeltaTransitionModel:
    """Simple world model: predicts next state from state + learned action deltas.

    The model estimates per-action average deltas from the dataset and applies those
    deltas to the current state to generate a simulated next state.
    """

    _tracked_cols = [
        "slice_prb",
        "scheduling_policy",
        "ratio_granted_req",
        "sum_granted_prbs",
        "tx_brate downlink [Mbps]",
        "tx_errors downlink (%)",
        "rx_errors uplink (%)",
        "dl_buffer [bytes]",
        "ul_buffer [bytes]",
    ]

    def __init__(self, dataset: pd.DataFrame):
        self.action_deltas = self._fit(dataset)

    def _fit(self, dataset: pd.DataFrame) -> dict[int, dict[str, float]]:
        base = {a: {c: 0.0 for c in self._tracked_cols} for a in ACTION_MAP}
        work = dataset.sort_values(["source_file", "slice_id", "Timestamp"], kind="stable")
        for sid, part in work.groupby("slice_id", sort=False):
            part = part.reset_index(drop=True)
            for i in range(len(part) - 1):
                cur = part.iloc[i]
                nxt = part.iloc[i + 1]
                synthetic_action = self._synthetic_action_from_delta(float(nxt.get("slice_prb", 0.0)) - float(cur.get("slice_prb", 0.0)), sid)
                for col in self._tracked_cols:
                    base[synthetic_action][col] += float(nxt.get(col, 0.0)) - float(cur.get(col, 0.0))
        for action in base:
            # lightweight shrinkage to avoid large unstable jumps
            for col in base[action]:
                base[action][col] *= 0.05
        return base

    def _synthetic_action_from_delta(self, prb_delta: float, sid: int) -> int:
        if prb_delta > 0.0:
            return 4 + int(sid)
        if prb_delta < 0.0:
            return 7 + int(sid)
        return 0

    def predict_next_state(self, state: dict, action: int) -> dict:
        nxt = dict(state)
        deltas = self.action_deltas.get(action, {})
        for col in self._tracked_cols:
            nxt[col] = float(state.get(col, 0.0)) + float(deltas.get(col, 0.0))

        # explicit control effects by action semantics
        sid = int(state.get("slice_id", 0))
        if action in {1, 2, 3}:
            nxt["scheduling_policy"] = float(action - 1)
        if action in {4, 5, 6} and action - 4 == sid:
            nxt["slice_prb"] = float(state.get("slice_prb", 0.0)) + 1.0
        if action in {7, 8, 9} and action - 7 == sid:
            nxt["slice_prb"] = max(0.0, float(state.get("slice_prb", 0.0)) - 1.0)

        nxt["ratio_granted_req"] = float(np.clip(nxt.get("ratio_granted_req", 0.0), 0.0, 1.0))
        nxt["tx_errors downlink (%)"] = max(0.0, float(nxt.get("tx_errors downlink (%)", 0.0)))
        nxt["rx_errors uplink (%)"] = max(0.0, float(nxt.get("rx_errors uplink (%)", 0.0)))
        nxt["dl_buffer [bytes]"] = max(0.0, float(nxt.get("dl_buffer [bytes]", 0.0)))
        nxt["ul_buffer [bytes]"] = max(0.0, float(nxt.get("ul_buffer [bytes]", 0.0)))
        return nxt


class RANControlEnv:
    def __init__(self, dataset: pd.DataFrame, window_size: int = 8, transition_model: TransitionModel | None = None):
        self.dataset = dataset.sort_values(["Timestamp", "source_file"], kind="stable").reset_index(drop=True)
        self.window_size = window_size
        self.encoder = SliceSequenceEncoder(num_features=len(SLICE_OBSERVATION_COLS), latent_dim=32)
        self.cursor = window_size
        self.previous_action = 0
        self.transition_model = transition_model or DatasetDeltaTransitionModel(self.dataset)
        self._simulated_states: list[dict] = []

        obs_dim = 3 * (32 + 7)
        self.observation_spec = Spec(shape=(obs_dim,), dtype="float32", minimum=-1e6, maximum=1e6)
        self.action_spec = Spec(shape=(), dtype="int32", minimum=0, maximum=10)

    def _state_at(self, idx: int) -> dict:
        if idx < len(self._simulated_states):
            return self._simulated_states[idx]
        if len(self.dataset) == 0:
            return {}
        base_idx = min(max(idx, 0), len(self.dataset) - 1)
        return self.dataset.iloc[base_idx].to_dict()

    def _slice_state(self, sid: int, idx: int) -> dict:
        candidates = [s for s in self._simulated_states[max(0, idx - self.window_size + 1) : idx + 1] if int(s.get("slice_id", -1)) == sid]
        if candidates:
            return candidates[-1]
        part = self.dataset.loc[self.dataset["slice_id"] == sid]
        if part.empty:
            return {k: 0.0 for k in ["slice_prb", "scheduling_policy", "ratio_granted_req", "tx_brate downlink [Mbps]", "tx_errors downlink (%)", "dl_buffer [bytes]"]}
        pos = min(max(0, idx), len(part) - 1)
        return part.iloc[pos].to_dict()

    def _message(self, action: int) -> str:
        policy_val = 2
        if action == 1:
            policy_val = 0
        elif action == 2:
            policy_val = 1
        elif action == 3:
            policy_val = 2
        elif action == 10:
            policy_val = 0
        policies = [policy_val, policy_val, policy_val]
        return ",".join(str(x) for x in policies)

    def _observation(self) -> np.ndarray:
        sim_df = pd.DataFrame(self._simulated_states[: self.cursor])
        hist = sim_df.iloc[max(0, self.cursor - self.window_size) : self.cursor].copy()
        windows = split_data(hist, window_size=self.window_size)
        chunks = []
        for sid in [0, 1, 2]:
            st = self._slice_state(sid, self.cursor - 1)
            obs = build_drl_observation(
                encoder=self.encoder,
                slice_window=windows[sid],
                current_slice_prb=float(st.get("slice_prb", 0.0)),
                current_scheduling_policy=float(st.get("scheduling_policy", 0.0)),
                previous_action=float(self.previous_action),
                ratio_granted_req=float(st.get("ratio_granted_req", 0.0)),
                predicted_next_throughput=float(st.get("tx_brate downlink [Mbps]", 0.0)),
                predicted_peak_risk=float(st.get("tx_errors downlink (%)", 0.0)) / 100.0,
                predicted_buffer_pressure=float(st.get("dl_buffer [bytes]", 0.0)),
            )
            chunks.append(obs)
        return np.concatenate(chunks, axis=0).astype(np.float32)

    def reset(self):
        self.cursor = self.window_size
        self.previous_action = 0
        self._simulated_states = [self.dataset.iloc[i].to_dict() for i in range(min(len(self.dataset), self.window_size))]
        return self._observation()

    def step(self, action: int):
        if action < self.action_spec.minimum or action > self.action_spec.maximum:
            raise ValueError(f"Action {action} out of bounds")
        if self.cursor == 0 or not self._simulated_states:
            _ = self.reset()

        cur_row = self._simulated_states[self.cursor - 1]
        next_row = self.transition_model.predict_next_state(cur_row, action)
        next_row["Timestamp"] = cur_row.get("Timestamp")
        next_row["source_file"] = cur_row.get("source_file", "simulated")
        next_row["slice_id"] = cur_row.get("slice_id", 0)
        self._simulated_states.append(next_row)

        sid = int(cur_row.get("slice_id", 0))
        if sid == 0:
            reward = reward_embb(cur_row, next_row, self.previous_action, action)
        elif sid == 1:
            reward = reward_mtc(cur_row, next_row, self.previous_action, action)
        else:
            reward = reward_urllc(cur_row, next_row, self.previous_action, action)

        self.previous_action = action
        self.cursor += 1
        done = self.cursor >= max(len(self.dataset), self.window_size * 4)
        info = {"control_message": self._message(action), "action_name": ACTION_MAP[action], "slice_id": sid, "simulated": True}
        return self._observation(), float(reward), done, info
