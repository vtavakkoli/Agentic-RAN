from __future__ import annotations

from dataclasses import dataclass

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


class RANControlEnv:
    def __init__(self, dataset: pd.DataFrame, window_size: int = 8):
        self.dataset = dataset.sort_values(["Timestamp", "source_file"], kind="stable").reset_index(drop=True)
        self.window_size = window_size
        self.encoder = SliceSequenceEncoder(num_features=len(SLICE_OBSERVATION_COLS), latent_dim=32)
        self.cursor = window_size
        self.previous_action = 0

        obs_dim = 3 * (32 + 7)
        self.observation_spec = Spec(shape=(obs_dim,), dtype="float32", minimum=-1e6, maximum=1e6)
        self.action_spec = Spec(shape=(), dtype="int32", minimum=0, maximum=10)

    def _slice_state(self, sid: int, idx: int) -> dict:
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
        hist = self.dataset.iloc[max(0, self.cursor - self.window_size) : self.cursor].copy()
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
        return self._observation()

    def step(self, action: int):
        if action < self.action_spec.minimum or action > self.action_spec.maximum:
            raise ValueError(f"Action {action} out of bounds")
        if self.cursor >= len(self.dataset) - 1:
            return self._observation(), 0.0, True, {"control_message": self._message(action)}

        cur_row = self.dataset.iloc[self.cursor - 1].to_dict()
        next_row = self.dataset.iloc[self.cursor].to_dict()
        sid = int(cur_row.get("slice_id", 0))

        if sid == 0:
            reward = reward_embb(cur_row, next_row, self.previous_action, action)
        elif sid == 1:
            reward = reward_mtc(cur_row, next_row, self.previous_action, action)
        else:
            reward = reward_urllc(cur_row, next_row, self.previous_action, action)

        self.previous_action = action
        self.cursor += 1
        done = self.cursor >= len(self.dataset)
        info = {
            "control_message": self._message(action),
            "action_name": ACTION_MAP[action],
            "slice_id": sid,
        }
        return self._observation(), float(reward), done, info
