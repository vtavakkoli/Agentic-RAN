from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

ACTION_SPACE = {
    0: {"action_name": "keep_allocation", "target_slice": -1, "recommended_policy": None, "recommended_prb_delta": 0},
    1: {"action_name": "increase_embb_prb", "target_slice": 0, "recommended_policy": None, "recommended_prb_delta": 1},
    2: {"action_name": "increase_mtc_prb", "target_slice": 1, "recommended_policy": None, "recommended_prb_delta": 1},
    3: {"action_name": "increase_urllc_prb", "target_slice": 2, "recommended_policy": None, "recommended_prb_delta": 1},
    4: {"action_name": "decrease_embb_prb", "target_slice": 0, "recommended_policy": None, "recommended_prb_delta": -1},
    5: {"action_name": "decrease_mtc_prb", "target_slice": 1, "recommended_policy": None, "recommended_prb_delta": -1},
    6: {"action_name": "decrease_urllc_prb", "target_slice": 2, "recommended_policy": None, "recommended_prb_delta": -1},
    7: {"action_name": "switch_to_rr", "target_slice": -1, "recommended_policy": "RR", "recommended_prb_delta": 0},
    8: {"action_name": "switch_to_wf", "target_slice": -1, "recommended_policy": "WF", "recommended_prb_delta": 0},
    9: {"action_name": "switch_to_pf", "target_slice": -1, "recommended_policy": "PF", "recommended_prb_delta": 0},
}


def _mk(action_id: int, confidence: float, reason: str) -> dict[str, Any]:
    base = ACTION_SPACE[action_id]
    return {
        "action_id": action_id,
        "action_name": base["action_name"],
        "target_slice": base["target_slice"],
        "recommended_policy": base["recommended_policy"],
        "recommended_prb_delta": base["recommended_prb_delta"],
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "reason": reason,
    }


def recommend_action(state: pd.Series | dict[str, Any]) -> dict[str, Any]:
    row = dict(state)
    sid = int(row.get("slice_id", -1))
    ratio = float(row.get("ratio_granted_req", 1.0))
    pressure = float(row.get("prb_pressure", 1.0))
    dl_buffer = float(row.get("dl_buffer [bytes]", 0.0))
    throughput = float(row.get("tx_brate downlink [Mbps]", 0.0))
    policy_name = str(row.get("scheduling_policy_name", "unknown"))
    spectral_quality = float(row.get("spectral_quality_proxy", 0.0))

    if sid == 2 and (ratio < 0.7 or pressure > 1.4):
        return _mk(3, 0.9, "URLLC grant ratio is low or pressure is high; prioritize URLLC resources.")
    if sid == 0 and dl_buffer > 200_000 and throughput < 0.8:
        if policy_name == "RR":
            return _mk(9, 0.82, "eMBB buffer high with low throughput under RR; PF is better for throughput fairness.")
        return _mk(1, 0.78, "eMBB buffer is high with low throughput; increase eMBB PRB allocation.")
    if sid == 1 and (ratio < 0.75 or pressure > 1.3):
        return _mk(2, 0.8, "MTC request pressure is high with weak grant ratio; increase MTC PRB.")

    if ratio > 0.98 and pressure < 0.9:
        if sid == 0:
            return _mk(4, 0.6, "eMBB appears over-allocated under low pressure; consider reducing PRB.")
        if sid == 1:
            return _mk(5, 0.6, "MTC appears over-allocated under low pressure; consider reducing PRB.")
        if sid == 2:
            return _mk(6, 0.6, "URLLC appears over-allocated under low pressure; consider reducing PRB.")

    if policy_name == "WF" and spectral_quality < 20:
        return _mk(9, 0.65, "WF with poor quality can underperform; switch to PF.")
    if policy_name == "RR" and pressure > 1.2:
        return _mk(9, 0.72, "Persistent load under RR; switch to PF.")

    return _mk(0, 0.55, "No critical pressure signals; keep current allocation.")


def build_slice_state(df_window: pd.DataFrame) -> pd.DataFrame:
    work = df_window.copy()
    if work.empty:
        return work
    grp = work.groupby("slice_id", as_index=False)
    out = grp.agg(
        mean_tx_brate_downlink_mbps=("tx_brate downlink [Mbps]", "mean"),
        mean_ratio_granted_req=("ratio_granted_req", "mean"),
        mean_prb_pressure=("prb_pressure", "mean"),
        mean_dl_buffer=("dl_buffer [bytes]", "mean"),
        mean_ul_buffer=("ul_buffer [bytes]", "mean"),
        mean_dl_cqi=("dl_cqi", "mean"),
        mean_ul_sinr=("ul_sinr", "mean"),
        num_ues=("ue_id", lambda s: int(s.nunique())),
        expected_rbg_for_slice=("expected_rbg_for_slice", "mean"),
        current_slice_prb=("slice_prb", "mean"),
        scheduling_policy=("scheduling_policy_name", lambda s: Counter(s).most_common(1)[0][0]),
    )
    if "traffic_class" in work.columns:
        dom = grp["traffic_class"].agg(lambda s: Counter(s).most_common(1)[0][0]).rename("dominant_traffic_class")
        out = out.merge(dom, on="slice_id", how="left")
    else:
        out["dominant_traffic_class"] = "unknown"
    return out
