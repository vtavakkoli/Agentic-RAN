from __future__ import annotations


def _common_penalties(state: dict, prev_action: int | None, action: int) -> dict[str, float]:
    tx_err = float(state.get("tx_errors downlink (%)", 0.0))
    rx_err = float(state.get("rx_errors uplink (%)", 0.0))
    packet_error_penalty = 0.02 * (tx_err + rx_err)
    prb_overuse_penalty = max(float(state.get("slice_prb", 0.0)) - 12.0, 0.0) * 0.05
    action_switching_penalty = 0.05 if prev_action is not None and prev_action != action else 0.0
    return {
        "packet_error_penalty": packet_error_penalty,
        "prb_overuse_penalty": prb_overuse_penalty,
        "action_switching_penalty": action_switching_penalty,
    }


def reward_embb(state: dict, next_state: dict, prev_action: int | None, action: int) -> float:
    p = _common_penalties(next_state, prev_action, action)
    throughput_gain = float(next_state.get("tx_brate downlink [Mbps]", 0.0)) - float(state.get("tx_brate downlink [Mbps]", 0.0))
    return throughput_gain - p["packet_error_penalty"] - p["prb_overuse_penalty"] - p["action_switching_penalty"]


def reward_mtc(state: dict, next_state: dict, prev_action: int | None, action: int) -> float:
    p = _common_penalties(next_state, prev_action, action)
    ratio = float(next_state.get("ratio_granted_req", 0.0))
    stability_bonus = 0.15 if abs(float(next_state.get("sum_granted_prbs", 0.0)) - float(state.get("sum_granted_prbs", 0.0))) <= 1 else 0.0
    return ratio + stability_bonus - p["prb_overuse_penalty"] - p["packet_error_penalty"] - p["action_switching_penalty"]


def reward_urllc(state: dict, next_state: dict, prev_action: int | None, action: int) -> float:
    p = _common_penalties(next_state, prev_action, action)
    reliability_bonus = 1.0 - min(1.0, 0.01 * (float(next_state.get("tx_errors downlink (%)", 0.0)) + float(next_state.get("rx_errors uplink (%)", 0.0))))
    ratio = float(next_state.get("ratio_granted_req", 0.0))
    buffer_penalty = 0.01 * (float(next_state.get("dl_buffer [bytes]", 0.0)) + float(next_state.get("ul_buffer [bytes]", 0.0)))
    latency_proxy_penalty = 0.05 * max(float(next_state.get("dl_buffer [bytes]", 0.0)) - 1.0, 0.0)
    unsafe_action_penalty = 0.2 if action in {7, 8} and ratio < 0.7 else 0.0
    return reliability_bonus + ratio - buffer_penalty - p["packet_error_penalty"] - latency_proxy_penalty - unsafe_action_penalty - p["action_switching_penalty"]
