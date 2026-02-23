from __future__ import annotations

from typing import Dict

import numpy as np


def react_loop(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.72) -> Dict[str, float]:
    reward = 0.0
    penalties = 0.0
    tx_power = 1.0
    rb_alloc = 1.0

    for yt, yp in zip(y_true, y_pred):
        error = abs(yt - yp)
        congestion_risk = yp
        confidence = max(0.0, 1.0 - error)

        if congestion_risk > threshold:
            tx_power = min(1.25, tx_power + 0.02)
            rb_alloc = min(1.40, rb_alloc + 0.03)
        else:
            tx_power = max(0.85, tx_power - 0.01)
            rb_alloc = max(0.85, rb_alloc - 0.01)

        control_cost = abs(tx_power - 1.0) * 0.05 + abs(rb_alloc - 1.0) * 0.05
        reward += (1.0 - error) * confidence - control_cost
        penalties += control_cost

    n = max(1, len(y_true))
    return {
        "react_avg_reward": float(reward / n),
        "react_avg_penalty": float(penalties / n),
        "react_final_tx_power": float(tx_power),
        "react_final_rb_alloc": float(rb_alloc),
    }
