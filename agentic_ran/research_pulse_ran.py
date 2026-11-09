"""PULSE-RAN: Priority-driven URLLC and Load-Sensitive Engine.

A deliberately simple, interpretable queue-priority controller for the
three-slice *simulated* RAN environment. This does not claim certified
reliability, O-RAN deployment suitability, or compliance with the
experimental learned-twin action-support mask.
"""
from __future__ import annotations

import numpy as np

NAME = "PULSE-RAN"
EXPANSION = "Priority-driven URLLC and Load-Sensitive Engine"
URLLC_QUEUE_THRESHOLD = 0.45
EMBB_QUEUE_THRESHOLD = 1.0


def select_pulse_ran(state: np.ndarray) -> int:
    """Return 1 (URLLC), 2 (eMBB), or 0 (balanced) by strict queue priority.

    The thresholds are those of the *original* research heuristic, unchanged.
    State is [3 queues, 3 demands, 3 channels] with URLLC queue at index 2.
    No new training, threshold tuning, or held-out test adaptation.
    """
    s = np.asarray(state, dtype=float)
    if s.shape != (9,) or not np.isfinite(s).all():
        raise ValueError("PULSE-RAN requires a finite 9-feature twin state")
    if s[2] > URLLC_QUEUE_THRESHOLD:
        return 1
    if s[0] > EMBB_QUEUE_THRESHOLD:
        return 2
    return 0
