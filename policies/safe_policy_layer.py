from __future__ import annotations
import numpy as np


DEFAULT_ACTIONS = ["do_nothing","decrease_embb","increase_embb","decrease_mtc","increase_mtc","decrease_urllc","increase_urllc","rebalance_slices","safe_fallback"]


class SafePolicyLayer:
    def __init__(self, actions=None, sla_threshold: float = 0.8):
        self.actions = actions or DEFAULT_ACTIONS
        self.sla_threshold = sla_threshold

    def enforce(self, logits, allowed_mask=None, resource_ok: bool = True, sla_ok: bool = True):
        logits = np.asarray(logits, dtype=float)
        if allowed_mask is not None:
            logits = np.where(np.asarray(allowed_mask, dtype=bool), logits, -1e9)
        proposed = int(np.argmax(logits))
        safe_idx = self.actions.index("safe_fallback")
        safe = resource_ok and sla_ok and logits[proposed] > -1e8
        selected = proposed if safe else safe_idx
        conf = float(np.max(np.exp(logits - np.max(logits)) / np.sum(np.exp(logits - np.max(logits)))))
        return {"selected_action": selected, "decision_confidence": conf, "used_fallback": selected == safe_idx}
