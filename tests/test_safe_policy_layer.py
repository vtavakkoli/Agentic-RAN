import numpy as np
from policies.safe_policy_layer import SafePolicyLayer

def test_fallback_when_forbidden():
    s=SafePolicyLayer()
    out=s.enforce(np.array([1.0]*9), allowed_mask=[0,0,0,0,0,0,0,0,1])
    assert out['used_fallback']
