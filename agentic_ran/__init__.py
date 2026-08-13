"""Agentic-RAN: safety-governed agentic intelligence for open radio access networks."""
from agentic_ran.domain import ExecutionMode, NetworkObservation, PolicyDecision, SLAIntent
__all__=["ExecutionMode","NetworkObservation","PolicyDecision","PolicyService","SLAIntent"]
__version__="2.0.0"
def __getattr__(name:str):
    if name=="PolicyService":
        from agentic_ran.service import PolicyService
        return PolicyService
    raise AttributeError(name)
