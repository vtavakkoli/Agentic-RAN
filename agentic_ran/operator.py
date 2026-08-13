"""Operator-facing explanations. LLMs remain outside the control authority path."""
from __future__ import annotations
from collections.abc import Callable
from agentic_ran.domain import PolicyDecision
class TraceExplainer:
    def explain(self,decision:PolicyDecision)->str:
        rejected=", ".join(decision.trace.rejected_policies) or "none"; uncertainty=decision.uncertainty.level if decision.uncertainty else "not assessed"
        return f"Policy {decision.selected_policy} was selected with {decision.confidence:.0%} confidence. Rejected candidates: {rejected}. Uncertainty: {uncertainty}; OOD score: {decision.ood_score:.2f}. {decision.trace.critique}"
class ExternalLLMExplainer:
    def __init__(self,generate:Callable[[str],str]): self.generate=generate
    def explain(self,question:str,decision:PolicyDecision)->str:
        prompt="You are explaining an already completed RAN decision. You have no control authority. Do not invent metrics. Answer only from this structured trace.\n\n"+f"Question: {question}\nTrace: {decision.model_dump_json()}"
        return self.generate(prompt)
