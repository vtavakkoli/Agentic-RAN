"""Near-RT xApp and Non-RT rApp orchestration primitives."""
from __future__ import annotations
import asyncio,json,urllib.request
from collections.abc import Awaitable,Callable
from dataclasses import dataclass
from typing import Any,Protocol
from agentic_ran.control_loop import AgenticControlLoop,ControlStepResult
from agentic_ran.domain import SLAIntent
from agentic_ran.mlops import DriftMonitor,PromotionGate
class A1PolicyTransport(Protocol):
    def publish_policy(self,policy_type_id:str,policy_instance_id:str,payload:dict[str,Any])->dict[str,Any]: ...
class HttpA1PolicyTransport:
    def __init__(self,base_url:str,timeout_seconds:float=3.0): self.base_url=base_url.rstrip("/"); self.timeout_seconds=timeout_seconds
    def publish_policy(self,policy_type_id:str,policy_instance_id:str,payload:dict[str,Any]):
        endpoint=f"{self.base_url}/A1-P/v2/policytypes/{policy_type_id}/policies/{policy_instance_id}"; request=urllib.request.Request(endpoint,data=json.dumps(payload).encode("utf-8"),headers={"content-type":"application/json"},method="PUT")
        with urllib.request.urlopen(request,timeout=self.timeout_seconds) as response:
            body=response.read().decode("utf-8"); return json.loads(body) if body else {"accepted":True}
class NearRTRICXApp:
    def __init__(self,loop:AgenticControlLoop): self.loop=loop; self.running=False
    async def run(self,intent:SLAIntent|str|None=None,max_steps:int|None=None,on_step:Callable[[ControlStepResult],Awaitable[None]]|None=None):
        results=[]; self.running=True; step=0
        while self.running and (max_steps is None or step<max_steps):
            result=await self.loop.step(intent=intent); results.append(result)
            if on_step: await on_step(result)
            step+=1
        return results
    def stop(self): self.running=False
@dataclass(frozen=True,slots=True)
class RAppEvaluation:
    drift_score:float; promotion_allowed:bool; promotion_reasons:tuple[str,...]
class NonRTRICRApp:
    def __init__(self,drift_monitor:DriftMonitor|None=None,promotion_gate:PromotionGate|None=None,a1_transport:A1PolicyTransport|None=None): self.drift_monitor=drift_monitor or DriftMonitor(); self.promotion_gate=promotion_gate or PromotionGate(); self.a1_transport=a1_transport
    def evaluate_model(self,recent_observations:list[Any],metrics:dict[str,float]):
        drift=self.drift_monitor.score(recent_observations); allowed,reasons=self.promotion_gate.evaluate({**metrics,"drift_score":drift}); return RAppEvaluation(drift,allowed,tuple(reasons))
    async def publish_intent(self,intent:SLAIntent,instance_id:str="agentic-ran-default"):
        if self.a1_transport is None: return {"published":False,"reason":"no A1 transport configured","intent":intent.model_dump(mode="json")}
        payload={"scope":{"slice_type":intent.slice_type},"intent":intent.model_dump(mode="json")}; return await asyncio.to_thread(self.a1_transport.publish_policy,"agentic-ran-intent",instance_id,payload)
