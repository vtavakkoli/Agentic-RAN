"""Offline fitted-Q and safety-wrapped RL baselines for discrete RAN policies."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from agentic_ran.config import NUMERIC_FEATURES
from agentic_ran.domain import NetworkObservation

@dataclass(frozen=True, slots=True)
class OfflineTransition:
    state: NetworkObservation
    action: str
    reward: float
    next_state: NetworkObservation
    done: bool = False

class FittedQAgent:
    def __init__(self, actions:list[str], gamma:float=0.97, random_seed:int=42):
        if not actions: raise ValueError("at least one action is required")
        self.actions=list(actions); self.gamma=gamma; self.random_seed=random_seed; self.model:ExtraTreesRegressor|None=None; self._action_index={name:index for index,name in enumerate(self.actions)}
    def _vector(self, observation:NetworkObservation, action:str):
        record=observation.feature_record(); slice_values=[1.0 if str(observation.slice_type)==name else 0.0 for name in ("eMBB","URLLC","mMTC")]; action_values=[1.0 if index==self._action_index[action] else 0.0 for index in range(len(self.actions))]
        return [float(record[name]) for name in NUMERIC_FEATURES]+slice_values+action_values
    def fit(self,transitions:list[OfflineTransition],iterations:int=6):
        if len(transitions)<max(20,len(self.actions)*3): raise ValueError("too few transitions for fitted Q iteration")
        x=np.asarray([self._vector(item.state,item.action) for item in transitions],dtype=float); rewards=np.asarray([item.reward for item in transitions],dtype=float); targets=rewards.copy()
        for _ in range(max(1,iterations)):
            model=ExtraTreesRegressor(n_estimators=120,min_samples_leaf=2,random_state=self.random_seed,n_jobs=1); model.fit(x,targets); next_values=[]
            for item in transitions:
                if item.done: next_values.append(0.0); continue
                q=[float(model.predict(np.asarray([self._vector(item.next_state,action)]))[0]) for action in self.actions]; next_values.append(max(q))
            targets=rewards+self.gamma*np.asarray(next_values); self.model=model
        return self
    def q_values(self,observation:NetworkObservation):
        if self.model is None: raise RuntimeError("FittedQAgent has not been trained")
        return {action:float(self.model.predict(np.asarray([self._vector(observation,action)],dtype=float))[0]) for action in self.actions}
    def top_k(self,record:dict[str,Any],k:int=4): raise TypeError("Use top_k_observation for FittedQAgent because fitted Q needs the full observation")
    def top_k_observation(self,observation:NetworkObservation,k:int=4):
        q=self.q_values(observation); ordered=sorted(q.items(),key=lambda item:item[1],reverse=True)[:k]; values=np.asarray([value for _,value in ordered],dtype=float)
        if not len(values): return []
        shifted=values-np.max(values); probs=np.exp(shifted)/np.sum(np.exp(shifted)); return [(name,float(probability)) for (name,_),probability in zip(ordered,probs,strict=True)]

class ConstrainedRLPolicy:
    def __init__(self,agent:FittedQAgent,allowed:Callable[[NetworkObservation,str],bool]): self.agent=agent; self.allowed=allowed
    def choose(self,observation:NetworkObservation):
        values=self.agent.q_values(observation); ordered=sorted(values,key=values.get,reverse=True)
        for action in ordered:
            if self.allowed(observation,action): return action,values
        return "balanced",values
