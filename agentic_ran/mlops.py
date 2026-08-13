"""Small model registry, drift monitoring, and promotion gates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from agentic_ran.domain import NetworkObservation


@dataclass(frozen=True, slots=True)
class RegistryEntry:
    version: str
    registered_at: str
    artifact_sha256: str
    metrics: dict[str, float]
    stage: str


class ModelRegistry:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self.root / "registry.json"

    @staticmethod
    def sha256_file(path: Path | str) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _load(self) -> list[dict[str, Any]]:
        if not self.index.exists():
            return []
        return json.loads(self.index.read_text(encoding="utf-8"))

    def register(self, version: str, artifact: Path | str, metrics: dict[str, float], stage: str = "candidate") -> RegistryEntry:
        entry = RegistryEntry(version=version, registered_at=datetime.now(timezone.utc).isoformat(), artifact_sha256=self.sha256_file(artifact), metrics={key: float(value) for key, value in metrics.items()}, stage=stage)
        entries = [item for item in self._load() if item.get("version") != version]
        entries.append(asdict(entry))
        self.index.write_text(json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8")
        return entry

    def promote(self, version: str, stage: str = "production") -> None:
        entries = self._load(); found = False
        for item in entries:
            if item.get("version") == version:
                item["stage"] = stage; found = True
        if not found:
            raise KeyError(version)
        self.index.write_text(json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8")


class DriftMonitor:
    def __init__(self): self._reference: dict[str, tuple[float, float]] = {}
    def fit(self, observations: list[NetworkObservation]) -> None:
        if not observations: raise ValueError("reference observations are empty")
        numeric = observations[0].feature_record()
        for key in numeric:
            if key == "slice_type": continue
            values = np.asarray([float(item.feature_record()[key]) for item in observations], dtype=float)
            self._reference[key] = (float(np.mean(values)), max(float(np.std(values)), 1e-6))
    def score(self, observations: list[NetworkObservation]) -> float:
        if not self._reference or not observations: return 0.0
        scores=[]
        for key,(mean,std) in self._reference.items():
            current=float(np.mean([float(item.feature_record()[key]) for item in observations])); scores.append(min(1.0,abs(current-mean)/(4.0*std)))
        return round(float(np.mean(scores)),4)


class PromotionGate:
    def __init__(self,min_macro_f1:float=0.75,max_drift:float=0.35,min_safety_rate:float=0.995): self.min_macro_f1=min_macro_f1; self.max_drift=max_drift; self.min_safety_rate=min_safety_rate
    def evaluate(self,metrics:dict[str,float]):
        reasons=[]
        if metrics.get("macro_f1",0.0)<self.min_macro_f1: reasons.append("macro F1 is below the promotion threshold")
        if metrics.get("drift_score",0.0)>self.max_drift: reasons.append("drift score exceeds the promotion threshold")
        if metrics.get("safety_rate",0.0)<self.min_safety_rate: reasons.append("safety rate is below the promotion threshold")
        return not reasons,reasons
