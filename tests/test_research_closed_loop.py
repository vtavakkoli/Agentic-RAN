"""Physical capacity/closed-loop tests; the mock is NOT ns-3 or 5G-LENA."""
import json
import sys

import numpy as np
import pytest

from agentic_ran.research_physical_prb import (
    CellBudget, JointAction, discrete_joint_catalog, project_reference_action,
)
from agentic_ran.research_ns3_session import ExternalClosedLoopSession
import agentic_ran.research_closed_loop as ctl


def physical(total=52, reserve=4):
    return CellBudget.from_handshake({
        "carrier_id": "car0", "bwp_id": "bwp0", "total_prbs": total,
        "reserved_prbs": reserve, "numerology": 1, "source": "configured_ns3_demo",
    }, "ns3-event-demo")


def test_integer_physical_budget_and_exact_projection():
    cell = physical()
    assert cell.usable_prbs == 48
    assert all(sum(a.prbs) == 48 for a in discrete_joint_catalog(cell))
    with pytest.raises(ValueError, match="exceeds"):
        JointAction((30, 15, 10)).validate(cell)
    projected, changed = project_reference_action((20, 20, 20), cell)
    assert changed and projected.prbs == (16, 16, 16)
    unmodified, changed = project_reference_action((16, 16, 16), cell)
    assert not changed and unmodified.prbs == (16, 16, 16)
    with pytest.raises(ValueError, match="historical"):
        CellBudget.from_handshake({
            "carrier_id": "x", "bwp_id": "bwp", "total_prbs": 14,
            "reserved_prbs": 0, "numerology": 1,
            "source": "historical_allocation_quantile",
        }, "ns3-5glena")


MOCK = '''
import json,sys
slices=("eMBB","mMTC","URLLC")
step=0
def obs():
  return {"type":"observation","step":step,"slices":{
    s:{"metrics":{"dl_buffer_bytes":float(1500+300*step),
    "dl_cqi":8.0,"ul_sinr":9.0,"requested_prbs":12.0,
    "grant_ratio":.75,"dl_errors_pct":2.0,"num_ues":3.0}}
    for s in slices}}
print(json.dumps({"type":"hello","protocol":"jagc-closed-loop-v1",
  "backend":"ns3-event-demo","real_actuation":False,
  "cell":{"carrier_id":"c","bwp_id":"b","total_prbs":52,
  "reserved_prbs":4,"numerology":1,"source":"configured_ns3_demo"}}),flush=True)
for line in sys.stdin:
  data=json.loads(line)
  if data["type"]=="reset":
    step=0
    print(json.dumps(obs()),flush=True)
  else:
    prb=[data["action"]["slices"][s]["prbs"] for s in slices]
    step+=1
    print(json.dumps({"type":"transition","observation":obs(),
       "reward":.4+.01*prb[0]-.004*prb[2],
       "done":step>=4}),flush=True)
'''


def command(tmp_path):
    script = tmp_path / "FAKE_NOT_A_REAL_SIM.py"
    script.write_text(MOCK)
    return [sys.executable, str(script)]


def test_real_mode_fails_closed_and_actions_cause_distinct_observations(tmp_path):
    with pytest.raises(RuntimeError, match="refusing synthetic"):
        ExternalClosedLoopSession(command(tmp_path))
    with ExternalClosedLoopSession(command(tmp_path), require_real=False) as session:
        assert session.reset(1, "normal")["step"] == 0
        _, a, _ = session.step(JointAction((20, 14, 14)))
        session.reset(1, "normal")
        _, b, _ = session.step(JointAction((12, 14, 22)))
        assert a != b
        with pytest.raises(ValueError, match="exceeds"):
            session.step(JointAction((30, 20, 10)))


class FrozenTest:
    def __init__(self, path, fold):
        self.path, self.fold = str(path), fold

    def choose(self, obs, budget):
        return JointAction((16, 16, 16)), False


def test_full_action_variation_validation_and_paired_results(tmp_path, monkeypatch):
    monkeypatch.setattr(ctl, "FrozenOriginal", FrozenTest)
    out = ctl.run_study(
        command(tmp_path), tmp_path / "unused.joblib", tmp_path / "out",
        require_real=False, train_episodes=6, validation_episodes=3,
        test_episodes=4, steps=4, seed=42,
    )
    assert not out["real_phy_actuation_claim"]
    assert out["training_executed_action_variants"] >= 2
    assert out["training_episodes_with_action_variation"] >= 1
    assert not out["test_used_for_selection"]
    assert out["test_paired"]["paired_scenario_seed_episodes"] == 4
    assert len(out["test_paired"]["episode_bootstrap_95ci"]) == 2
    assert (tmp_path / "out" / "jagc_closed_loop_summary.json").exists()
    assert len(json.loads((tmp_path / "out" / "jagc_closed_loop_summary.json").read_text())["validation_candidates"]) == 4
