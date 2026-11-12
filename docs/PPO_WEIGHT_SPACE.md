# Experimental PPO + Weight-Space Programming (WSP)

This workflow trains an **on-policy PPO actor-critic** on Agentic-RAN's
**action-dependent toy three-slice twin**, then edits a *small subset of the
trained actor's output-layer weights directly*. The second stage uses **no
gradient or optimizer**, unlike PPO. It is a testable WSP hypothesis, NOT a
validated performance improvement or an implementation of a production PPO xApp.

See:
- R. Takehi et al., *Sparks of Weight-Space Programming* (MIT CSAIL, 2026):
  https://rikiyat.github.io/blog/weight-space-programming/
- J. Hu et al., *Exploring Weight-Space Programming: Capabilities,
  Generalization, and Limitations* (2026): https://arxiv.org/abs/2611.09382

## Run

Requires Python 3.13 (the CI version) and opt-in PyTorch:

```bash
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[ppo]"
python -m agentic_ran.research_ppo_wsp \
  --seed 2026 --ppo-updates 30 --episodes-per-update 12 \
  --validation-episodes 12 --test-episodes 24 \
  --steps 24 --rounds 2 --output results/ppo-wsp
```

For a locally trusted PPO `state_dict` with the **exact same 9→64→64→5
architecture and observation preprocessing**:

```bash
python -m agentic_ran.research_ppo_wsp \
  --checkpoint /path/to/trusted_ppo.pt \
  --validation-episodes 12 --test-episodes 24 \
  --output results/ppo-wsp-from-checkpoint
```

The `--checkpoint` mode does **not** train PPO; the checkpoint must have been
trained with on-policy interactions in a compatible environment. PPO cannot
legitimately be trained from fixed COMMAG CSV logs.

## Experimental protocol

1. Standard PPO optimizes a clipped policy surrogate and critic with GAE,
   advantage normalization, entropy regularization and gradient clipping,
   on independent toy training episodes.
2. The trained original PPO actor and critic are frozen. A diagnostic pass
   examines hidden activations in high eMBB/mMTC/URLLC queues.
3. WSP directly modifies up to **8 actor final-layer parameters per edit**.
   Predeclared action intents favor URLLC, eMBB, mMTC or balanced allocations.
4. Proposed edits are screened on identical, independent validation seeds.
   An edit is accepted only for a measurable reward gain and no greater than
   0.015 absolute SLA-rate regression for **any** slice against original PPO.
   If no candidate passes, the original weights are retained.
5. A budget-matched random-direction weight editing control, frozen PPO, and
   PULSE-RAN rule baseline are separately evaluated. Test scenarios include
   weak-channel shift absent from validation.
6. Only after WSP selection is frozen, independent held-out episodes are
   executed with matched exogenous randomness and 95% paired bootstrap CIs.

Output:
- `ppo_frozen.pt` — original PPO state_dict
- `ppo_wsp.pt` — selected edited actor/critic state_dict
- `ppo_wsp_summary.json` — validation edit log and final held-out metrics

**Limitations:** Greedy actor selection is used at test time; PPO rollout
training remains stochastic. The benchmark is a toy network environment and
its SLA indicators are queue-threshold proxies. Neither physical RAN capacity,
real scheduler semantics nor on-air intervention is validated. No claim that
PPO-WSP beats PPO or other controllers is warranted unless independent
held-out measurements, uncertainty bounds, SLA constraints and multiple
training seeds support it. Weight editing on arbitrary PPO/SB3 network
architectures is **not** provided; adapt/export the state dict explicitly.

For the separate physical closed-loop evaluation of the existing frozen
Agentic-RAN controller, see [physical closed-loop](PHYSICAL_CLOSED_LOOP_RESEARCH.md).
