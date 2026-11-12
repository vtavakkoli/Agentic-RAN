"""Experimental PPO + sparse Weight-Space Programming on an INTERVENTIONAL TOY RAN.

The initial policy is trained with ordinary on-policy clipped PPO. The WSP
stage edits *only* the actor's last-layer weights, without gradients or a
training optimizer. Every edit is evaluated on independent validation seeds,
and rejected if it fails a predeclared reward/SLA gate. Held-out test
episodes are evaluated once after the edits are frozen.

This is NOT a Colosseum, ns-3/5G-LENA, or live O-RAN result. WSP does not
guarantee a gain over PPO. Requires optional dependency: pip install -e '.[ppo]'.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from agentic_ran.research_twin_environment import (
    ACTIONS, MultiSliceToyTwin, SCENARIOS, rollout_episode,
)

# All selectors are scored on the same (scenario, seed) pairs.
TRAIN_SCENARIOS = ("normal", "bursty", "congested")
VALIDATION_SCENARIOS = TRAIN_SCENARIOS
TEST_SCENARIOS = SCENARIOS
SLA_TOLERANCE = 0.015


class PPOActorCritic(nn.Module):
    """Small, auditable shared-body, discrete-action actor-critic."""

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(9, 64), nn.Tanh(),
                                  nn.Linear(64, 64), nn.Tanh())
        self.actor = nn.Linear(64, len(ACTIONS))
        self.critic = nn.Linear(64, 1)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(states)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    def choose(self, state: np.ndarray, _previous: int, _t: int) -> int:
        with torch.no_grad():
            logits, _ = self(torch.as_tensor(state, dtype=torch.float32))
        return int(logits.argmax().item())


def train_ppo(
    *, seed: int, updates: int = 30, episodes_per_update: int = 12,
    steps: int = 24, learning_rate: float = 3e-4,
) -> PPOActorCritic:
    """PPO clipped surrogate, GAE(lambda), entropy and value objectives."""
    if updates < 1 or episodes_per_update < 2 or steps < 3:
        raise ValueError("invalid PPO training episode budget")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = PPOActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for update in range(updates):
        states, acts, old_logps, advantages, returns = [], [], [], [], []
        for episode in range(episodes_per_update):
            scenario = TRAIN_SCENARIOS[(update * episodes_per_update + episode) % 3]
            env = MultiSliceToyTwin(seed=seed + 101 * update + episode + 1, scenario=scenario)
            ep_states, ep_acts, ep_logps, ep_rewards, ep_values = [], [], [], [], []
            for _ in range(steps):
                state = torch.as_tensor(env.state.copy(), dtype=torch.float32)
                with torch.no_grad():
                    logits, value = model(state)
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    logp = dist.log_prob(action)
                _, reward, _ = env.step(int(action.item()))
                ep_states.append(state)
                ep_acts.append(action)
                ep_logps.append(logp)
                ep_rewards.append(float(reward))
                ep_values.append(float(value.item()))
            # Each finite episode horizon is a terminal boundary.
            gae, next_value = 0.0, 0.0
            ep_advantages = []
            for t in range(steps - 1, -1, -1):
                delta = ep_rewards[t] + 0.99 * next_value - ep_values[t]
                gae = delta + 0.99 * 0.95 * gae
                ep_advantages.append(gae)
                next_value = ep_values[t]
            ep_advantages.reverse()
            states.extend(ep_states)
            acts.extend(ep_acts)
            old_logps.extend(ep_logps)
            advantages.extend(ep_advantages)
            returns.extend([adv + val for adv, val in zip(ep_advantages, ep_values)])
        x = torch.stack(states)
        a = torch.stack(acts)
        old = torch.stack(old_logps)
        adv = torch.tensor(advantages, dtype=torch.float32)
        ret = torch.tensor(returns, dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        for _ in range(4):
            for indices in torch.randperm(len(a)).split(128):
                logits, values = model(x[indices])
                dist = Categorical(logits=logits)
                ratio = (dist.log_prob(a[indices]) - old[indices]).exp()
                clipped = ratio.clamp(0.8, 1.2)
                policy_loss = -torch.minimum(
                    ratio * adv[indices], clipped * adv[indices],
                ).mean()
                value_loss = 0.5 * (values - ret[indices]).square().mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * dist.entropy().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
    model.eval()
    return model


def _evaluations(
    selector, *, seed: int, count: int, steps: int, scenarios: tuple[str, ...],
) -> list[dict[str, Any]]:
    if count < 4:
        raise ValueError("need >=4 independently seeded evaluation episodes")
    return [{
        "seed": seed + i * 997, "scenario": scenarios[i % len(scenarios)],
        **rollout_episode(
            selector, seed=seed + i * 997,
            scenario=scenarios[i % len(scenarios)], steps=steps,
        ),
    } for i in range(count)]


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def _eligible(proposal: list[dict], original: list[dict]) -> bool:
    # Original frozen PPO is the same conservative safety reference for
    # every candidate. The WSP optimizer may not weaken these thresholds.
    return all(
        _mean(proposal, name) <= _mean(original, name) + SLA_TOLERANCE
        for name in ("sla_URLLC", "sla_eMBB", "sla_mMTC")
    )


def _signature(
    model: PPOActorCritic, *, seed: int, episodes: int, steps: int,
) -> dict[str, np.ndarray]:
    """Read frozen hidden activations in troublesome network states.

    This yields diagnostic intervention directions; edits are NOT gradients.
    """
    collected = {"URLLC": [], "eMBB": [], "mMTC": [], "balanced": []}
    with torch.no_grad():
        for i in range(episodes):
            env = MultiSliceToyTwin(seed=seed + 7013 * i, scenario=TRAIN_SCENARIOS[i % 3])
            for t in range(steps):
                state = torch.tensor(env.state.copy(), dtype=torch.float32)
                features = model.body(state).detach().cpu().numpy()
                if state[2] > 0.45:
                    collected["URLLC"].append(features)
                if state[0] > 1.0:
                    collected["eMBB"].append(features)
                if state[1] > 0.75:
                    collected["mMTC"].append(features)
                collected["balanced"].append(features)
                env.step(model.choose(env.state, env.previous_action, t))
    return {
        key: np.mean(values, axis=0) if values else np.zeros(64)
        for key, values in collected.items()
    }


def _edit(
    model: PPOActorCritic, *,
    action: int, signature: np.ndarray, amplitude: float,
    count: int = 8,
) -> PPOActorCritic:
    """Bounded, localized, reversible edits to the PPO actor output layer."""
    if not 0 <= action < len(ACTIONS) or not np.isfinite(amplitude):
        raise ValueError("invalid weight-space edit")
    candidate = copy.deepcopy(model)
    indices = np.argsort(-np.abs(signature), kind="stable")[:count]
    direction = np.sign(signature[indices])
    with torch.no_grad():
        candidate.actor.weight[action, indices] += torch.as_tensor(
            amplitude * direction / np.sqrt(count), dtype=torch.float32,
        )
    return candidate


def _search(
    original: PPOActorCritic, original_val: list[dict], *,
    seed: int, validation_seed: int, val_episodes: int, steps: int,
    rounds: int, random_control: bool = False,
) -> tuple[PPOActorCritic, list[dict]]:
    rng = np.random.default_rng(seed)
    current = copy.deepcopy(original)
    history: list[dict] = []
    reference_score = _mean(original_val, "return")
    directions = _signature(original, seed=seed + 10000, episodes=12, steps=steps)
    # Predeclared action intents derived only from PHYSICAL TOY share allocation:
    # action 1/4 prioritize URLLC, 2 eMBB, 3 mMTC, 0 is balanced.
    targets = [
        (1, directions["URLLC"]), (4, directions["URLLC"]),
        (2, directions["eMBB"]), (3, directions["mMTC"]),
        (0, directions["balanced"]),
    ]
    best = reference_score
    for phase in range(rounds):
        winner = current
        winner_description = None
        candidate_best = best
        for j, (target, signature) in enumerate(targets):
            for magnitude in (0.20, 0.45):
                for sign in (-1, 1):
                    if random_control:
                        target = int(rng.integers(len(ACTIONS)))
                        signature = rng.normal(size=64)
                    attempt = _edit(
                        current, action=target, signature=signature,
                        amplitude=sign * magnitude,
                    )
                    rows = _evaluations(
                        attempt.choose, seed=validation_seed,
                        count=val_episodes, steps=steps,
                        scenarios=VALIDATION_SCENARIOS,
                    )
                    score = _mean(rows, "return")
                    safe = _eligible(rows, original_val)
                    if safe and score > candidate_best + 1e-4:
                        candidate_best, winner = score, attempt
                        winner_description = {
                            "round": phase, "proposal_index": j,
                            "action": target, "signed_amplitude": sign * magnitude,
                            "validation_return": score,
                            "validation_sla_URLLC": _mean(rows, "sla_URLLC"),
                        }
        if winner_description is None:
            break
        history.append(winner_description)
        current, best = winner, candidate_best
    return current, history


def _paired(candidate: list[dict], reference: list[dict], seed: int) -> dict[str, Any]:
    assert [(x["seed"], x["scenario"]) for x in candidate] == [
        (x["seed"], x["scenario"]) for x in reference
    ]
    delta = np.asarray([a["return"] - b["return"]
                        for a, b in zip(candidate, reference)], dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(2000, len(delta)))
    ci = np.quantile(delta[indices].mean(axis=1), (0.025, 0.975))
    return {
        "mean_paired_delta": float(delta.mean()),
        "bootstrap_95ci": ci.tolist(),
        "episode_win_rate": float(np.mean(delta > 0)),
        "episodes": int(len(delta)),
    }


def run(
    *, seed: int = 2026, updates: int = 30,
    episodes_per_update: int = 12, validation_episodes: int = 12,
    test_episodes: int = 24, steps: int = 24, rounds: int = 2,
    checkpoint: Path | None = None,
) -> tuple[dict[str, Any], PPOActorCritic, PPOActorCritic]:
    """Frozen original PPO -> validation WSP selection -> untouched final test."""
    if min(validation_episodes, test_episodes) < 4 or not 1 <= rounds <= 5:
        raise ValueError("invalid validation/test split or edit budget")
    if checkpoint is None:
        original = train_ppo(seed=seed + 100000, updates=updates,
                             episodes_per_update=episodes_per_update, steps=steps)
    else:
        # Only open trusted state_dicts; user must ensure its PPO architecture,
        # normalizer and feature ordering match this 9->64->64->5 model.
        original = PPOActorCritic()
        original.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        original.eval()
    val_seed = seed + 20000000
    original_val = _evaluations(
        original.choose, seed=val_seed, count=validation_episodes,
        steps=steps, scenarios=VALIDATION_SCENARIOS,
    )
    edited, edits = _search(
        original, original_val, seed=seed + 310,
        validation_seed=val_seed, val_episodes=validation_episodes,
        steps=steps, rounds=rounds,
    )
    randomized, random_edits = _search(
        original, original_val, seed=seed + 719,
        validation_seed=val_seed, val_episodes=validation_episodes,
        steps=steps, rounds=rounds, random_control=True,
    )
    test_seed = seed + 30000000
    def pulse(state, _previous, _t):
        return 1 if state[2] > 0.45 else 2 if state[0] > 1.0 else 0
    selectors = {
        "ppo_frozen": original.choose, "ppo_wsp": edited.choose,
        "ppo_random_edit_control": randomized.choose,
        "pulse_ran": pulse,
    }
    heldout = {
        name: _evaluations(policy, seed=test_seed,
                           count=test_episodes, steps=steps, scenarios=TEST_SCENARIOS)
        for name, policy in selectors.items()
    }
    summary = {
        "scope": "interventional_action_dependent_toy_not_REAL_5G_NR",
        "algorithm": "trained_PPO_plus_sparse_gradient_free_validation_selected_head_edits",
        "train_episodes_total": None if checkpoint else updates * episodes_per_update,
        "validation_episodes": validation_episodes,
        "heldout_test_episodes": test_episodes,
        "edited_parameters_per_step_at_most": 8,
        "wsp_accepted_edits": edits,
        "random_accepted_edits": random_edits,
        "test_used_for_edit_selection": False,
        "test_results": {
            key: {
                metric: _mean(rows, metric)
                for metric in ("return", "sla_eMBB", "sla_mMTC", "sla_URLLC", "switch_rate")
            } for key, rows in heldout.items()
        },
        "paired_vs_frozen_ppo": {
            name: _paired(rows, heldout["ppo_frozen"], seed + 82)
            for name, rows in heldout.items() if name != "ppo_frozen"
        },
        "limitations": [
            "WSP superiority is not assumed: examine held-out paired CIs and SLA metrics.",
            "A toy queue simulator does not validate NR PHY, RIC, Colosseum, or 5G-LENA.",
            "The test comparison uses deterministic greedy selection; PPO training is stochastic.",
            "Only head-level weight edits are supported; WSP does not replace PPO training.",
        ],
    }
    return summary, original, edited


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO + validation-only weight-space edits (toy RAN)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--ppo-updates", type=int, default=30)
    parser.add_argument("--episodes-per-update", type=int, default=12)
    parser.add_argument("--validation-episodes", type=int, default=12)
    parser.add_argument("--test-episodes", type=int, default=24)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--checkpoint", type=Path, help="TRUSTED matching PPO state_dict")
    parser.add_argument("--output", type=Path, default=Path("results/ppo-wsp"))
    args = parser.parse_args()
    result, original, edited = run(
        seed=args.seed, updates=args.ppo_updates,
        episodes_per_update=args.episodes_per_update,
        validation_episodes=args.validation_episodes,
        test_episodes=args.test_episodes, steps=args.steps, rounds=args.rounds,
        checkpoint=args.checkpoint,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(original.state_dict(), args.output / "ppo_frozen.pt")
    torch.save(edited.state_dict(), args.output / "ppo_wsp.pt")
    (args.output / "ppo_wsp_summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
