"""Impact-aware paired reward and direct GRPO training formulation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .editor import Action, ActionKind


def node_quantities(
    local_delta: dict[str, float],
    children: dict[str, list[str]],
    edge_entailment: dict[tuple[str, str], float],
) -> dict[str, dict[str, float]]:
    """Compute the paper's U, K, and I on a completed DAG.

    ``local_delta`` is ``(J(C_i+, y)-J(C_i-, y))/abs(J_0)``.  The
    teacher-forced J scores and edge entailment probabilities are privileged
    training signals; they never enter the Editor observation.
    """
    utility = {node: math.tanh(delta) for node, delta in local_delta.items()}
    def descendants(start: str) -> set[str]:
        seen, pending = set(), list(children.get(start, []))
        while pending:
            child = pending.pop()
            if child not in seen:
                seen.add(child)
                pending.extend(children.get(child, []))
        return seen

    reach = {node: len(descendants(node)) for node in utility}
    max_reach = max(reach.values(), default=0)
    g = {node: count / max_reach if max_reach else 0.0 for node, count in reach.items()}

    quantities: dict[str, dict[str, float]] = {}
    for node, value in utility.items():
        k_value = sum(
            edge_entailment.get((node, child), 0.0) * (1.0 + g[child])
            for child in children.get(node, [])
        )
        quantities[node] = {
            "utility": value,
            "propagation": k_value,
            "impact": value * (1.0 + k_value),
            "risk": max(-value, 0.0) * (1.0 + k_value),
        }
    return quantities


def trace_quality(quantities: dict[str, dict[str, float]]) -> float:
    """Q(tau): mean node impact for one completed execution trace."""
    if not quantities:
        return 0.0
    return sum(item["impact"] for item in quantities.values()) / len(quantities)


ACTION_COST = {
    ActionKind.KEEP: 0.00,
    ActionKind.ANNOTATE_CURRENT: 0.03,
    ActionKind.GUIDE_CHILD: 0.03,
    ActionKind.RETRY_CURRENT: 0.08,
    ActionKind.INSERT_SUBTASK: 0.10,
    ActionKind.BACKTRACK_PARENT: 0.12,
}


@dataclass(frozen=True)
class Counterfactual:
    action: Action
    task_score: float
    trace_quality: float
    valid: bool = True


@dataclass(frozen=True)
class RewardConfig:
    quality_weight: float = 0.1
    cost_weight: float = 1.0
    invalid_reward: float = -0.25


def paired_reward(
    keep: Counterfactual,
    edited: Counterfactual,
    config: RewardConfig = RewardConfig(),
) -> float:
    """Reward an action only for improvement over the same gate's KEEP trace."""
    if not edited.valid:
        return config.invalid_reward
    if edited.action.kind is ActionKind.KEEP:
        return 0.0
    return (
        edited.task_score - keep.task_score
        + config.quality_weight * (edited.trace_quality - keep.trace_quality)
        - config.cost_weight * ACTION_COST[edited.action.kind]
    )


def group_relative_advantages(rewards: list[float], epsilon: float = 1e-6) -> list[float]:
    """Normalize rewards among actions sampled at the same gate."""
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    if len(rewards) == 1:
        return [0.0]
    variance = sum((reward - mean) ** 2 for reward in rewards) / (len(rewards) - 1)
    scale = math.sqrt(variance + epsilon)
    return [(reward - mean) / scale for reward in rewards]


def paired_group(
    keep: Counterfactual,
    sampled: list[Counterfactual],
    config: RewardConfig = RewardConfig(),
) -> list[tuple[Action, float, float]]:
    """Produce ``(action, reward, GRPO advantage)`` for one offline gate.

    The KEEP continuation is cached once. Each sampled action changes only the
    affected suffix, is compared with that same cache, and is optimized directly
    with GRPO; there is no supervised warm-up stage.
    """
    rewards = [paired_reward(keep, candidate, config) for candidate in sampled]
    advantages = group_relative_advantages(rewards)
    return [
        (candidate.action, reward, advantage)
        for candidate, reward, advantage in zip(sampled, rewards, advantages)
    ]


@dataclass(frozen=True)
class ExperimentConfig:
    train_domain: str
    examples: int
    consumed_examples: int
    steps: int
    checkpoints: tuple[int, ...]
    batch_size: int = 256
    rollouts_per_gate: int = 16
    learning_rate: float = 1e-6
    epochs: int = 1
    lora_rank: int = 64
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    max_interventions: int = 1
    max_extra_calls: int = 4


MAIN_EXPERIMENTS = {
    "ALFWorld": ExperimentConfig(
        train_domain="ALFWorld",
        examples=4032,
        consumed_examples=3840,
        steps=15,
        checkpoints=(2, 3, 5, 6, 8, 9, 11, 12, 14, 15),
    ),
    "HotpotQA": ExperimentConfig(
        train_domain="HotpotQA",
        examples=10000,
        consumed_examples=9984,
        steps=39,
        checkpoints=(4, 8, 12, 16, 20, 24, 28, 32, 36, 39),
    ),
    "DeepScaleR": ExperimentConfig(
        train_domain="DeepScaleR",
        examples=10000,
        consumed_examples=9984,
        steps=39,
        checkpoints=(4, 8, 12, 16, 20, 24, 28, 32, 36, 39),
    ),
}
