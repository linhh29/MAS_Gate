"""One task-level Editor episode with successive KEEP decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from .editor import Action, ActionKind, validate_action
from .prompts import initial_messages, next_gate_message
from .quality import score_trace
from .runtime import Trace, Worker, execute_edit
from .training import Counterfactual, RewardConfig, paired_reward


class EditorSampler(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> tuple[str, list[int], list[int]]:
        """Return raw action, prompt token IDs, and generated action token IDs."""

    def count_tokens(self, text: str) -> int: ...


@dataclass
class Turn:
    ordinal: int
    node_id: str
    raw_action: str
    prompt_ids: list[int]
    action_ids: list[int]
    reward: float
    status: str


def sample_episode(
    *, reference: Trace, answer: str, reference_c: float, reference_q: float,
    sampler: EditorSampler, worker: Worker, quantifier,
    correctness: Callable[[str, str], float],
    reward_config: RewardConfig = RewardConfig(), max_output_chars: int = 6000,
) -> list[Turn]:
    """Score KEEP locally and the first intervention against the same reference."""
    order = reference.graph.order()
    messages: list[dict[str, str]] = []
    turns: list[Turn] = []
    baseline = Counterfactual(Action(ActionKind.KEEP), reference_c, reference_q)
    for ordinal, node_id in enumerate(order):
        state = reference.gate(ordinal)
        if ordinal == 0:
            messages = initial_messages(state, max_output_chars)
        else:
            messages.append(next_gate_message(state, max_output_chars))
        raw, prompt_ids, action_ids = sampler.generate(messages)
        if not action_ids:
            raise RuntimeError("Editor produced no action tokens")
        try:
            action = Action.parse(raw)
            validate_action(state, action, sampler.count_tokens)
        except (ValueError, KeyError, TypeError):
            turns.append(Turn(ordinal, node_id, raw, prompt_ids, action_ids,
                              reward_config.invalid_reward, "invalid"))
            break
        if action.kind is ActionKind.KEEP:
            turns.append(Turn(ordinal, node_id, raw, prompt_ids, action_ids, 0.0, "keep"))
            messages.append({"role": "assistant", "content": raw})
            continue
        edited = execute_edit(state, action, worker)
        edited_c, edited_q = score_trace(
            reference.task, answer, edited.graph, edited.outputs, quantifier, correctness
        )
        candidate = Counterfactual(action, edited_c, edited_q)
        reward = paired_reward(baseline, candidate, reward_config)
        turns.append(Turn(ordinal, node_id, raw, prompt_ids, action_ids,
                          reward, "intervention"))
        break
    return turns
