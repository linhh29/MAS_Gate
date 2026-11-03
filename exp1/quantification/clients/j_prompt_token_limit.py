"""Adaptive J-scoring prompt token limit learned from runtime OOM."""

from __future__ import annotations

import math
import threading
from typing import Callable, Sequence

from quantification.clients.j_tilde import tokenize_context_and_answer
from quantification.graph.data_nodes import is_data_node
from quantification.graph.structure import GraphStructure
from quantification.schemas import ReasoningGraph

_lock = threading.Lock()
_max_prompt_tokens: float = math.inf


class JPromptTokenLimitExceeded(RuntimeError):
    """Prompt exceeds the learned J-scoring token limit (or triggered OOM)."""

    def __init__(
        self,
        prompt_tokens: int,
        limit: int,
        *,
        from_oom: bool = False,
        node_id: str = "",
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.limit = limit
        self.from_oom = from_oom
        self.node_id = node_id
        kind = "OOM" if from_oom else "over limit"
        node_part = f" node={node_id}" if node_id else ""
        super().__init__(
            f"J scoring skipped ({kind}): prompt_tokens={prompt_tokens} "
            f"learned_limit={limit}{node_part}"
        )


def get_max_j_prompt_tokens() -> float:
    with _lock:
        return _max_prompt_tokens


def finite_max_j_prompt_tokens() -> int | None:
    """Return the learned limit as int, or None while still unlimited."""
    limit = get_max_j_prompt_tokens()
    if math.isinf(limit):
        return None
    return int(limit)


def format_learned_j_prompt_token_limit() -> str:
    limit = get_max_j_prompt_tokens()
    if math.isinf(limit):
        return "unlimited"
    return str(int(limit))


def reset_max_j_prompt_tokens() -> None:
    global _max_prompt_tokens
    with _lock:
        _max_prompt_tokens = math.inf


def prompt_tokens_allowed(prompt_tokens: int) -> bool:
    return prompt_tokens <= get_max_j_prompt_tokens()


def tighten_max_j_prompt_tokens_from_oom(failed_prompt_tokens: int) -> int:
    """
    After OOM on a prompt of length N, only allow prompts with length <= N-1.
    Repeated OOMs monotonically tighten the limit.
    """
    global _max_prompt_tokens
    with _lock:
        prev = _max_prompt_tokens
        new_limit = min(_max_prompt_tokens, max(1, failed_prompt_tokens - 1))
        _max_prompt_tokens = new_limit
        if new_limit < prev:
            print(
                f"[j-token-limit] tightened max prompt tokens to {int(new_limit)} "
                f"(OOM at {failed_prompt_tokens})",
                flush=True,
            )
        return int(new_limit)


def check_prompt_tokens_or_raise(
    prompt_tokens: int,
    *,
    node_id: str = "",
) -> None:
    limit = get_max_j_prompt_tokens()
    if prompt_tokens > limit:
        raise JPromptTokenLimitExceeded(
            prompt_tokens,
            int(limit),
            from_oom=False,
            node_id=node_id,
        )


def max_prompt_tokens_for_graph(
    graph: ReasoningGraph,
    tokenizer: object,
    *,
    format_context: Callable[[str, Sequence[str]], str],
    ground_truth: str,
) -> tuple[int, str]:
    """Longest teacher-forced prompt (context + answer) across all J calls."""
    if not ground_truth.strip():
        raise ValueError("ground_truth is required")

    structure = GraphStructure(graph)
    outputs = graph.output_by_id()
    order = [nid for nid in structure.execution_order() if not is_data_node(nid)]

    max_tokens = 0
    worst_node = ""
    prefix_outputs: list[str] = []
    for node_id in order:
        prefix_outputs.append(outputs.get(node_id, ""))
        context = format_context(graph.question, prefix_outputs)
        ctx_ids, ans_ids = tokenize_context_and_answer(tokenizer, context, ground_truth)
        prompt_len = len(ctx_ids) + len(ans_ids)
        if prompt_len > max_tokens:
            max_tokens = prompt_len
            worst_node = node_id
    return max_tokens, worst_node
