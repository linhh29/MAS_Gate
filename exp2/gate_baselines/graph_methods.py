"""Dynamic-team and learned graph-pruning baseline references."""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Awaitable, Callable, Sequence

from .protocol import ChatClient, Graph


@dataclass(frozen=True)
class DyLANConfig:
    roles: tuple[str, ...] = (
        "General problem solver",
        "Logical verifier",
        "Domain specialist",
        "Constraint and edge-case critic",
    )
    rounds: int = 3
    max_calls: int = 8
    survivors: int = 2
    seed: int = 0


def _answer_key(text: str) -> str:
    boxed = re.findall(r"\\boxed\s*\{([^{}]+)\}", text)
    if boxed:
        return re.sub(r"\s+", "", boxed[-1]).lower()
    answers = re.findall(
        r"(?:final answer|answer is)\s*[:=]?\s*([^\n]+)",
        text,
        flags=re.IGNORECASE,
    )
    candidate = answers[-1] if answers else text.strip().splitlines()[-1]
    return re.sub(r"\s+", " ", candidate.strip()).lower()


def _selected_indices(text: str, count: int) -> list[int]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return []
    try:
        value = json.loads(text[start : end + 1])
        raw = value.get("selected", []) if isinstance(value, dict) else []
        selected: list[int] = []
        for item in raw:
            index = int(item) - 1
            if 0 <= index < count and index not in selected:
                selected.append(index)
        return selected
    except (ValueError, TypeError, json.JSONDecodeError):
        return []


async def dylan(
    task: str,
    agent: ChatClient,
    ranker: ChatClient,
    config: DyLANConfig = DyLANConfig(),
) -> str:
    """Run a compact DyLAN-style dynamic multi-agent reasoning process."""
    if len(config.roles) < 2 or config.rounds < 1 or config.max_calls < 1:
        raise ValueError("DyLAN requires multiple roles and a positive call budget")
    rng = random.Random(config.seed)
    active = list(range(len(config.roles)))
    histories: list[list[str]] = [[] for _ in config.roles]
    calls = 0

    for round_index in range(config.rounds):
        rng.shuffle(active)
        previous = [histories[index][-1] for index in active if histories[index]]
        updated: list[int] = []
        for index in active:
            if calls >= config.max_calls:
                break
            peer_text = "\n\n".join(
                f"Peer {position + 1}:\n{answer}"
                for position, answer in enumerate(previous)
            )
            prompt = task
            if peer_text:
                prompt += (
                    "\n\nOther agents proposed:\n"
                    + peer_text
                    + "\n\nUse them only as advice. Verify the reasoning and return "
                    "an improved answer."
                )
            answer = await agent.complete(
                [
                    {
                        "role": "system",
                        "content": (
                            config.roles[index]
                            + ". Reason independently and finish with the requested answer."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.6,
            )
            histories[index].append(answer)
            updated.append(index)
            calls += 1

        active = updated
        if not active:
            break
        counts = Counter(_answer_key(histories[index][-1]) for index in active)
        if counts.most_common(1)[0][1] > math.floor(2 * len(active) / 3):
            break

        if len(active) > config.survivors and calls < config.max_calls:
            candidate_block = "\n\n".join(
                f"[{position + 1}] {histories[index][-1]}"
                for position, index in enumerate(active)
            )
            ranking = await ranker.complete(
                [
                    {
                        "role": "system",
                        "content": (
                            "Select the most correct candidates. Return JSON only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"TASK:\n{task}\n\nCANDIDATES:\n{candidate_block}\n\n"
                            f"Return {{\"selected\":[...]}} with at most "
                            f"{config.survivors} one-based indices."
                        ),
                    },
                ],
                temperature=0.0,
            )
            calls += 1
            picked = _selected_indices(ranking, len(active))
            if picked:
                active = [active[index] for index in picked[: config.survivors]]
            else:
                active = active[: config.survivors]

    final_candidates = [histories[index][-1] for index in active if histories[index]]
    if not final_candidates:
        raise RuntimeError("DyLAN produced no answer")
    winning_key = Counter(
        _answer_key(value) for value in final_candidates
    ).most_common(1)[0][0]
    return next(
        value for value in final_candidates if _answer_key(value) == winning_key
    )


@dataclass(frozen=True)
class AgentPruneExample:
    """A data-agnostic source example paired with its candidate MAS graph."""

    value: object
    graph: Graph


GraphEvaluator = Callable[[object, Graph], Awaitable[float]]


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _edge_key(graph: Graph, edge: tuple[str, str]) -> str:
    positions = {node.node_id: index for index, node in enumerate(graph.nodes)}
    return f"{positions[edge[0]]}->{positions[edge[1]]}"


def _sample_graph(
    graph: Graph,
    logits: dict[str, float],
    rng: random.Random,
) -> tuple[Graph, dict[str, int]]:
    probabilities = {
        edge: _sigmoid(logits.get(_edge_key(graph, edge), 0.0))
        for edge in graph.edges
    }
    decisions = {
        _edge_key(graph, edge): int(rng.random() < probability)
        for edge, probability in probabilities.items()
    }
    kept = {
        edge
        for edge in graph.edges
        if decisions[_edge_key(graph, edge)] == 1
    }
    for node in graph.nodes:
        incoming = [edge for edge in graph.edges if edge[1] == node.node_id]
        if incoming and not any(edge in kept for edge in incoming):
            best = max(
                incoming,
                key=lambda edge: (probabilities[edge], _edge_key(graph, edge)),
            )
            kept.add(best)
            decisions[_edge_key(graph, best)] = 1
    return graph.with_edges([edge for edge in graph.edges if edge in kept]), decisions


def _reinforce_update(
    logits: dict[str, float],
    decisions: dict[str, int],
    advantage: float,
    learning_rate: float,
) -> None:
    for key, decision in decisions.items():
        probability = _sigmoid(logits.get(key, 0.0))
        value = logits.get(key, 0.0) + learning_rate * advantage * (
            decision - probability
        )
        logits[key] = max(-6.0, min(6.0, value))


async def train_agent_prune(
    examples: Sequence[AgentPruneExample],
    evaluate_graph: GraphEvaluator,
    *,
    iterations: int = 10,
    batch_size: int = 4,
    learning_rate: float = 0.1,
    cost_weight: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Learn canonical edge logits using source-only task utility."""
    if not examples or iterations < 1 or batch_size < 1:
        raise ValueError("examples, iterations, and batch_size must be positive")
    logits: dict[str, float] = {}
    for example in examples:
        for edge in example.graph.edges:
            logits.setdefault(_edge_key(example.graph, edge), 0.0)

    rng = random.Random(seed)
    moving_baseline = 0.0
    for _ in range(iterations):
        batch = rng.sample(list(examples), min(batch_size, len(examples)))
        for example in batch:
            sampled, decisions = _sample_graph(example.graph, logits, rng)
            score = float(await evaluate_graph(example.value, sampled))
            retained_ratio = len(sampled.edges) / max(1, len(example.graph.edges))
            utility = score - cost_weight * retained_ratio
            advantage = utility - moving_baseline
            _reinforce_update(logits, decisions, advantage, learning_rate)
            moving_baseline = 0.9 * moving_baseline + 0.1 * utility
    return dict(sorted(logits.items()))


def prune_graph(
    graph: Graph,
    edge_logits: dict[str, float],
    *,
    pruning_rate: float = 0.25,
) -> Graph:
    """Freeze an AgentPrune policy by removing the lowest-logit safe edges."""
    if not 0.0 <= pruning_rate < 1.0:
        raise ValueError("pruning_rate must be in [0, 1)")
    if len(graph.edges) <= 1 or pruning_rate == 0.0:
        return graph
    target = min(
        max(1, round(len(graph.edges) * pruning_rate)),
        len(graph.edges) - 1,
    )
    ranked = sorted(
        graph.edges,
        key=lambda edge: (edge_logits.get(_edge_key(graph, edge), 0.0), edge),
    )
    kept = set(graph.edges)
    removed = 0
    for edge in ranked:
        if removed >= target:
            break
        child = edge[1]
        if sum(candidate[1] == child for candidate in kept) <= 1:
            continue
        kept.remove(edge)
        removed += 1
    return graph.with_edges([edge for edge in graph.edges if edge in kept])
