"""Output-repair baselines: Self-Refine and AgentDropoutV2-style repair."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .protocol import ChatClient, ExecutionState


async def self_refine(
    task: str,
    candidate: str,
    critic: ChatClient,
    refiner: ChatClient,
    *,
    iterations: int = 1,
) -> str:
    """Apply the feedback-then-revision loop from Self-Refine."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    current = candidate
    for _ in range(iterations):
        feedback = await critic.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Critique the candidate using only the task and answer. "
                        "Identify specific factual, logical, completeness, or "
                        "format defects. Do not write a replacement answer."
                    ),
                },
                {
                    "role": "user",
                    "content": f"TASK:\n{task}\n\nCANDIDATE:\n{current}",
                },
            ],
            temperature=0.0,
        )
        current = await refiner.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Revise the candidate using the feedback. Independently "
                        "verify every correction and return only the requested answer."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"TASK:\n{task}\n\nCANDIDATE:\n{current}"
                        f"\n\nFEEDBACK:\n{feedback}"
                    ),
                },
            ],
            temperature=0.0,
        )
    return current


@dataclass(frozen=True)
class FailureIndicator:
    name: str
    trigger: str
    guidance: str


@dataclass(frozen=True)
class RepairPlan:
    """One bounded correction and the graph suffix affected by it."""

    node_id: str
    replacement: str | None = None
    removed_edges: tuple[tuple[str, str], ...] = ()
    replay_nodes: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_]+", text.lower()))


def _retrieve_indicators(
    query: str,
    indicators: Sequence[FailureIndicator],
    top_k: int,
) -> list[FailureIndicator]:
    query_tokens = _tokens(query)
    ranked = sorted(
        indicators,
        key=lambda item: (
            -len(query_tokens & _tokens(item.trigger + " " + item.guidance)),
            item.name,
        ),
    )
    return ranked[: min(top_k, len(ranked))]


def _json_object(text: str) -> dict[str, object]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model response contains no JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model response must be a JSON object")
    return value


async def _audit(
    *,
    task: str,
    role: str,
    output: str,
    indicators: Sequence[FailureIndicator],
    auditor: ChatClient,
) -> tuple[bool, str]:
    indicator_text = "\n\n".join(
        f"[{item.name}]\nConcern: {item.trigger}\nGuidance: {item.guidance}"
        for item in indicators
    )
    raw = await auditor.complete(
        [
            {
                "role": "system",
                "content": (
                    "Audit this intermediate milestone. Mark it flawed only for a "
                    "concrete, fatal, actionable error that can change the final result. "
                    "Ignore style. Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"GLOBAL TASK:\n{task}\n\nROLE:\n{role}\n\nOUTPUT:\n{output}"
                    f"\n\nFAILURE INDICATORS:\n{indicator_text}\n\n"
                    'Return {"is_flawed":true|false,'
                    '"impact_changes_result":true|false,"suggestion":"..."}.'
                ),
            },
        ],
        temperature=0.0,
    )
    try:
        verdict = _json_object(raw)
    except (ValueError, json.JSONDecodeError):
        return False, ""
    flawed = bool(verdict.get("is_flawed")) and bool(
        verdict.get("impact_changes_result")
    )
    return flawed, str(verdict.get("suggestion") or "").strip()


async def agent_dropout_v2(
    state: ExecutionState,
    indicators: Sequence[FailureIndicator],
    auditor: ChatClient,
    refiner: ChatClient,
    *,
    top_k: int = 5,
) -> RepairPlan | None:
    """Return the first justified rectify-or-isolate intervention.

    Only completed non-final milestones are inspected. A repaired milestone is
    re-audited before acceptance; otherwise its outgoing handoff is isolated when
    doing so leaves every affected child connected. At most one intervention is
    returned, matching the Runtime Editor budget used by the comparison.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not indicators or state.interventions_left <= 0:
        return None

    for node_id in state.graph.topological_order():
        if node_id not in state.outputs:
            continue
        node = state.graph.node(node_id)
        if node.is_final:
            continue
        selected = _retrieve_indicators(
            f"{state.task}\n{node.role}\n{state.outputs[node_id]}",
            indicators,
            top_k,
        )
        flawed, suggestion = await _audit(
            task=state.task,
            role=node.role,
            output=state.outputs[node_id],
            indicators=selected,
            auditor=auditor,
        )
        if not flawed:
            continue

        upstream = {
            parent: state.outputs[parent]
            for parent in state.graph.parents(node_id)
            if parent in state.outputs
        }
        replacement = await refiner.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Correct only the assigned milestone using the concrete "
                        "audit feedback. Preserve valid work and return only the "
                        "corrected milestone."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"GLOBAL TASK:\n{state.task}\n\nROLE:\n{node.role}"
                        f"\n\nUPSTREAM:\n{json.dumps(upstream, ensure_ascii=False)}"
                        f"\n\nORIGINAL:\n{state.outputs[node_id]}"
                        f"\n\nAUDIT FEEDBACK:\n{suggestion}"
                    ),
                },
            ],
            temperature=0.0,
        )
        still_flawed, _ = await _audit(
            task=state.task,
            role=node.role,
            output=replacement,
            indicators=selected,
            auditor=auditor,
        )
        descendants = state.graph.descendants(node_id)
        replay = tuple(
            candidate
            for candidate in state.graph.topological_order()
            if candidate in descendants
        )
        if not still_flawed:
            return RepairPlan(
                node_id=node_id,
                replacement=replacement,
                replay_nodes=replay,
                metadata={"decision": "RECTIFY"},
            )

        outgoing = tuple(
            edge for edge in state.graph.edges if edge[0] == node_id
        )
        safe_to_remove = outgoing and all(
            len(state.graph.parents(child)) > 1 for _, child in outgoing
        )
        if safe_to_remove:
            return RepairPlan(
                node_id=node_id,
                removed_edges=outgoing,
                replay_nodes=replay,
                metadata={"decision": "ISOLATE"},
            )
        return None
    return None
