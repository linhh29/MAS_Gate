"""Public Runtime Editor state and the six constrained actions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from graphlib import TopologicalSorter
from typing import Any, Callable


class ActionKind(str, Enum):
    KEEP = "KEEP"
    BACKTRACK_PARENT = "BACKTRACK_PARENT"
    RETRY_CURRENT = "RETRY_CURRENT"
    ANNOTATE_CURRENT = "ANNOTATE_CURRENT"
    GUIDE_CHILD = "GUIDE_CHILD"
    INSERT_SUBTASK = "INSERT_SUBTASK"


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    target: str | None = None
    payload: dict[str, str] = field(default_factory=dict)

    @classmethod
    def parse(cls, text: str) -> "Action":
        text = text.rsplit("</think>", 1)[-1].strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Editor response contains no JSON action")
        value = json.loads(text[start : end + 1])
        if not isinstance(value, dict) or set(value) - {"action", "target", "payload"}:
            raise ValueError("Expected only action, target, and payload")
        target, payload = value.get("target"), value.get("payload", {})
        if target is not None and not isinstance(target, str):
            raise ValueError("target must be a node ID or null")
        if not isinstance(payload, dict) or any(not isinstance(v, str) for v in payload.values()):
            raise ValueError("payload must map strings to strings")
        return cls(ActionKind(value["action"]), target, payload)

    def as_dict(self) -> dict[str, Any]:
        return {"action": self.kind.value, "target": self.target, "payload": self.payload}


@dataclass
class Node:
    id: str
    role: str
    task: str = ""
    parents: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    is_final: bool = False
    replay_safe: bool = True


@dataclass
class Graph:
    nodes: dict[str, Node]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Graph":
        nodes = [Node(**item) for item in value["nodes"]]
        if not nodes or len({n.id for n in nodes}) != len(nodes):
            raise ValueError("graph needs unique, nonempty nodes")
        graph = cls({node.id: node for node in nodes})
        graph.order()
        if sum(node.is_final for node in nodes) != 1:
            raise ValueError("graph needs exactly one final node")
        return graph

    def as_dict(self) -> dict[str, Any]:
        return {"nodes": [vars(node).copy() for node in self.nodes.values()]}

    def order(self) -> list[str]:
        deps = {nid: set(node.parents) for nid, node in self.nodes.items()}
        for nid, parents in deps.items():
            if nid in parents or not parents <= self.nodes.keys():
                raise ValueError(f"invalid dependencies for {nid}")
        return list(TopologicalSorter(deps).static_order())

    def children(self, nid: str) -> list[str]:
        return [node.id for node in self.nodes.values() if nid in node.parents]

    def descendants(self, nid: str) -> set[str]:
        seen, pending = set(), self.children(nid)
        while pending:
            child = pending.pop()
            if child not in seen:
                seen.add(child)
                pending.extend(self.children(child))
        return seen

    def final_id(self) -> str:
        return next(node.id for node in self.nodes.values() if node.is_final)


@dataclass
class GateState:
    task: str
    graph: Graph
    current_node: str
    outputs: dict[str, str]
    prompts: dict[str, str]
    extra_calls_left: int = 4
    interventions_left: int = 1


def legal_actions(state: GateState) -> dict[ActionKind, list[str]]:
    legal: dict[ActionKind, list[str]] = {ActionKind.KEEP: []}
    if state.interventions_left <= 0:
        return legal
    current = state.current_node
    node = state.graph.nodes[current]
    children = [nid for nid in state.graph.children(current) if nid not in state.outputs]
    if state.extra_calls_left > 0 and node.replay_safe:
        legal[ActionKind.RETRY_CURRENT] = [current]
    if children:
        legal[ActionKind.ANNOTATE_CURRENT] = [current]
        legal[ActionKind.GUIDE_CHILD] = children
        if state.extra_calls_left > 0:
            legal[ActionKind.INSERT_SUBTASK] = children
    parents = []
    for parent in node.parents:
        affected = {parent, *state.graph.descendants(parent)} & state.outputs.keys()
        if len(affected) <= state.extra_calls_left and all(
            state.graph.nodes[nid].replay_safe for nid in affected
        ):
            parents.append(parent)
    if parents:
        legal[ActionKind.BACKTRACK_PARENT] = parents
    return legal


def public_editor_input(state: GateState) -> dict[str, Any]:
    """Expose the full plan and only outputs already observed at this gate."""
    return {
        "question": state.task,
        "graph": state.graph.as_dict(),
        "current_node": state.current_node,
        "observed_outputs": state.outputs.copy(),
        "legal_actions": {k.value: v for k, v in legal_actions(state).items()},
        "remaining_interventions": state.interventions_left,
        "remaining_extra_node_calls": state.extra_calls_left,
    }


def validate_action(
    state: GateState, action: Action, token_count: Callable[[str], int] | None = None
) -> None:
    legal = legal_actions(state)
    if action.kind not in legal:
        raise ValueError(f"illegal action: {action.kind.value}")
    if action.kind is ActionKind.KEEP:
        if action.target is not None or action.payload:
            raise ValueError("KEEP takes no target or payload")
        return
    if action.target not in legal[action.kind]:
        raise ValueError("target is not legal at this gate")
    expected = {
        ActionKind.RETRY_CURRENT: set(),
        ActionKind.BACKTRACK_PARENT: set(),
        ActionKind.ANNOTATE_CURRENT: {"error_type", "target_span", "diagnosis", "downstream_check"},
        ActionKind.GUIDE_CHILD: {"guidance_type", "focus"},
        ActionKind.INSERT_SUBTASK: {"mode", "task"},
    }[action.kind]
    if set(action.payload) != expected:
        raise ValueError(f"{action.kind.value} requires payload fields {sorted(expected)}")
    count = token_count or (lambda text: len(text.split()))
    payload = action.payload
    if action.kind is ActionKind.ANNOTATE_CURRENT:
        if payload["error_type"] not in {
            "LOGIC", "CALCULATION", "EVIDENCE", "CONSTRAINT", "INCOMPLETE",
            "CONFLICT", "LOW_CONFIDENCE",
        }:
            raise ValueError("invalid annotation error type")
        if not payload["target_span"] or payload["target_span"] not in state.outputs[state.current_node]:
            raise ValueError("target_span must be copied from the current output")
        if not payload["diagnosis"].strip() or not payload["downstream_check"].strip():
            raise ValueError("annotation diagnosis and check must be nonempty")
        if any(count(payload[key]) > limit for key, limit in (
            ("target_span", 24), ("diagnosis", 48), ("downstream_check", 24)
        )) or count(" ".join(payload.values())) > min(
            96, max(24, int(0.35 * count(state.outputs[state.current_node])))
        ):
            raise ValueError("annotation exceeds its token limit")
    if action.kind is ActionKind.GUIDE_CHILD:
        if payload["guidance_type"] not in {
            "VERIFY_UPSTREAM", "RECOMPUTE", "CHECK_CONSTRAINT", "RESOLVE_CONFLICT"
        } or not payload["focus"].strip() or count(payload["focus"]) > 32:
            raise ValueError("invalid child guidance")
    if action.kind is ActionKind.INSERT_SUBTASK:
        if payload["mode"] not in {"VERIFY", "BRIDGE", "SUPPLEMENT"}:
            raise ValueError("invalid inserted subtask mode")
        if payload["mode"] != "VERIFY" and not payload["task"].strip():
            raise ValueError("BRIDGE and SUPPLEMENT require a task")
        if count(payload["task"]) > 48:
            raise ValueError("inserted subtask is too long")
