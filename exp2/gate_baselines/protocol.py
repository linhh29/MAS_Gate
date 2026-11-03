"""Task-agnostic interfaces shared by the baseline reference methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence


Message = dict[str, str]


class ChatClient(Protocol):
    """Small model interface used by every baseline."""

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
    ) -> str:
        """Return only the generated text."""


@dataclass(frozen=True)
class Node:
    node_id: str
    role: str
    task: str
    constraints: tuple[str, ...] = ()
    is_final: bool = False


@dataclass(frozen=True)
class Graph:
    nodes: tuple[Node, ...]
    edges: tuple[tuple[str, str], ...]

    def node(self, node_id: str) -> Node:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)

    def parents(self, node_id: str) -> tuple[str, ...]:
        return tuple(source for source, target in self.edges if target == node_id)

    def children(self, node_id: str) -> tuple[str, ...]:
        return tuple(target for source, target in self.edges if source == node_id)

    def descendants(self, node_id: str) -> set[str]:
        found: set[str] = set()
        frontier = list(self.children(node_id))
        while frontier:
            child = frontier.pop()
            if child not in found:
                found.add(child)
                frontier.extend(self.children(child))
        return found

    def topological_order(self) -> tuple[str, ...]:
        """Return a stable topological order and reject cyclic graphs."""
        node_ids = [node.node_id for node in self.nodes]
        known = set(node_ids)
        if len(known) != len(node_ids):
            raise ValueError("node identifiers must be unique")
        for source, target in self.edges:
            if source not in known or target not in known:
                raise ValueError(f"edge references an unknown node: {(source, target)}")

        indegree = {node_id: 0 for node_id in node_ids}
        for _, target in self.edges:
            indegree[target] += 1
        ready = [node_id for node_id in node_ids if indegree[node_id] == 0]
        ordered: list[str] = []
        while ready:
            current = ready.pop(0)
            ordered.append(current)
            for child in self.children(current):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if len(ordered) != len(node_ids):
            raise ValueError("graph must be acyclic")
        return tuple(ordered)

    def with_edges(self, edges: Sequence[tuple[str, str]]) -> Graph:
        graph = Graph(self.nodes, tuple(edges))
        graph.topological_order()
        return graph


@dataclass
class ExecutionState:
    """Only the public state available at a completed-node gate."""

    task: str
    graph: Graph
    current_node: str
    outputs: dict[str, str]
    interventions_left: int = 1
    extra_calls_left: int = 4


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
    payload: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EditorObservation:
    task: str
    current_node: str
    graph: Graph
    observed_outputs: Mapping[str, str]
    legal_actions: Mapping[str, tuple[str, ...]]
    interventions_left: int
    extra_calls_left: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "current_node": self.current_node,
            "graph": {
                "nodes": [
                    {
                        "id": node.node_id,
                        "role": node.role,
                        "task": node.task,
                        "constraints": list(node.constraints),
                        "is_final": node.is_final,
                    }
                    for node in self.graph.nodes
                ],
                "edges": [list(edge) for edge in self.graph.edges],
            },
            "observed_outputs": dict(self.observed_outputs),
            "legal_actions": {
                name: list(targets) for name, targets in self.legal_actions.items()
            },
            "budget": {
                "interventions_left": self.interventions_left,
                "extra_calls_left": self.extra_calls_left,
            },
        }


def editor_observation(state: ExecutionState) -> EditorObservation:
    """Expose the completed prefix and legal actions, but no correctness signal."""
    state.graph.node(state.current_node)
    if state.current_node not in state.outputs:
        raise ValueError("the current node must be completed before opening a gate")

    legal: dict[str, tuple[str, ...]] = {ActionKind.KEEP.value: ()}
    if state.interventions_left > 0 and state.extra_calls_left > 0:
        current = state.current_node
        legal[ActionKind.RETRY_CURRENT.value] = (current,)
        legal[ActionKind.ANNOTATE_CURRENT.value] = (current,)
        parents = state.graph.parents(current)
        if parents:
            legal[ActionKind.BACKTRACK_PARENT.value] = parents
        pending_children = tuple(
            child
            for child in state.graph.children(current)
            if child not in state.outputs
        )
        if pending_children:
            legal[ActionKind.GUIDE_CHILD.value] = pending_children
            legal[ActionKind.INSERT_SUBTASK.value] = pending_children

    ordered = state.graph.topological_order()
    observed = {
        node_id: state.outputs[node_id]
        for node_id in ordered
        if node_id in state.outputs
    }
    return EditorObservation(
        task=state.task,
        current_node=state.current_node,
        graph=state.graph,
        observed_outputs=observed,
        legal_actions=legal,
        interventions_left=state.interventions_left,
        extra_calls_left=state.extra_calls_left,
    )


def validate_action(observation: EditorObservation, action: Action) -> None:
    """Reject invented actions, targets, or incompatible payload shapes."""
    targets = observation.legal_actions.get(action.kind.value)
    if targets is None:
        raise ValueError(f"illegal action: {action.kind.value}")
    if targets:
        if action.target not in targets:
            raise ValueError(f"illegal target for {action.kind.value}: {action.target}")
    elif action.target is not None:
        raise ValueError(f"{action.kind.value} does not take a target")

    expected_payloads = {
        ActionKind.KEEP: set(),
        ActionKind.RETRY_CURRENT: set(),
        ActionKind.BACKTRACK_PARENT: set(),
        ActionKind.ANNOTATE_CURRENT: {
            "error_type",
            "target_span",
            "diagnosis",
            "downstream_check",
        },
        ActionKind.GUIDE_CHILD: {"guidance_type", "focus"},
        ActionKind.INSERT_SUBTASK: {"mode", "task"},
    }
    expected = expected_payloads[action.kind]
    if set(action.payload) != expected:
        raise ValueError(
            f"{action.kind.value} requires payload keys {sorted(expected)}"
        )
    if not all(isinstance(value, str) for value in action.payload.values()):
        raise ValueError("action payload values must be strings")
