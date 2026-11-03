"""Execute a frozen MAS and replay only nodes affected by one Editor action.

The bundled worker supports text-only nodes. An interactive benchmark must
provide a worker that restores its environment before replay-dependent actions.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Protocol

from .editor import Action, ActionKind, GateState, Graph, Node, validate_action


class Worker(Protocol):
    def run(
        self, node: Node, task: str, upstream: dict[str, str],
        guidance: str = "", fixed_prompt: str | None = None,
    ) -> tuple[str, str]:
        """Return (node output, prompt sent to the frozen sub-agent)."""


@dataclass
class Trace:
    task: str
    graph: Graph
    outputs: dict[str, str]
    prompts: dict[str, str]
    extra_calls: int = 0

    def final_output(self) -> str:
        return self.outputs[self.graph.final_id()]

    def gate(self, index: int, extra_calls_left: int = 4) -> GateState:
        order = self.graph.order()
        if not 0 <= index < len(order):
            raise IndexError(index)
        visible = set(order[: index + 1])
        if not visible <= self.outputs.keys():
            raise ValueError("trace is missing an executed prefix")
        return GateState(
            task=self.task,
            graph=copy.deepcopy(self.graph),
            current_node=order[index],
            outputs={k: v for k, v in self.outputs.items() if k in visible},
            prompts={k: v for k, v in self.prompts.items() if k in visible},
            extra_calls_left=extra_calls_left,
        )


def _execute(
    worker: Worker, graph: Graph, task: str, outputs: dict[str, str],
    prompts: dict[str, str], nid: str, guidance: str = "",
    fixed_prompt: str | None = None,
) -> None:
    node = graph.nodes[nid]
    upstream = {parent: outputs[parent] for parent in node.parents}
    output, prompt = worker.run(node, task, upstream, guidance, fixed_prompt)
    if not isinstance(output, str) or not output.strip():
        raise RuntimeError(f"node {nid} produced no output")
    outputs[nid], prompts[nid] = output, prompt


def execute_reference(task: str, graph: Graph, worker: Worker) -> Trace:
    """Cache one no-intervention execution for paired comparisons."""
    graph = copy.deepcopy(graph)
    outputs: dict[str, str] = {}
    prompts: dict[str, str] = {}
    for nid in graph.order():
        _execute(worker, graph, task, outputs, prompts, nid)
    return Trace(task, graph, outputs, prompts)


def execute_with_editor(
    task: str, graph: Graph, worker: Worker,
    decide: Callable[[GateState], Action], *, extra_calls: int = 4,
) -> Trace:
    """Run a task live; expose only each completed prefix to the Editor."""
    graph = copy.deepcopy(graph)
    outputs: dict[str, str] = {}
    prompts: dict[str, str] = {}
    for nid in graph.order():
        _execute(worker, graph, task, outputs, prompts, nid)
        state = GateState(
            task=task, graph=graph, current_node=nid,
            outputs=outputs.copy(), prompts=prompts.copy(),
            extra_calls_left=extra_calls,
        )
        action = decide(state)
        validate_action(state, action)
        if action.kind is not ActionKind.KEEP:
            return execute_edit(state, action, worker)
    return Trace(task, graph, outputs, prompts)


def execute_edit(state: GateState, action: Action, worker: Worker) -> Trace:
    """Continue from the observed prefix after applying a legal action."""
    validate_action(state, action)
    graph = copy.deepcopy(state.graph)
    outputs, prompts = state.outputs.copy(), state.prompts.copy()
    current = state.current_node
    extra = 0
    guidance: dict[str, str] = {}

    if action.kind in {ActionKind.RETRY_CURRENT, ActionKind.BACKTRACK_PARENT}:
        restore = getattr(worker, "prepare_replay", None)
        if callable(restore):
            restore(state, action)

    if action.kind is ActionKind.RETRY_CURRENT:
        _execute(worker, graph, state.task, outputs, prompts, current,
                 fixed_prompt=prompts[current])
        extra = 1
    elif action.kind is ActionKind.BACKTRACK_PARENT:
        assert action.target is not None
        affected = {action.target, *graph.descendants(action.target)} & outputs.keys()
        for nid in affected:
            del outputs[nid]
            prompts.pop(nid, None)
        for nid in graph.order():
            if nid in affected:
                _execute(worker, graph, state.task, outputs, prompts, nid)
                extra += 1
    elif action.kind is ActionKind.ANNOTATE_CURRENT:
        p = action.payload
        outputs[current] += (
            "\n\n[EDITOR ANNOTATION]\n"
            f"Error type: {p['error_type']}\nTarget span: {p['target_span']}\n"
            f"Diagnosis: {p['diagnosis']}\nDownstream check: {p['downstream_check']}\n"
            "[/EDITOR ANNOTATION]"
        )
    elif action.kind is ActionKind.GUIDE_CHILD:
        assert action.target is not None
        guidance[action.target] = (
            f"{action.payload['guidance_type']}: {action.payload['focus']}"
        )
    elif action.kind is ActionKind.INSERT_SUBTASK:
        assert action.target is not None
        child = graph.nodes[action.target]
        mode, task = action.payload["mode"], action.payload["task"]
        nid = f"runtime_insert_{current}_{child.id}"
        if nid in graph.nodes:
            raise ValueError("inserted node ID collides with the frozen graph")
        graph.nodes[nid] = Node(
            id=nid,
            role=(f"Runtime {mode} subtask: {task or 'Verify the upstream output.'} "
                  f"Prepare a bounded handoff for {child.id}; do not solve the global task."),
            task=task or "Verify the upstream output and report uncertainty.",
            parents=[current],
        )
        child.parents = [nid if parent == current else parent for parent in child.parents]
        _execute(worker, graph, state.task, outputs, prompts, nid)
        outputs[nid] = (
            "ORIGINAL MILESTONE:\n" + outputs[current]
            + "\n\nINSERTED SUBTASK RESULT:\n" + outputs[nid]
        )
        extra = 1

    if extra > state.extra_calls_left:
        raise ValueError("extra node call budget exceeded")
    for nid in graph.order():
        if nid not in outputs:
            _execute(worker, graph, state.task, outputs, prompts, nid,
                     guidance=guidance.get(nid, ""))
    return Trace(state.task, graph, outputs, prompts, extra_calls=extra)
