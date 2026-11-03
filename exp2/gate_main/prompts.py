"""Editor prompt used for task-level, multi-gate action generation."""

from __future__ import annotations

import json

from .editor import GateState, public_editor_input


EDITOR_SYSTEM = """You are the runtime editor for a multi-agent system.
At each gate, inspect only the visible execution prefix and select one legal action.
First check whether the current node fulfilled its role, task, constraints, and,
for a final node, the requested global deliverable. If its own deliverable is
clearly incomplete and RETRY_CURRENT is legal, choose RETRY_CURRENT. If the
failure comes from a direct parent handoff, consider BACKTRACK_PARENT. Otherwise
prefer KEEP unless a specific legal repair has positive expected net value.
Do not solve the task yourself or invent hidden outputs or a reference answer.
Return exactly one JSON object:
{"action":"ACTION","target":null,"payload":{}}
Use null only for KEEP; other actions require a legal node ID as target."""


ACTION_GUIDE = """Use only the action-target pairs listed in legal_actions.
- KEEP: preserve the current output and move to the next gate.
- RETRY_CURRENT: rerun the current node with the recorded prompt and context.
- BACKTRACK_PARENT: rerun a direct parent and its affected executed descendants.
- ANNOTATE_CURRENT: retain the raw output and append a bounded local diagnosis.
- GUIDE_CHILD: add a bounded instruction to an unexecuted direct child.
- INSERT_SUBTASK: insert one VERIFY, BRIDGE, or SUPPLEMENT node before a child.

Payload fields:
- KEEP, RETRY_CURRENT, BACKTRACK_PARENT: empty payload.
- ANNOTATE_CURRENT: error_type, target_span, diagnosis, downstream_check.
  Copy target_span from the current raw output; do not supply a new answer.
  error_type must be LOGIC, CALCULATION, EVIDENCE, CONSTRAINT, INCOMPLETE,
  CONFLICT, or LOW_CONFIDENCE. Limits: target_span 24 tokens, diagnosis 48,
  downstream_check 24; the combined annotation is also bounded.
- GUIDE_CHILD: guidance_type and focus.
  guidance_type must be VERIFY_UPSTREAM, RECOMPUTE, CHECK_CONSTRAINT, or
  RESOLVE_CONFLICT. The focus is at most 32 tokens.
- INSERT_SUBTASK: mode and task.
  mode must be VERIFY, BRIDGE, or SUPPLEMENT; task is at most 48 tokens.
Return exactly one JSON object, with no extra fields."""


def _trimmed_observation(state: GateState, max_output_chars: int) -> dict:
    observation = public_editor_input(state)
    observation["observed_outputs"] = {
        nid: output[-max_output_chars:]
        for nid, output in observation["observed_outputs"].items()
    }
    return observation


def initial_messages(state: GateState, max_output_chars: int = 6000) -> list[dict[str, str]]:
    observation = _trimmed_observation(state, max_output_chars)
    current = state.graph.nodes[state.current_node]
    observation["current_milestone_contract"] = {
        "role": current.role,
        "task": current.task,
        "constraints": current.constraints,
        "is_final": current.is_final,
    }
    return [
        {"role": "system", "content": EDITOR_SYSTEM},
        {"role": "user", "content": (
            "Check the current milestone against its assigned work before choosing. "
            "Select the least disruptive legal action with positive expected benefit.\n\n"
            + ACTION_GUIDE + "\n\nOBSERVATION:\n"
            + json.dumps(observation, ensure_ascii=False, separators=(",", ":"))
        )},
    ]


def next_gate_message(state: GateState, max_output_chars: int = 6000) -> dict[str, str]:
    observation = _trimmed_observation(state, max_output_chars)
    current = state.graph.nodes[state.current_node]
    compact = {
        "current_node": state.current_node,
        "current_node_spec": vars(current).copy(),
        "current_output": observation["observed_outputs"][state.current_node],
        "legal_actions": observation["legal_actions"],
        "remaining_interventions": state.interventions_left,
        "remaining_extra_node_calls": state.extra_calls_left,
    }
    return {
        "role": "user",
        "content": "Next gate. Apply the same decision rules.\nOBSERVATION:\n"
        + json.dumps(compact, ensure_ascii=False, separators=(",", ":")),
    }
