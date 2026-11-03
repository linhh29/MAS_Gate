from __future__ import annotations
import json
from typing import Any, Optional

from .schema import AgentNode, GraphEdge, MASGraph, Topology


_CLASS_TEMPLATE = """from __future__ import annotations
from typing import Any
import json
from template.sub_agent import SubAgentRequest, build_workflow_subagent_prompt

class {class_name}:
    def __init__(self, text_call_fn, tool_call_fn=None, **kw):
        self.text_call_fn = text_call_fn
        self.tool_call_fn = tool_call_fn
{agent_initializations}

    async def forward_async(self, task_text: str) -> dict[str, Any]:
        state: dict[str, Any] = {{}}
        task_info = task_text
{forward_body}
        return state
"""


def _dataset_runtime_tips(dataset_name: str, *, is_final: bool) -> str:
    """Return dataset execution tips without changing the policy-generated graph or roles."""
    ds = (dataset_name or "").strip().lower()
    if ds in {"aime", "hlemath", "hle"}:
        tips = (
            "AIME RUNTIME TIPS: Work with exact integer/rational forms rather than decimal "
            "approximations unless the task explicitly asks for one. Track the requested "
            "quantity, domain and endpoint restrictions, and divisibility/modular conditions. "
            "For counting, check overlaps, symmetry, duplicate cases, and omitted cases. Check "
            "for extraneous roots introduced by squaring, division, or other transformations. "
            "Treat upstream results as working notes and independently verify the step assigned "
            "by ROLE INSTRUCTION."
        )
        if is_final:
            tips += (
                " Before answering, substitute or otherwise sanity-check the resulting value. "
                "The final non-empty line must contain exactly one boxed integer from 0 through "
                "999, in the form \\boxed{N}."
            )
        return tips
    if ds in {"gpqa", "gpqa_diamond"}:
        tips = (
            "GPQA RUNTIME TIPS: Keep the original A/B/C/D mapping fixed; whenever upstream "
            "work names a letter, bind it back to the full option text before using it. Niche "
            "questions often hinge on one qualifier, sign, unit, definition, stereochemical "
            "label, or exception, so compare the decisive claim against the exact wording of "
            "each plausible option. Treat upstream confidence or consensus as non-evidence and "
            "execute the step assigned by ROLE INSTRUCTION."
        )
        if is_final:
            tips += (
                " Re-read the selected option text against the question, then output exactly "
                "one boxed uppercase option letter A, B, C, or D on the final non-empty line."
            )
        return tips
    return ""


def _generate_agent_init(node: AgentNode, dataset_name: str = "") -> str:
    user_prompt = (
        "Read GLOBAL TASK CONTEXT for the full task, constraints, and required output format. "
        "Use UPSTREAM DEPENDENCIES DATA when present. Execute only the sub-task described in "
        "ROLE INSTRUCTION."
    )
    dataset_tips = _dataset_runtime_tips(dataset_name, is_final=node.is_final)
    if dataset_tips:
        user_prompt = f"{user_prompt}\n\n{dataset_tips}"
    execution_mode = node.execution_mode or "llm_only"
    role_text = node.role
    if node.constraints:
        constraints_text = "\n".join(f"- {c}" for c in node.constraints)
        role_text = f"{role_text}\n\n--- CONSTRAINTS ---\n{constraints_text}"
    return f"""        # Agent: {node.id}
        self.{node.id} = SubAgentRequest(
            role_name={repr(node.id)},
            role_instruction={repr(role_text)},
            user_prompt={repr(user_prompt)},
            tool_context={{"execution_mode": {repr(execution_mode)}}}
        )"""


def _generate_sequential_forward(nodes: list[AgentNode]) -> str:
    lines = ['        upstream_outputs: dict[str, str] = {}']
    lines.append("")
    for i, node in enumerate(nodes):
        lines.append(f"        # Agent {node.id}")
        if i == 0:
            lines.append(
                f"        prompt_{node.id} = build_workflow_subagent_prompt("
                f"self.{node.id}.role_instruction, self.{node.id}.user_prompt, task_info)"
            )
        else:
            lines.append(
                f"        prompt_{node.id} = build_workflow_subagent_prompt("
                f"self.{node.id}.role_instruction, self.{node.id}.user_prompt, "
                f"task_info, upstream_outputs)"
            )
        lines.append(
            f"        result_{node.id} = await self.{node.id}.execute_async("
            f"text_call_fn=self.text_call_fn, tool_call_fn=self.tool_call_fn, "
            f"prompt_override=prompt_{node.id})"
        )
        lines.append(f'        state["out_{node.id}"] = result_{node.id}.text')
        lines.append(f'        state["usage_{node.id}"] = dict(result_{node.id}.usage)')
        lines.append(
            f'        state["usage_{node.id}"]["cost_usd"] = '
            f'result_{node.id}.usage.get("estimated_cost_usd", result_{node.id}.usage.get("cost_usd", 0.0))'
        )
        lines.append(f'        upstream_outputs["{node.id}"] = result_{node.id}.text')
        lines.append("")

    final = nodes[-1]
    for n in nodes:
        if n.is_final:
            final = n
            break
    lines.append(f'        state["final_output"] = state["out_{final.id}"]')
    lines.append(f'        state["usage_final"] = state["usage_{final.id}"]')
    return "\n".join(lines)


def _generate_dag_forward(
    nodes: list[AgentNode],
    edges: list[GraphEdge],
    *,
    scope_final_to_direct_dependencies: bool = False,
) -> str:
    in_degree: dict[str, int] = {n.id: 0 for n in nodes}
    adj: dict[str, list[str]] = {n.id: [] for n in nodes}
    for e in edges:
        in_degree[e.t] = in_degree.get(e.t, 0) + 1
        adj.setdefault(e.f, []).append(e.t)
    queue = [nid for nid, d in in_degree.items() if d == 0]
    order: list[str] = []
    while queue:
        nid = queue.pop(0)
        order.append(nid)
        for to_id in adj.get(nid, []):
            in_degree[to_id] -= 1
            if in_degree[to_id] == 0:
                queue.append(to_id)
    for n in nodes:
        if n.id not in order:
            order.append(n.id)

    node_map = {n.id: n for n in nodes}
    lines = ['        upstream_outputs: dict[str, str] = {}']
    lines.append("")
    for nid in order:
        node = node_map[nid]
        lines.append(f"        # Agent {node.id}")
        if node.is_final and not scope_final_to_direct_dependencies:
            lines.append(f"        upstream_{node.id} = dict(upstream_outputs)")
            lines.append(
                f"        prompt_{node.id} = build_workflow_subagent_prompt("
                f"self.{node.id}.role_instruction, self.{node.id}.user_prompt, "
                f"task_info, upstream_{node.id})"
            )
        else:
            lines.append(f"        upstream_{node.id} = {{}}")
            has_upstream = False
            for e in edges:
                if e.t == node.id:
                    lines.append(
                        f'        upstream_{node.id}["{e.f}"] = '
                        f'state.get("out_{e.f}", "")'
                    )
                    has_upstream = True
            if has_upstream:
                lines.append(
                    f"        prompt_{node.id} = build_workflow_subagent_prompt("
                    f"self.{node.id}.role_instruction, self.{node.id}.user_prompt, "
                    f"task_info, upstream_{node.id})"
                )
            else:
                lines.append(
                    f"        prompt_{node.id} = build_workflow_subagent_prompt("
                    f"self.{node.id}.role_instruction, self.{node.id}.user_prompt, "
                    f"task_info)"
                )
        lines.append(
            f"        result_{node.id} = await self.{node.id}.execute_async("
            f"text_call_fn=self.text_call_fn, tool_call_fn=self.tool_call_fn, "
            f"prompt_override=prompt_{node.id})"
        )
        lines.append(f'        state["out_{node.id}"] = result_{node.id}.text')
        lines.append(f'        state["usage_{node.id}"] = dict(result_{node.id}.usage)')
        lines.append(
            f'        state["usage_{node.id}"]["cost_usd"] = '
            f'result_{node.id}.usage.get("estimated_cost_usd", result_{node.id}.usage.get("cost_usd", 0.0))'
        )
        lines.append(f'        upstream_outputs["{node.id}"] = result_{node.id}.text')
        lines.append("")

    final_node = None
    for n in nodes:
        if n.is_final:
            final_node = n
            break
    if not final_node:
        final_node = node_map[order[-1]]
    lines.append(f'        state["final_output"] = state["out_{final_node.id}"]')
    lines.append(f'        state["usage_final"] = state["usage_{final_node.id}"]')
    return "\n".join(lines)


def compile_graph_to_code(
    graph: MASGraph, class_name: str = "MASWorkflow", dataset_name: str = ""
) -> str:
    agent_inits = [_generate_agent_init(node, dataset_name) for node in graph.nodes]
    agent_init_block = "\n\n".join(agent_inits)

    if graph.topology == Topology.SEQUENTIAL or not graph.edges:
        forward_body = _generate_sequential_forward(graph.nodes)
    else:
        strict_dependency_scope = (dataset_name or "").strip().lower() in {
            "gpqa", "gpqa_diamond"
        }
        forward_body = _generate_dag_forward(
            graph.nodes,
            graph.edges,
            scope_final_to_direct_dependencies=strict_dependency_scope,
        )

    return _CLASS_TEMPLATE.format(
        class_name=class_name,
        agent_initializations=agent_init_block,
        forward_body=forward_body,
    )


def compile_and_validate(
    graph: MASGraph, class_name: str = "MASWorkflow", dataset_name: str = ""
) -> tuple[str, bool, Optional[str]]:
    code = compile_graph_to_code(graph, class_name, dataset_name=dataset_name)
    try:
        import ast

        ast.parse(code)
        return code, True, None
    except SyntaxError as e:
        return code, False, str(e)