from __future__ import annotations
import json
from typing import Any

from building.skill_mas.openai_async_client import AsyncOpenAIClient
from .schema import (
    AgentNode,
    AgentSpec,
    AgentSpecs,
    GraphEdge,
    MASGraph,
    SubTask,
    TaskDecomposition,
    Topology,
    EdgeType,
    extract_xml,
)
from .prompts import (
    STAGE1_SYSTEM,
    STAGE1_TEMPLATE,
    STAGE2_SYSTEM,
    STAGE2_TEMPLATE,
    get_dataset_requirement,
)


def _text(v: Any) -> str:
    if isinstance(v, dict):
        return str(v.get("#text", ""))
    if isinstance(v, str):
        return v
    return str(v) if v else ""


_VALID_EXECUTION_MODES = {"llm_only", "multi_turn_search"}
_BCP_DATASETS = {"bcp", "browsecomp", "browsecomp_plus", "mas_gate_bcp"}
_HOTPOTQA_DATASETS = {"hotpotqa", "hotpot_qa"}
_SEARCH_CAPABLE_DATASETS = _BCP_DATASETS | _HOTPOTQA_DATASETS
_SEARCH_TERMS = (
    "search",
    "retrieve",
    "retrieval",
    "evidence",
    "corpus",
    "bm25",
    "document",
    "source",
    "lookup",
    "research",
    "fact finding",
)


def _normalize_execution_mode(value: Any) -> str | None:
    mode = _text(value).strip().lower()
    return mode if mode in _VALID_EXECUTION_MODES else None


def _legacy_search_execution_mode(*, role: str, task: str, is_final: bool) -> str:
    """Infer a route only for old XML that did not emit execution_mode."""
    if is_final:
        return "llm_only"
    blob = f"{role} {task}".lower()
    return "multi_turn_search" if any(term in blob for term in _SEARCH_TERMS) else "llm_only"


async def run_stage1_decomposition(
    client: AsyncOpenAIClient, task_text: str, dataset: str = "aime"
) -> tuple[TaskDecomposition, str, dict[str, Any]]:
    req = get_dataset_requirement(dataset)
    user_prompt = STAGE1_TEMPLATE.format(task_text=task_text, dataset_requirement=req)
    text, usage = await client.generate(
        user_prompt=user_prompt, system_prompt=STAGE1_SYSTEM
    )
    data = extract_xml(text)
    td = data.get("task_decomposition", {})
    goal = _text(td.get("goal", ""))
    sts = td.get("sub_tasks", {})
    sub_tasks = []
    if isinstance(sts, dict):
        st_list = sts.get("sub_task", [])
        if isinstance(st_list, dict):
            st_list = [st_list]
        for st in st_list:
            idx = int(st.get("@idx", len(sub_tasks)))
            sub_tasks.append(SubTask(idx=idx, desc=_text(st)))
    return TaskDecomposition(goal=goal, sub_tasks=sub_tasks), text, usage


async def run_stage2_agent_specs(
    client: AsyncOpenAIClient,
    task_text: str,
    decomposition: TaskDecomposition,
    dataset: str = "aime",
) -> tuple[AgentSpecs, str, dict[str, Any]]:
    sts_xml = "\n".join(
        f'    <sub_task idx="{st.idx}">{st.desc}</sub_task>'
        for st in decomposition.sub_tasks
    )
    req = get_dataset_requirement(dataset)
    user_prompt = STAGE2_TEMPLATE.format(
        task_text=task_text,
        goal=decomposition.goal,
        sub_tasks_xml=sts_xml,
        dataset_requirement=req,
    )
    text, usage = await client.generate(
        user_prompt=user_prompt, system_prompt=STAGE2_SYSTEM
    )
    data = extract_xml(text)
    as_data = data.get("agent_specs", {})
    agents_data = as_data.get("agents", {})
    specs = []
    if isinstance(agents_data, dict):
        alist = agents_data.get("agent", [])
        if isinstance(alist, dict):
            alist = [alist]
        for a in alist:
            aid = a.get("@id", "")
            role = _text(a.get("role", ""))
            is_final = a.get("@is_final", "false") == "true"
            deps_str = _text(a.get("dependencies", ""))
            deps = (
                [d.strip() for d in deps_str.split(",") if d.strip()]
                if deps_str
                else []
            )
            st_idx = int(_text(a.get("sub_task_idx", "0")))
            execution_mode = _normalize_execution_mode(
                a.get("@execution_mode", a.get("execution_mode"))
            )
            specs.append(
                AgentSpec(
                    id=aid,
                    role=role,
                    sub_task_idx=st_idx,
                    dependencies=deps,
                    is_final=is_final,
                    execution_mode=execution_mode,
                )
            )
    return AgentSpecs(specs=specs), text, usage


def assemble_graph(
    decomposition: TaskDecomposition, specs: AgentSpecs, dataset: str = "aime"
) -> MASGraph:
    nodes = []
    edges = []
    normalized_dataset = (dataset or "").strip().lower()
    for spec in specs.specs:
        task_desc = ""
        if spec.sub_task_idx < len(decomposition.sub_tasks):
            task_desc = decomposition.sub_tasks[spec.sub_task_idx].desc
        else:
            task_desc = spec.role
        explicit_mode = _normalize_execution_mode(spec.execution_mode)
        if normalized_dataset in _SEARCH_CAPABLE_DATASETS:
            execution_mode = explicit_mode or _legacy_search_execution_mode(
                role=spec.role,
                task=task_desc,
                is_final=spec.is_final,
            )
        else:
            execution_mode = "llm_only"
        nodes.append(
            AgentNode(
                id=spec.id,
                role=spec.role,
                task=task_desc,
                goal=decomposition.goal,
                is_final=spec.is_final,
                execution_mode=execution_mode,
            )
        )
    if normalized_dataset in _SEARCH_CAPABLE_DATASETS and nodes and not any(
        node.execution_mode == "multi_turn_search" for node in nodes
    ):
        target = next(
            (
                node
                for node in nodes
                if not node.is_final
                and _legacy_search_execution_mode(
                    role=node.role,
                    task=node.task,
                    is_final=False,
                )
                == "multi_turn_search"
            ),
            None,
        )
        if target is None:
            target = next((node for node in nodes if not node.is_final), nodes[0])
        target.execution_mode = "multi_turn_search"
    for spec in specs.specs:
        for dep in spec.dependencies:
            edges.append(GraphEdge(f=dep, t=spec.id, type=EdgeType.DATA))
    if not edges and len(nodes) > 1:
        for i in range(len(nodes) - 1):
            edges.append(
                GraphEdge(f=nodes[i].id, t=nodes[i + 1].id, type=EdgeType.DATA)
            )
    topology = Topology.DAG if edges else Topology.SEQUENTIAL
    return MASGraph(nodes=nodes, edges=edges, topology=topology)


async def run_formalized_orchestra(
    client: AsyncOpenAIClient, task_text: str, dataset: str = "aime"
) -> tuple[MASGraph, dict[str, Any]]:
    """Run the full 2-stage XML orchestra + assembly pipeline."""
    trace: dict[str, Any] = {}

    decomp, stage1_text, stage1_usage = await run_stage1_decomposition(
        client, task_text, dataset
    )
    trace["stage1"] = {
        "goal": decomp.goal,
        "num_sub_tasks": len(decomp.sub_tasks),
        "raw_output": stage1_text,
        "usage": stage1_usage,
        "parsed": {
            "goal": decomp.goal,
            "sub_tasks": [
                {"idx": st.idx, "desc": st.desc} for st in decomp.sub_tasks
            ],
        },
    }

    specs, stage2_text, stage2_usage = await run_stage2_agent_specs(
        client, task_text, decomp, dataset
    )
    trace["stage2"] = {
        "num_agents": len(specs.specs),
        "agents": [
            {
                "id": s.id,
                "is_final": s.is_final,
                "role": s.role,
                "sub_task_idx": s.sub_task_idx,
                "dependencies": s.dependencies,
                "execution_mode": s.execution_mode,
            }
            for s in specs.specs
        ],
        "raw_output": stage2_text,
        "usage": stage2_usage,
    }

    graph = assemble_graph(decomp, specs, dataset)
    trace["stage3"] = {
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "topology": graph.topology.value,
        "execution_modes": {
            node.id: node.execution_mode for node in graph.nodes
        },
    }
    return graph, trace
