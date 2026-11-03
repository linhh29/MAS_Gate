"""Skill-MAS sample_trace_json → ReasoningGraph adapter."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from quantification.graph.builder import build_reasoning_graph
from quantification.graph.structure import GraphStructure
from quantification.schemas import ReasoningGraph


def _extract_question(trace: dict[str, Any]) -> str:
    if trace.get("question"):
        return str(trace["question"])
    if trace.get("prompt"):
        return str(trace["prompt"])
    if trace.get("task_text_preview"):
        return str(trace["task_text_preview"])
    workflow = trace.get("workflow_state") or {}
    if workflow.get("task_info"):
        return str(workflow["task_info"])
    return ""


def _extract_ground_truth(trace: dict[str, Any], ground_truth_field: str | None = None) -> str:
    if ground_truth_field and trace.get(ground_truth_field):
        return str(trace[ground_truth_field])
    for key in ("article", "answer", "ground_truth"):
        if trace.get(key):
            return str(trace[key])
    return ""


def _extract_nodes(trace: dict[str, Any]) -> list[dict[str, str]]:
    nodes: list[dict[str, str]] = []
    outputs = trace.get("workflow_state") or {}

    sub_agents = trace.get("sub_agents") or []
    if sub_agents:
        for agent in sub_agents:
            agent_id = agent.get("agent_name") or agent.get("role_name") or ""
            if not agent_id:
                continue
            output = agent.get("output") or outputs.get(agent.get("output_key", "")) or ""
            nodes.append({"id": agent_id, "output": str(output)})
        return nodes

    normalized = trace.get("normalized_sub_agents") or []
    for agent in normalized:
        role = agent.get("role_name") or ""
        if not role:
            continue
        key = f"out_{role}"
        output = outputs.get(key, "")
        nodes.append({"id": role, "output": str(output)})
    return nodes


def _extract_edges(trace: dict[str, Any], known_node_ids: set[str]) -> list[dict[str, str]]:
    edges: list[dict[str, str]] = []
    build_traces = trace.get("build_stage_traces") or []
    for stage in build_traces:
        parsed = stage.get("parsed_json") or {}
        for dep in parsed.get("dependencies") or []:
            from_raw = dep.get("from") or dep.get("from_id")
            to_raw = dep.get("to") or dep.get("to_id")
            if not from_raw or not to_raw:
                continue
            from_id = _match_node_id(str(from_raw), known_node_ids) or str(from_raw)
            to_id = _match_node_id(str(to_raw), known_node_ids) or str(to_raw)
            if from_id not in known_node_ids or to_id not in known_node_ids:
                continue
            edges.append(
                {
                    "from": from_id,
                    "to": to_id,
                    "type": str(dep.get("type", "prerequisite")),
                }
            )

    if not edges and known_node_ids:
        # Prefer graph_edges from XML orchestra (directly saved in trace JSON)
        graph_edges = trace.get("graph_edges") or []
        if graph_edges:
            for e in graph_edges:
                f, t = e.get("from", ""), e.get("to", "")
                if f in known_node_ids and t in known_node_ids:
                    edges.append({"from": f, "to": t, "type": e.get("type", "data")})
        else:
            # Fallback: parse edges from forward_code.py (upstream_* = {...} patterns).
            import re
            forward_code = _find_forward_code(Path(trace.get("_trace_path", "")))
            if forward_code and forward_code.is_file():
                code = forward_code.read_text(encoding="utf-8")
                dep_map: dict[str, set[str]] = {}
                for m in re.finditer(r'upstream_(\w+)\s*=\s*\{([^}]*)\}', code):
                    target = m.group(1)
                    body = m.group(2)
                    deps = set()
                    for dep_m in re.finditer(r'"(\w+)"', body):
                        deps.add(dep_m.group(1))
                    dep_map[target] = deps
                for m in re.finditer(r'upstream_(\w+)\s*=\s*dict\(upstream_outputs\)', code):
                    target = m.group(1)
                    dep_map[target] = known_node_ids - {target}
                for m in re.finditer(r'upstream_(\w+)\["(\w+)"\]', code):
                    target = m.group(1)
                    dep = m.group(2)
                    if target not in dep_map:
                        dep_map[target] = set()
                    dep_map[target].add(dep)
                for target, deps in dep_map.items():
                    if target in known_node_ids:
                        for dep in deps:
                            if dep in known_node_ids:
                                edges.append({"from": dep, "to": target, "type": "data"})
    return edges


def _find_forward_code(trace_path: Path) -> Path | None:
    """Locate forward_code.py relative to sample_trace_json layout."""
    trace_dir = trace_path.parent
    if trace_dir.name == "sample_trace_json":
        process_trace_dir = trace_dir.parent
        sample_logs = process_trace_dir / "sample_logs"
        if sample_logs.is_dir():
            stem = trace_path.stem
            candidate = sample_logs / stem / "forward_code.py"
            if candidate.is_file():
                return candidate
            for sub in sample_logs.iterdir():
                if sub.is_dir() and stem in sub.name:
                    fc = sub / "forward_code.py"
                    if fc.is_file():
                        return fc
    return None


def _normalize_agent_attr(attr_name: str) -> str:
    """Map workflow attribute names to sub-agent role IDs."""
    if attr_name.startswith("agent_"):
        return attr_name[len("agent_") :]
    if attr_name.startswith("final_agent_"):
        return attr_name[len("final_agent_") :]
    return attr_name


def _match_node_id(candidate: str, known: set[str]) -> str | None:
    if candidate in known:
        return candidate
    for nid in known:
        if candidate.endswith(nid) or nid.endswith(candidate):
            return nid
    return None


def _parse_execution_order_from_forward_code(forward_code_path: Path) -> list[str]:
    text = forward_code_path.read_text(encoding="utf-8")
    order: list[str] = []
    seen: set[str] = set()

    patterns = [
        r"await\s+self\.(\w+)\.execute_async",
        r"self\.(\w+)\.execute_async",
        r"run_sub_agent\(\s*[\"'](\w+)[\"']",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            raw_name = match.group(1)
            if raw_name in ("text_call_fn", "tool_call_fn"):
                continue
            name = _normalize_agent_attr(raw_name)
            if name not in seen:
                seen.add(name)
                order.append(name)

    return order


def _resolve_execution_order(
    trace: dict[str, Any],
    trace_path: Path | None,
    node_ids: list[str],
    edges: list[dict[str, str]],
) -> list[str]:
    if trace_path is not None:
        fc = _find_forward_code(trace_path)
        if fc is not None:
            parsed = _parse_execution_order_from_forward_code(fc)
            known = set(node_ids)
            matched_from_fc: list[str] = []
            for name in parsed:
                matched = _match_node_id(name, known)
                if matched and matched not in matched_from_fc:
                    matched_from_fc.append(matched)
            if matched_from_fc:
                order = list(matched_from_fc)
                for nid in node_ids:
                    if nid not in order:
                        order.append(nid)
                return order

    if edges:
        temp_graph = build_reasoning_graph(
            question="",
            ground_truth="",
            nodes=[{"id": nid, "output": ""} for nid in node_ids],
            edges=edges,
        )
        return GraphStructure(temp_graph).topological_order()

    normalized = trace.get("normalized_sub_agents") or []
    if normalized:
        order = [a.get("role_name") for a in normalized if a.get("role_name")]
        known = set(node_ids)
        resolved = [nid for nid in order if nid in known]
        for nid in node_ids:
            if nid not in resolved:
                resolved.append(nid)
        return resolved

    sub_agents = trace.get("sub_agents") or []
    if sub_agents:
        order = [a.get("agent_name") for a in sub_agents if a.get("agent_name")]
        known = set(node_ids)
        resolved = [nid for nid in order if nid in known]
        for nid in node_ids:
            if nid not in resolved:
                resolved.append(nid)
        return resolved

    return list(node_ids)


def load_reasoning_graph_from_trace(
    trace_path: Path | str,
    ground_truth: str | None = None,
    ground_truth_field: str | None = None,
) -> ReasoningGraph:
    """Load a Skill-MAS trace JSON file and convert to ReasoningGraph."""
    path = Path(trace_path).resolve()
    with path.open(encoding="utf-8") as f:
        trace = json.load(f)
    trace["_trace_path"] = str(path)  # for forward_code.py lookup in _extract_edges

    question = _extract_question(trace)
    gt = ground_truth if ground_truth is not None else _extract_ground_truth(trace, ground_truth_field)
    nodes = _extract_nodes(trace)
    if not nodes:
        raise ValueError(f"No nodes found in trace: {path}")

    node_ids = [n["id"] for n in nodes]
    known = set(node_ids)
    edges = _extract_edges(trace, known)
    execution_order = _resolve_execution_order(trace, path, node_ids, edges)

    graph = build_reasoning_graph(
        question=question,
        ground_truth=gt,
        nodes=nodes,
        edges=edges if edges else None,
        execution_order=execution_order,
        metadata={
            "trace_path": str(path),
            "task_id": trace.get("task_id"),
            "schema_version": trace.get("schema_version"),
        },
    )
    return graph
