"""Unified quantification API — single canonical entry point for all experiments.

This module provides the canonical implementations that all experiment scripts
should call instead of reimplementing K, G, delta, U_i, or impact computation.

Usage:
    from quantification.unified import quantify_from_j_state, quantify_one_trace, compute_unified_u_i, EPSILON

    # Option A: Full pipeline (J-scoring + everything)
    result = quantify_one_trace(
        trace_path="path/to/trace.json",
        threshold=0.5,
        ground_truth="42",
        judge_score=1.0,
        engine=engine,
        nli_client=nli_client,
    )

    # Option B: Phase 2 only (given a pre-computed JPhaseState)
    result = quantify_from_j_state(
        j_state=j_state,
        trace_path="path/to/trace.json",
        judge_score=1.0,
        threshold=0.5,
        nli_client=nli_client,
    )
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

# Ensure repo root is importable
if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quantification.config import DEFAULT_INTER_NODE_MODE, QuantificationConfig
from quantification.graph.data_nodes import is_data_node
from quantification.schemas import JPhaseState
from quantification.metrics import QuantificationEngine
from quantification.adapters.trace_adapter import _find_forward_code, load_reasoning_graph_from_trace
from quantification.communication.nli import NLIClient, get_nli_client, reset_nli_client
from quantification.utility.utility import compute_unified_u_i, compute_delta

EPSILON = 1e-8

__all__ = [
    "quantify_from_j_state",
    "quantify_one_trace",
    "summarize_results",
    "compute_unified_u_i",
    "compute_delta",
    "EPSILON",
]


# ================================================================
# Phase 2: Canonical NLI R_ij + K + G + U_i + delta + impact
# ================================================================

def _build_node_outputs_and_edges(
    trace: dict[str, Any], nodes: list[str]
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Extract node outputs and edges from a MAS trace JSON."""
    node_outputs: dict[str, str] = {}
    workflow = trace.get("workflow_state") or {}
    node_set = set(nodes)
    for agent in (trace.get("sub_agents") or []):
        aid = agent.get("agent_name") or agent.get("role_name") or ""
        if aid in node_set:
            node_outputs[aid] = str(agent.get("output") or workflow.get(agent.get("output_key", "")) or "")

    edges: list[tuple[str, str]] = []
    for stage in (trace.get("build_stage_traces") or []):
        for dep in (stage.get("parsed_json") or {}).get("dependencies") or []:
            f_, t_ = dep.get("from") or dep.get("from_id"), dep.get("to") or dep.get("to_id")
            if f_ and t_ and str(f_) in node_set and str(t_) in node_set:
                edges.append((str(f_), str(t_)))

    # XML orchestra traces: prefer graph_edges directly saved in trace JSON
    if not edges and node_set:
        graph_edges = trace.get("graph_edges") or []
        if graph_edges:
            for e in graph_edges:
                f_, t_ = e.get("from", ""), e.get("to", "")
                if f_ in node_set and t_ in node_set:
                    edges.append((f_, t_))
        else:
            # Fallback: parse edges from forward_code.py
            trace_path = trace.get("_trace_path", "")
            if trace_path:
                from quantification.adapters.trace_adapter import _find_forward_code
                fc = _find_forward_code(Path(trace_path))
                if fc and fc.is_file():
                    code = fc.read_text(encoding="utf-8")
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
                        dep_map[target] = node_set - {target}
                    for m in re.finditer(r'upstream_(\w+)\["(\w+)"\]', code):
                        target = m.group(1)
                        dep = m.group(2)
                        if target not in dep_map:
                            dep_map[target] = set()
                        dep_map[target].add(dep)
                    for target, deps in dep_map.items():
                        if target in node_set:
                            for dep in deps:
                                if dep in node_set:
                                    edges.append((dep, target))

    return node_outputs, edges


def _compute_r_matrix(
    edges: list[tuple[str, str]],
    node_outputs: dict[str, str],
    nli_client: NLIClient,
) -> dict[tuple[str, str], float]:
    """Compute NLI R_ij for all edges."""
    r_matrix: dict[tuple[str, str], float] = {}
    if not edges:
        return r_matrix
    p_texts = [node_outputs.get(p, "") for p, _ in edges]
    c_texts = [node_outputs.get(c, "") for _, c in edges]
    scores = nli_client.entailment_probabilities(list(zip(p_texts, c_texts)))
    for e, s in zip(edges, scores):
        r_matrix[e] = float(s)
    return r_matrix


def _topo_sort_and_adj(
    nodes: list[str], edges: list[tuple[str, str]]
) -> tuple[dict[str, list[str]], list[str]]:
    """Kahn's algorithm topological sort."""
    adj: dict[str, list[str]] = {nid: [] for nid in nodes}
    for src, dst in edges:
        if src in adj:
            adj[src].append(dst)

    in_deg = {nid: 0 for nid in nodes}
    for src, dst in edges:
        if dst in in_deg:
            in_deg[dst] += 1

    q = deque([n for n in nodes if in_deg.get(n, 0) == 0])
    topo: list[str] = []
    while q:
        n = q.popleft()
        topo.append(n)
        for nxt in adj.get(n, []):
            in_deg[nxt] -= 1
            if in_deg[nxt] == 0:
                q.append(nxt)
    return adj, topo


def _compute_g_values(
    adj: dict[str, list[str]], topo: list[str],
    use_depth_fair: bool = False, gamma: float = 0.5,
) -> dict[str, float]:
    """
    G_j = propagation scope of node j.

    Simple G (default, use_depth_fair=False):
      G_j = |Descendants(j)| / max_descendants
      Terminal nodes (no children) get G=0.0 — they have no downstream influence.

    Depth-fair G (use_depth_fair=True, Section 3.6 of the improvement doc):
      T_j = sum_{k in Desc(j)} gamma^{d(j,k)-1}
      G_j = T_j / (T_max + epsilon)
      Terminal nodes get G=0.0.
      Uses gamma-decay weighting so that closer descendants contribute more,
      reducing the depth bias where early nodes dominate purely by position.
    """
    if not use_depth_fair:
        desc_sets: dict[str, set[str]] = {}
        for n in reversed(topo):
            s: set[str] = set()
            for child in adj.get(n, []):
                s.add(child)
                s.update(desc_sets.get(child, set()))
            desc_sets[n] = s
        max_desc = max(len(s) for s in desc_sets.values()) if desc_sets else 1

        g_values: dict[str, float] = {}
        for n in topo:
            if not adj.get(n):
                g_values[n] = 0.0  # terminal: no downstream influence
            else:
                g_values[n] = len(desc_sets[n]) / max(max_desc, 1)
        return g_values

    # Depth-fair G: compute distances first, then gamma-decay weighted sum
    # BFS-based distance from each node to each descendant
    from collections import deque as dq
    dist: dict[tuple[str, str], int] = {}
    for n in topo:
        visited = {n: 0}
        q = dq([n])
        while q:
            cur = q.popleft()
            for child in adj.get(cur, []):
                if child not in visited:
                    visited[child] = visited[cur] + 1
                    dist[(n, child)] = visited[child]
                    q.append(child)

    t_values: dict[str, float] = {}
    for n in topo:
        total = 0.0
        for (src, dst), d in dist.items():
            if src == n and d >= 1:
                total += gamma ** (d - 1)
        t_values[n] = total

    t_max = max(t_values.values(), default=0.0)
    g_values: dict[str, float] = {}
    for n in topo:
        if not adj.get(n):
            g_values[n] = 0.0  # terminal: no downstream influence
        else:
            g_values[n] = t_values.get(n, 0.0) / (t_max + EPSILON) if t_max > 0 else 0.0
    return g_values


def _compute_k_values(
    adj: dict[str, list[str]],
    topo: list[str],
    r_matrix: dict[tuple[str, str], float],
    g_values: dict[str, float],
    nodes: list[str],
    inter_node_mode: str = DEFAULT_INTER_NODE_MODE,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Compute K using either the latest baseline-shifted formula or the previous
    canonical implementation retained for reproducibility.

    unified_baseline (default):
      K_i = 1 + sum_{j in Child(i)} R_ij * (1 + G_j)
      All nodes are included; terminal nodes naturally have K=1.

    legacy_canonical:
      K_i = sum_{j in Child(i)} R_ij * G_j, with terminal-child G overridden
      to 1 and terminal nodes assigned K=0.
    """
    if inter_node_mode not in {"unified_baseline", "legacy_canonical"}:
        raise ValueError(
            "quantification.unified supports inter_node_mode "
            f"'unified_baseline' or 'legacy_canonical'; got {inter_node_mode!r}"
        )

    k_raw: dict[str, float] = {}
    for n in topo:
        children = adj.get(n, [])
        if inter_node_mode == "unified_baseline":
            k_raw[n] = 1.0 + sum(
                r_matrix.get((n, child), 0.0)
                * (1.0 + g_values.get(child, 0.0))
                for child in children
            )
            continue

        if not children:
            k_raw[n] = 0.0
            continue
        total = 0.0
        for child in children:
            r_ij = r_matrix.get((n, child), 0.0)
            is_terminal_child = not adj.get(child)
            g_j = 1.0 if is_terminal_child else g_values.get(child, 0.0)
            total += r_ij * g_j
        k_raw[n] = total

    k_norm: dict[str, float] = {}
    for n in nodes:
        if inter_node_mode == "unified_baseline":
            k_norm[n] = k_raw.get(n, 1.0)
        elif not adj.get(n):
            k_norm[n] = 0.0  # terminal: no downstream communication
        else:
            k_norm[n] = k_raw.get(n, 0.0)  # no normalization
    return k_raw, k_norm


def _compute_node_metrics(
    nodes: list[str],
    j_baseline: float,
    j_by_node: dict[str, float],
    k_norm: dict[str, float],
    g_values: dict[str, float],
    adj: dict[str, list[str]],
    judge_score: float,
    threshold: float,
    j_by_chunk: dict[str, list[float]] | None = None,
    j_before: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Compute delta, U_i, impact for each node.

    In the latest mode, `nodes` includes terminal nodes; legacy mode filters them
    upstream for exact reproducibility.
    When j_by_chunk is provided, delta is computed per chunk and averaged,
    giving a more robust U_i. Otherwise falls back to single-delta.
    """
    prev_j = j_baseline
    node_data: list[dict[str, Any]] = []

    for nid in nodes:
        current_j = j_by_node.get(nid, prev_j)

        # DAG-aware delta: use j_before[nid] = J(ancestors) as baseline, not prev_j
        baseline_j = j_before.get(nid, prev_j) if j_before else prev_j

        if j_by_chunk and nid in j_by_chunk and j_by_chunk[nid]:
            chunk_js = j_by_chunk[nid]
            chunk_deltas = []
            chunk_prev = baseline_j
            for cj in chunk_js:
                d = compute_delta(chunk_prev, cj, j_baseline, EPSILON)
                chunk_deltas.append(d)
                chunk_prev = cj
            avg_delta = float(np.mean(chunk_deltas))
            delta = avg_delta
        else:
            delta = compute_delta(baseline_j, current_j, j_baseline, EPSILON)

        u_i = math.tanh(delta)
        kn = k_norm.get(nid, 0.0)
        impact = u_i * kn
        node_data.append({
            "id": nid, "delta": round(delta, 6), "U_i": round(u_i, 4),
            "K": round(kn, 4), "G": round(g_values.get(nid, 0.0), 4),
            "impact": round(impact, 4),
            "j_prefix": round(current_j, 4),
        })
        prev_j = current_j
    return node_data


def quantify_from_j_state(
    j_state: JPhaseState,
    trace_path: Path | str,
    judge_score: float,
    threshold: float,
    nli_client: NLIClient | None = None,
    use_depth_fair_g: bool = False,
    gamma: float = 0.5,
    inter_node_mode: str = DEFAULT_INTER_NODE_MODE,
) -> dict[str, Any] | None:
    """Given a pre-computed JPhaseState, run Phase 2 (NLI + K + G + U_i + delta + impact).

    This is the **canonical implementation** that all scripts should delegate to
    after computing J-scores.  It handles:

    - NLI R_ij computation (bge-m3-zeroshot-v2.0 entailment probability)
    - Raw G values (simple descendant ratio or depth-fair gamma-decay, terminal=0)
    - Latest default K: K_i = 1 + sum_j R_ij * (1 + G_j)
    - Legacy canonical K remains selectable for reproducing previous results
    - U_i: tanh(delta) for every quantified node
    - delta = (J_prev - J_curr) / |J_baseline|
    - impact = U_i * K_norm

    Args:
        use_depth_fair_g: If True, use gamma-decay weighted G (Section 3.6).
                          Default False uses simple descendant count ratio.
        gamma: Decay factor for depth-fair G (only used if use_depth_fair_g=True).
        inter_node_mode: ``unified_baseline`` (default) or ``legacy_canonical``.
    """
    if nli_client is None:
        nli_client = get_nli_client()

    trace_path = Path(trace_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace["_trace_path"] = str(trace_path)  # for forward_code.py lookup

    j_baseline = j_state.j_baseline
    j_by_node = j_state.j_by_node
    nodes = j_state.execution_order
    j_by_chunk = j_state.j_by_chunk
    j_before = j_state.j_before if hasattr(j_state, 'j_before') else {}

    node_outputs, edges = _build_node_outputs_and_edges(trace, nodes)
    if not edges and inter_node_mode != "unified_baseline":
        return {"idx": 0, "error": "no_edges", "note": "could not parse dependency edges from trace"}
    r_matrix = _compute_r_matrix(edges, node_outputs, nli_client)
    adj, topo = _topo_sort_and_adj(nodes, edges)
    g_values = _compute_g_values(adj, topo, use_depth_fair=use_depth_fair_g, gamma=gamma)
    k_raw, k_norm = _compute_k_values(
        adj, topo, r_matrix, g_values, nodes, inter_node_mode=inter_node_mode
    )

    quantified_nodes = (
        list(nodes)
        if inter_node_mode == "unified_baseline"
        else [n for n in nodes if adj.get(n)]
    )
    node_data = _compute_node_metrics(
        quantified_nodes, j_baseline, j_by_node, k_norm, g_values, adj, judge_score, threshold,
        j_by_chunk=j_by_chunk, j_before=j_before,
    )

    edge_data = [{"from": s, "to": d, "R_ij": round(r, 4)} for (s, d), r in r_matrix.items()]

    stem = trace_path.stem
    idx = int(stem.split("__", 1)[0]) if "__" in stem else 0
    task_id = stem.split("__", 1)[1].rsplit("_t", 1)[0] if "__" in stem else str(idx)

    u_list = [nd["U_i"] for nd in node_data]
    k_list = [nd["K"] for nd in node_data]
    r_list = [e["R_ij"] for e in edge_data]
    impact_list = [nd["impact"] for nd in node_data]

    return {
        "idx": idx, "task_id": task_id,
        "prompt": trace.get("question", trace.get("prompt", ""))[:120],
        "judge_score": round(judge_score, 4),
        "threshold": threshold,
        "j_baseline": round(j_baseline, 4),
        "num_nodes": len(node_data), "num_edges": len(edge_data),
        "mean_U_i": round(float(np.mean(u_list)), 4) if u_list else 0.0,
        "mean_K": round(float(np.mean(k_list)), 4) if k_list else 0.0,
        "mean_R": round(float(np.mean(r_list)), 4) if r_list else 0.0,
        "mean_impact": round(float(np.mean(impact_list)), 4) if impact_list else 0.0,
        "pos_U": sum(1 for u in u_list if u > 0.01),
        "neg_U": sum(1 for u in u_list if u < -0.01),
        "nodes": node_data, "edges": edge_data,
        "g_type": "depth_fair" if use_depth_fair_g else "simple_descendants",
        "inter_node_mode": inter_node_mode,
        "k_formula": (
            "1 + sum_j R_ij * (1 + G_j)"
            if inter_node_mode == "unified_baseline"
            else "sum_j R_ij * G_j (legacy terminal handling)"
        ),
        "terminal_nodes_included": inter_node_mode == "unified_baseline",
    }


def quantify_one_trace(
    trace_path: Path | str,
    threshold: float,
    ground_truth: str,
    judge_score: float,
    engine: QuantificationEngine,
    nli_client: NLIClient | None = None,
    config: QuantificationConfig | None = None,
) -> dict[str, Any] | None:
    """Run full quantification pipeline (Phase 1 J-scoring + Phase 2).

    Convenience wrapper that runs J-scoring first, then delegates to
    quantify_from_j_state() for Phase 2.
    """
    if config is None:
        config = QuantificationConfig.from_env()

    trace_path = Path(trace_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))

    try:
        graph = load_reasoning_graph_from_trace(trace_path, ground_truth=ground_truth)
    except Exception as exc:
        return {"idx": 0, "error": f"load_graph: {exc}"}

    try:
        j_state = engine.compute_j_phase(graph)
    except Exception as exc:
        return {"idx": 0, "error": f"j_phase: {exc}"}

    return quantify_from_j_state(
        j_state,
        trace_path,
        judge_score,
        threshold,
        nli_client,
        use_depth_fair_g=config.use_depth_fair_g,
        gamma=config.gamma,
        inter_node_mode=config.inter_node_mode,
    )


def summarize_results(
    results: list[dict[str, Any]], label: str
) -> dict[str, Any]:
    """Aggregate per-trace results into dataset-level summary."""
    successes = [r for r in results if "error" not in r]
    if not successes:
        return {"label": label, "num_results": 0}

    all_nodes = [nd for r in successes for nd in r["nodes"]]
    all_u = [nd["U_i"] for nd in all_nodes]
    all_k = [nd["K"] for nd in all_nodes]
    all_r = [e["R_ij"] for r in successes for e in r["edges"]]
    all_d = [nd["delta"] for nd in all_nodes]
    all_impact = [nd["impact"] for nd in all_nodes]

    return {
        "label": label,
        "num_results": len(successes),
        "total_nodes": len(all_nodes),
        "total_edges": len(all_r),
        "mean_U_i": round(float(np.mean(all_u)), 4) if all_u else 0.0,
        "mean_delta": round(float(np.mean(all_d)), 4) if all_d else 0.0,
        "mean_K": round(float(np.mean(all_k)), 4) if all_k else 0.0,
        "mean_R_ij": round(float(np.mean(all_r)), 4) if all_r else 0.0,
        "mean_impact": round(float(np.mean(all_impact)), 4) if all_impact else 0.0,
        "inter_node_modes": sorted({
            r.get("inter_node_mode", "unknown") for r in successes
        }),
        "pos_U": sum(1 for u in all_u if u > 0.01),
        "neg_U": sum(1 for u in all_u if u < -0.01),
        "pos_delta": sum(1 for d in all_d if d > 0),
        "neg_delta": sum(1 for d in all_d if d <= 0),
        "results": successes,
    }
