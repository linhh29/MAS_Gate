"""Propagation scope G_j and communication strength K_i."""

from __future__ import annotations

from quantification.config import QuantificationConfig
from quantification.graph.data_nodes import DATA_OUTPUT_NODE_ID, is_data_node
from quantification.graph.structure import GraphStructure
from quantification.schemas import EdgeUtilization


def compute_agent_propagation_scope(
    structure: GraphStructure,
    config: QuantificationConfig,
    *,
    use_depth_fair: bool = False,
) -> dict[str, float]:
    """Compute raw agent-only propagation scope G in [0, 1].

    Boundary data nodes are excluded from descendant counts.  Terminal agents
    therefore have G=0.  The latest inter-node formula applies the direct-uptake
    baseline as ``1 + G_j`` when computing K rather than changing this raw scope.
    """
    agent_ids = [nid for nid in structure.node_ids if not is_data_node(nid)]
    agent_set = set(agent_ids)
    order = [nid for nid in structure.topological_order() if nid in agent_set]
    adj = {
        nid: [child for child in structure.children(nid) if child in agent_set]
        for nid in agent_ids
    }

    g_values: dict[str, float] = {nid: 0.0 for nid in structure.node_ids}
    if not agent_ids:
        return g_values

    if not use_depth_fair:
        descendants: dict[str, set[str]] = {}
        for nid in reversed(order):
            reachable: set[str] = set()
            for child in adj[nid]:
                reachable.add(child)
                reachable.update(descendants.get(child, set()))
            descendants[nid] = reachable
        max_descendants = max((len(v) for v in descendants.values()), default=0)
        denominator = max(max_descendants, 1)
        for nid in agent_ids:
            g_values[nid] = len(descendants.get(nid, set())) / denominator
        return g_values

    # Optional gamma-decayed scope, also restricted to actual agent nodes.
    from collections import deque

    totals: dict[str, float] = {}
    for nid in agent_ids:
        distances = {nid: 0}
        queue = deque([nid])
        while queue:
            current = queue.popleft()
            for child in adj[current]:
                if child not in distances:
                    distances[child] = distances[current] + 1
                    queue.append(child)
        totals[nid] = sum(
            config.gamma ** (distance - 1)
            for target, distance in distances.items()
            if target != nid and distance >= 1
        )

    max_total = max(totals.values(), default=0.0)
    if max_total > 0:
        for nid in agent_ids:
            g_values[nid] = totals[nid] / (max_total + config.epsilon)
    return g_values


def compute_baseline_shifted_communication_strengths(
    structure: GraphStructure,
    r_matrix: dict[tuple[str, str], float],
    g_values: dict[str, float],
) -> tuple[dict[str, float], dict[str, float], dict[str, list[EdgeUtilization]]]:
    """Latest default: K_i = 1 + sum_j R_ij * (1 + G_j).

    The outer baseline preserves local contribution (terminal K=1).  The inner
    baseline represents direct uptake at a child, so a terminal child contributes
    R_ij even though its raw downstream propagation scope is G_j=0.
    """
    agent_ids = [nid for nid in structure.node_ids if not is_data_node(nid)]
    agent_set = set(agent_ids)
    k_values: dict[str, float] = {
        nid: (1.0 if nid in agent_set else 0.0) for nid in structure.node_ids
    }
    edge_utils: dict[str, list[EdgeUtilization]] = {
        nid: [] for nid in structure.node_ids
    }

    for parent_id in agent_ids:
        for child_id in structure.children(parent_id):
            if child_id not in agent_set:
                continue
            r_ij = r_matrix.get((parent_id, child_id), 0.0)
            k_values[parent_id] += r_ij * (1.0 + g_values.get(child_id, 0.0))
            edge_utils[parent_id].append(
                EdgeUtilization(from_id=parent_id, to_id=child_id, r_ij=r_ij)
            )

    # The latest K is an influence multiplier, not a [0, 1] normalization.
    return k_values, dict(k_values), edge_utils


def compute_legacy_canonical_communication_strengths(
    structure: GraphStructure,
    r_matrix: dict[tuple[str, str], float],
    g_values: dict[str, float],
) -> tuple[dict[str, float], dict[str, float], dict[str, list[EdgeUtilization]]]:
    """Previous unified.py behavior, retained for reproducibility."""
    agent_ids = [nid for nid in structure.node_ids if not is_data_node(nid)]
    agent_set = set(agent_ids)
    k_values: dict[str, float] = {nid: 0.0 for nid in structure.node_ids}
    edge_utils: dict[str, list[EdgeUtilization]] = {
        nid: [] for nid in structure.node_ids
    }
    for parent_id in agent_ids:
        agent_children = [
            child for child in structure.children(parent_id) if child in agent_set
        ]
        for child_id in agent_children:
            r_ij = r_matrix.get((parent_id, child_id), 0.0)
            child_is_terminal = not any(
                grandchild in agent_set for grandchild in structure.children(child_id)
            )
            effective_g = 1.0 if child_is_terminal else g_values.get(child_id, 0.0)
            k_values[parent_id] += r_ij * effective_g
            edge_utils[parent_id].append(
                EdgeUtilization(from_id=parent_id, to_id=child_id, r_ij=r_ij)
            )
    return k_values, dict(k_values), edge_utils


def compute_propagation_scope(
    structure: GraphStructure,
    config: QuantificationConfig,
) -> dict[str, float]:
    """
    G_j = T_j / (T^{max}_{global} + epsilon)
    where T_j = sum_{k in Desc(j)} gamma^{d(j,k)-1}

    Uses global normalization (not per-layer) so that deeper nodes with fewer
    descendants naturally receive lower G_j values.
    """
    sub_agent_ids = [nid for nid in structure.node_ids if not is_data_node(nid)]
    t_values: dict[str, float] = {}
    for node_id in sub_agent_ids:
        t_values[node_id] = structure.reachable_decay_sum(node_id, config.gamma)

    t_max_global = max(t_values.values(), default=0.0)

    g_values: dict[str, float] = {}
    for node_id in structure.node_ids:
        if node_id == DATA_OUTPUT_NODE_ID:
            g_values[node_id] = 1.0
        elif is_data_node(node_id):
            g_values[node_id] = 0.0
        else:
            g_values[node_id] = t_values[node_id] / (t_max_global + config.epsilon)
    return g_values


def compute_communication_strengths(
    structure: GraphStructure,
    r_matrix: dict[tuple[str, str], float],
    g_values: dict[str, float],
    config: QuantificationConfig,
) -> tuple[dict[str, float], dict[str, float], dict[str, list[EdgeUtilization]]]:
    """
    K_i = sum_{j in Child(i)} R_ij * G_j
    K̃_i = K_i / (max_v K_v + epsilon)
    """
    k_raw: dict[str, float] = {nid: 0.0 for nid in structure.node_ids}
    edge_utils: dict[str, list[EdgeUtilization]] = {nid: [] for nid in structure.node_ids}

    for parent_id in structure.node_ids:
        for child_id in structure.children(parent_id):
            r_ij = r_matrix.get((parent_id, child_id), 0.0)
            g_j = g_values.get(child_id, 0.0)
            k_raw[parent_id] += r_ij * g_j
            edge_utils[parent_id].append(
                EdgeUtilization(from_id=parent_id, to_id=child_id, r_ij=r_ij)
            )

    sub_agent_ids = [nid for nid in structure.node_ids if not is_data_node(nid)]
    max_k = max((k_raw[nid] for nid in sub_agent_ids), default=0.0)
    k_normalized: dict[str, float] = {
        nid: (0.0 if is_data_node(nid) else k_raw[nid] / (max_k + config.epsilon))
        for nid in structure.node_ids
    }
    return k_raw, k_normalized, edge_utils


def compute_communication_strengths_no_sink(
    structure: GraphStructure,
    r_matrix: dict[tuple[str, str], float],
    g_values: dict[str, float],
    config: QuantificationConfig,
) -> tuple[dict[str, float], dict[str, float], dict[str, list[EdgeUtilization]]]:
    """
    Like compute_communication_strengths but excludes the virtual __output__ sink.
    K_i = sum_{j in Child(i), j != __output__} R_ij * G_j
    Terminal nodes (no sub-agent children) get K_i = 1.0 after normalization.
    """
    sub_agent_ids = [nid for nid in structure.node_ids if not is_data_node(nid)]
    k_raw: dict[str, float] = {nid: 0.0 for nid in structure.node_ids}
    edge_utils: dict[str, list[EdgeUtilization]] = {nid: [] for nid in structure.node_ids}

    for parent_id in structure.node_ids:
        for child_id in structure.children(parent_id):
            if child_id == DATA_OUTPUT_NODE_ID:
                continue
            r_ij = r_matrix.get((parent_id, child_id), 0.0)
            g_j = g_values.get(child_id, 0.0)
            k_raw[parent_id] += r_ij * g_j
            edge_utils[parent_id].append(
                EdgeUtilization(from_id=parent_id, to_id=child_id, r_ij=r_ij)
            )

    max_k = max((k_raw[nid] for nid in sub_agent_ids), default=0.0)

    k_normalized: dict[str, float] = {}
    for nid in structure.node_ids:
        if is_data_node(nid):
            k_normalized[nid] = 0.0
        elif not structure.children(nid) or all(
            child == DATA_OUTPUT_NODE_ID for child in structure.children(nid)
        ):
            k_normalized[nid] = 1.0
        else:
            k_normalized[nid] = k_raw.get(nid, 0.0) / (max_k + config.epsilon)

    return k_raw, k_normalized, edge_utils
