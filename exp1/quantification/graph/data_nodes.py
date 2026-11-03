"""Boundary data nodes (input question / output answer) for reasoning graphs."""

from __future__ import annotations

from quantification.graph.structure import GraphStructure
from quantification.schemas import DependencyEdge, NodeOutput, ReasoningGraph

DATA_INPUT_NODE_ID = "__input__"
DATA_OUTPUT_NODE_ID = "__output__"
DATA_NODE_IDS = frozenset({DATA_INPUT_NODE_ID, DATA_OUTPUT_NODE_ID})


def is_data_node(node_id: str) -> bool:
    return node_id in DATA_NODE_IDS


def sub_agent_node_ids(graph: ReasoningGraph) -> list[str]:
    """Return sub-agent node IDs, excluding boundary data nodes."""
    return [n.id for n in graph.nodes if not is_data_node(n.id)]


def augment_graph_with_data_nodes(graph: ReasoningGraph) -> ReasoningGraph:
    """
    Attach question/answer boundary nodes so leaf sub-agents have downstream targets.

    - ``__input__`` carries the task question and feeds root sub-agents.
    - ``__output__`` carries the final sub-agent answer and receives sink sub-agents.

    Idempotent: safe to call multiple times.
    """
    if graph.metadata.get("data_nodes_augmented"):
        return graph

    sub_agent_nodes = [n for n in graph.nodes if not is_data_node(n.id)]
    if not sub_agent_nodes:
        return graph

    sub_agent_ids = [n.id for n in sub_agent_nodes]
    sub_agent_set = set(sub_agent_ids)

    temp_structure = GraphStructure(
        ReasoningGraph(
            question=graph.question,
            ground_truth=graph.ground_truth,
            nodes=sub_agent_nodes,
            edges=graph.edges,
            execution_order=graph.execution_order,
        )
    )

    roots = [nid for nid in sub_agent_ids if not temp_structure.parents(nid)]
    sinks = [nid for nid in sub_agent_ids if not temp_structure.children(nid)]

    final_output = ""
    base_order = graph.execution_order or temp_structure.execution_order()
    for nid in reversed(base_order):
        if nid in sub_agent_set:
            final_output = graph.output_by_id().get(nid, "")
            break
    if not final_output:
        final_output = graph.output_by_id().get(sinks[-1], "") if sinks else ""

    new_nodes = [
        NodeOutput(id=DATA_INPUT_NODE_ID, output=graph.question),
        *sub_agent_nodes,
        NodeOutput(id=DATA_OUTPUT_NODE_ID, output=final_output),
    ]

    new_edges = list(graph.edges)
    for root in roots:
        new_edges.append(
            DependencyEdge(from_id=DATA_INPUT_NODE_ID, to_id=root, edge_type="data")
        )
    for sink in sinks:
        new_edges.append(
            DependencyEdge(from_id=sink, to_id=DATA_OUTPUT_NODE_ID, edge_type="data")
        )

    sub_order = [nid for nid in base_order if nid in sub_agent_set]
    for nid in sub_agent_ids:
        if nid not in sub_order:
            sub_order.append(nid)

    metadata = dict(graph.metadata)
    metadata["data_nodes_augmented"] = True
    metadata["data_node_ids"] = [DATA_INPUT_NODE_ID, DATA_OUTPUT_NODE_ID]

    return ReasoningGraph(
        question=graph.question,
        ground_truth=graph.ground_truth,
        nodes=new_nodes,
        edges=new_edges,
        execution_order=[DATA_INPUT_NODE_ID, *sub_order, DATA_OUTPUT_NODE_ID],
        metadata=metadata,
    )
