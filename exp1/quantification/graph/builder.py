"""Build ReasoningGraph from explicit inputs."""

from __future__ import annotations

from quantification.schemas import DependencyEdge, NodeOutput, ReasoningGraph


def build_reasoning_graph(
    question: str,
    ground_truth: str,
    nodes: list[NodeOutput] | list[dict[str, str]],
    edges: list[DependencyEdge] | list[dict[str, str]] | None = None,
    execution_order: list[str] | None = None,
    metadata: dict | None = None,
) -> ReasoningGraph:
    """Construct a ReasoningGraph from Python objects or plain dicts."""
    node_objs: list[NodeOutput] = []
    for item in nodes:
        if isinstance(item, NodeOutput):
            node_objs.append(item)
        else:
            node_objs.append(NodeOutput(id=item["id"], output=item["output"]))

    edge_objs: list[DependencyEdge] = []
    if edges:
        for item in edges:
            if isinstance(item, DependencyEdge):
                edge_objs.append(item)
            else:
                edge_objs.append(
                    DependencyEdge(
                        from_id=item["from"],
                        to_id=item["to"],
                        edge_type=item.get("type", "prerequisite"),
                    )
                )

    return ReasoningGraph(
        question=question,
        ground_truth=ground_truth,
        nodes=node_objs,
        edges=edge_objs,
        execution_order=execution_order,
        metadata=metadata or {},
    )
