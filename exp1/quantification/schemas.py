"""Data schemas for MAS reasoning-graph quantification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NodeOutput:
    """A single accountable generation/decision unit in the dependency graph."""

    id: str
    output: str


@dataclass(frozen=True)
class DependencyEdge:
    """Directed edge: child generation depends on parent output."""

    from_id: str
    to_id: str
    edge_type: str = "prerequisite"


@dataclass
class ReasoningGraph:
    """Full input for node-level quantification."""

    question: str
    ground_truth: str
    nodes: list[NodeOutput]
    edges: list[DependencyEdge] = field(default_factory=list)
    execution_order: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def node_ids(self) -> list[str]:
        return [n.id for n in self.nodes]

    def output_by_id(self) -> dict[str, str]:
        return {n.id: n.output for n in self.nodes}


@dataclass
class EdgeUtilization:
    """R_ij between parent i and child j."""

    from_id: str
    to_id: str
    r_ij: float


@dataclass
class NodeMetrics:
    """Per-node unified metric state s_i = (U_i, K_i) and main value M_i.

    In the default ``unified_baseline`` mode, K is the unnormalized influence
    multiplier ``1 + sum_j R_ij * (1 + G_j)``.  The historical modes retain
    their previous normalized semantics in ``communication_normalized``.

    Key derived values use the selected mode's effective K value:
      M_i = U_i * K_i           — overall contribution
      O_i = (U_i)_+ * K_i       — opportunity score
      E_i = (-U_i)_+ * K_i      — risk score
    """

    id: str
    utility: float
    communication: float
    communication_normalized: float
    main_value: float
    risk_score: float = 0.0
    opportunity_score: float = 0.0
    j_prefix: float | None = None
    j_previous: float | None = None
    propagation_scope: float | None = None
    edge_utilizations: list[EdgeUtilization] = field(default_factory=list)


@dataclass
class JPhaseState:
    """Intermediate J-scoring result; K metrics are computed in a second GPU phase."""

    graph: ReasoningGraph
    j_baseline: float
    j_by_node: dict[str, float]
    j_before: dict[str, float] = field(default_factory=dict)
    utilities: dict[str, float] = field(default_factory=dict)
    execution_order: list[str] = field(default_factory=list)
    j_by_chunk: dict[str, list[float]] | None = None


@dataclass
class GraphMetrics:
    """Aggregated quantification result for one reasoning graph."""

    question: str
    ground_truth: str
    j_baseline: float
    execution_order: list[str]
    node_metrics: list[NodeMetrics]
    metadata: dict[str, Any] = field(default_factory=dict)
    execution_status: str = "unknown"
    failure_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "question": self.question,
            "ground_truth": self.ground_truth,
            "j_baseline": self.j_baseline,
            "execution_status": self.execution_status,
            "execution_order": self.execution_order,
            "node_metrics": [
                {
                    "id": m.id,
                    "utility": m.utility,
                    "communication": m.communication,
                    "communication_normalized": m.communication_normalized,
                    "main_value": m.main_value,
                    "risk_score": m.risk_score,
                    "opportunity_score": m.opportunity_score,
                    "j_prefix": m.j_prefix,
                    "j_previous": m.j_previous,
                    "propagation_scope": m.propagation_scope,
                    "edge_utilizations": [
                        {"from_id": e.from_id, "to_id": e.to_id, "r_ij": e.r_ij}
                        for e in m.edge_utilizations
                    ],
                }
                for m in self.node_metrics
            ],
            "metadata": self.metadata,
        }
        if self.failure_reason:
            result["failure_reason"] = self.failure_reason
        return result
