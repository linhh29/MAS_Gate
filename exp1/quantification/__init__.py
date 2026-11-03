"""
MAS node quantification: Utility (U) and inter-node influence multiplier (K).

Default formula: K_i = 1 + sum_j R_ij * (1 + G_j), M_i = U_i * K_i.
"""

from quantification.config import QuantificationConfig
from quantification.metrics import QuantificationEngine
from quantification.schemas import GraphMetrics, NodeMetrics, ReasoningGraph

__all__ = [
    "QuantificationConfig",
    "QuantificationEngine",
    "GraphMetrics",
    "NodeMetrics",
    "ReasoningGraph",
]
