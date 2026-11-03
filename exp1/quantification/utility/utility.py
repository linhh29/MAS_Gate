"""Marginal utility U_i from prefix cross-entropy improvements."""

from __future__ import annotations

import math

from quantification.config import QuantificationConfig
from quantification.graph.data_nodes import is_data_node
from quantification.graph.structure import GraphStructure


def compute_delta(
    prev_j: float,
    current_j: float,
    j_baseline: float,
    epsilon: float = 1e-8,
) -> float:
    """delta = (J_prev - J_curr) / (|J_baseline| + epsilon)"""
    return (prev_j - current_j) / (abs(j_baseline) + epsilon)


def compute_unified_u_i(delta: float, score: float | None = None, is_terminal: bool = False) -> float:
    """U_i = tanh(delta) for non-terminal nodes; score-symmetric for terminal nodes.

    Non-terminal (is_terminal=False):
      U_i = tanh(delta)  — zero hyperparameters, delta sign = U_i sign

    Terminal (is_terminal=True):
      U_i = score * tanh(delta) + (1 - score) * tanh(-delta)
      This formula is symmetric and principled:
      - score=1 (correct) → U_i = tanh(delta)  — same as non-terminal
      - score=0 (wrong) → U_i = tanh(-delta) = -tanh(delta)  — utility is FLIPPED:
        even if delta>0 (the node improved GT prediction), the wrong answer means
        the node's overall contribution was harmful, so U_i becomes negative.
      - score=0.5 (partial) → U_i = 0.5*tanh(delta)+0.5*tanh(-delta) = 0  — neutral

    No +1/-1 shift, no threshold. The delta sign is preserved for correct answers,
    flipped for wrong answers, and neutralized for partial correctness.
    """
    if is_terminal and score is not None:
        return score * math.tanh(delta) + (1.0 - score) * math.tanh(-delta)
    return math.tanh(delta)


def compute_utilities(
    structure: GraphStructure,
    j_baseline: float,
    j_by_node: dict[str, float],
    config: QuantificationConfig,
) -> dict[str, float]:
    """
    U_i = tanh(delta) where delta = (J_prev - J_curr) / (|J_baseline| + epsilon)

    Zero hyperparameters, universal for all nodes.
    - delta > 0  → U_i > 0  (node helped predict GT)
    - delta = 0  → U_i = 0  (node contributed nothing)
    - delta < 0  → U_i < 0  (node hurt GT prediction)

    Note: This is the legacy non-terminal-aware path. The canonical path
    (quantify_from_j_state) applies score-weighting for terminal nodes.
    """
    order = [nid for nid in structure.execution_order() if not is_data_node(nid)]
    utilities: dict[str, float] = {}
    denom = abs(j_baseline) + config.epsilon

    prev_j = j_baseline
    for node_id in order:
        current_j = j_by_node.get(node_id, prev_j)
        delta = (prev_j - current_j) / denom
        utilities[node_id] = math.tanh(delta)
        prev_j = current_j

    return utilities


def compute_utilities_dag_aware(
    structure: GraphStructure,
    j_baseline: float,
    j_before: dict[str, float],
    j_after: dict[str, float],
    config: QuantificationConfig,
) -> dict[str, float]:
    """
    DAG-aware utility: U_i = tanh(delta)
    where delta = (J(C_i) - J(C_i^+)) / (|J_0| + epsilon)

    C_i = question + ancestors' outputs (without node i's own output)
    C_i^+ = C_i + node i's output
    """
    order = [nid for nid in structure.execution_order() if not is_data_node(nid)]
    utilities: dict[str, float] = {}
    denom = abs(j_baseline) + config.epsilon

    for node_id in order:
        j_before_val = j_before.get(node_id, j_baseline)
        j_after_val = j_after.get(node_id, j_before_val)
        delta = (j_before_val - j_after_val) / denom
        utilities[node_id] = math.tanh(delta)

    return utilities