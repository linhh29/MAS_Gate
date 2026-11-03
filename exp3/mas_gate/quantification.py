"""Node quantification on the chronological chain used in Experiment 3.

J is stored as mean negative log-likelihood (NLL). Consequently J_before minus
J_after is the improvement in reference-answer log-likelihood. In this module
K denotes the paper's communication term, not the already-shifted multiplier.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from .attribution import Example


def _finite(value: float, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"invalid {name}")
    return value


def score_utility(
    example: Example, score: Callable[[str, str], float]
) -> dict[str, Any]:
    """Compute all before/after NLLs without accessing error annotations.

    User/System turns in AEGIS remain in the chain and scoring contexts. Their
    exclusion from target selection is handled only by the attribution module.
    Empty outputs are omitted from context concatenation, as in the experiment.
    """
    baseline = _finite(score(example.question, example.answer), "baseline NLL", nonnegative=True)
    cache: dict[str, float] = {}

    def nll(context: str) -> float:
        if context not in cache:
            cache[context] = _finite(score(context, example.answer), "NLL", nonnegative=True)
        return cache[context]

    prefix = [example.question]
    nodes = []
    for step in example.steps:
        before = nll("\n".join(prefix))
        if step.content:
            prefix.append(step.content)
        after = nll("\n".join(prefix))
        nodes.append({"step": step.index, "j_before": before, "j_after": after})
    return {"id": example.id, "j_baseline": baseline, "nodes": nodes}


def finalize_scores(
    example: Example, utility: dict[str, Any], edge_probabilities: Sequence[float]
) -> dict[str, Any]:
    """Compute U, G, K, I and risk from saved NLLs and adjacent-edge NLI scores.

    G counts all later real trajectory turns, without synthetic boundary nodes.
    The chronological chain is a scoring approximation, not an observed MAS
    communication topology. No correctness label is used in this calculation.
    """
    steps = example.steps
    if not steps:
        raise ValueError("empty trajectory")
    rows = utility["nodes"]
    if utility["id"] != example.id or [r["step"] for r in rows] != [s.index for s in steps]:
        raise ValueError("utility state does not match the trajectory")
    if len(edge_probabilities) != len(steps) - 1:
        raise ValueError("one NLI score is required per chronological edge")
    probabilities = [_finite(r, "entailment probability", nonnegative=True) for r in edge_probabilities]
    if any(r > 1 for r in probabilities):
        raise ValueError("entailment probability exceeds one")
    baseline = _finite(utility["j_baseline"], "baseline NLL", nonnegative=True)
    denominator = abs(baseline) + 1e-8
    count = len(steps)
    nodes = []
    for position, (step, row) in enumerate(zip(steps, rows)):
        before = _finite(row["j_before"], "before NLL", nonnegative=True)
        after = _finite(row["j_after"], "after NLL", nonnegative=True)
        u = math.tanh((before - after) / denominator)
        g = (count - position - 1) / max(count - 1, 1)
        child_g = (count - position - 2) / max(count - 1, 1)
        k = probabilities[position] * (1.0 + child_g) if position < count - 1 else 0.0
        impact = u * (1.0 + k)
        nodes.append({
            "step": step.index, "agent": step.agent, "selectable": step.selectable,
            "j_before": before, "j_after": after, "utility": u,
            "propagation_scope": g, "communication": k,
            "influence_multiplier": 1.0 + k, "impact": impact,
            "risk_score": max(0.0, -u) * (1.0 + k),
        })
    edges = [
        {"from": left.index, "to": right.index, "entailment": probability}
        for left, right, probability in zip(steps, steps[1:], probabilities)
    ]
    return {"id": example.id, "graph": "chronological_chain", "j_baseline": baseline,
            "nodes": nodes, "edges": edges}
