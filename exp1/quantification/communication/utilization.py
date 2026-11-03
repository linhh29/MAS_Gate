"""Content utilization R_ij between parent and child nodes.

Two approaches are provided:

1. **NLI-based** (recommended): Uses DeBERTa-v3 trained on MultiNLI.
   R_ij = P(entailment | premise=parent_output, hypothesis=child_output)
   This is a continuous value in [0, 1] representing the degree to which
   the parent's output logically entails the child's output.
   Pure entailment -> R_ij ~ 1.0, pure contradiction -> R_ij ~ 0.0.

2. **Embedding-based** (fallback): R_ij = (1/m_i) * sum_k max_{z in Z_j} cos(phi(z_ik), phi(z))
   Semantic similarity via cosine between fragment embeddings.
   Used when NLI model is unavailable or for offline recomputation.
"""

from __future__ import annotations

import numpy as np

from quantification.communication.embedding import EmbeddingStore
from quantification.communication.nli import NLIClient
from quantification.config import QuantificationConfig


def compute_r_ij_nli(
    parent_output: str,
    child_output: str,
    nli_client: NLIClient,
) -> float:
    """NLI-based R_ij: continuous entailment probability.

    R_ij = P(entailment | premise=parent_output, hypothesis=child_output)

    This is the recommended approach. It directly measures logical
    dependency using a DeBERTa-v3 model fine-tuned on MultiNLI.

    Args:
        parent_output: Full text output of the parent node.
        child_output: Full text output of the child node.
        nli_client: NLI model client.

    Returns:
        Continuous entailment degree in [0, 1].
    """
    if not child_output or not child_output.strip():
        return 0.0
    if not parent_output or not parent_output.strip():
        return 0.0

    return nli_client.entailment_probability(
        premise=parent_output,
        hypothesis=child_output,
    )


def compute_r_ij_nli_batch(
    parent_outputs: list[str],
    child_outputs: list[str],
    nli_client: NLIClient,
) -> np.ndarray:
    """Batch NLI-based R_ij for multiple (parent, child) pairs.

    Returns:
        Array of entailment probabilities, shape (len(parent_outputs),).
    """
    if not parent_outputs or not child_outputs:
        return np.array([], dtype=np.float32)

    pairs = [
        (p, c) for p, c in zip(parent_outputs, child_outputs)
        if p and p.strip() and c and c.strip()
    ]
    if not pairs:
        return np.zeros(len(parent_outputs), dtype=np.float32)

    results = nli_client.entailment_probabilities(pairs)
    return results


def compute_r_ij_embedding(
    parent_units: list[str],
    child_units: list[str],
    embedding_store: EmbeddingStore,
    config: QuantificationConfig,
) -> float:
    """Embedding-based R_ij (fallback):
    R_ij = (1/m_i) * sum_k max_{z in Z_j} cos(phi(z_ik), phi(z))

    For each parent fragment, find the most similar child fragment
    by cosine similarity, then average across all parent fragments.
    """
    m_i = len(parent_units)
    if m_i == 0 or not child_units:
        return 0.0

    parent_emb = embedding_store.embed_many(parent_units)
    child_emb = embedding_store.embed_many(child_units)

    total = 0.0
    for parent_vec in parent_emb:
        cos_scores = child_emb @ parent_vec
        max_cos = float(np.max(cos_scores))
        total += max_cos

    return total / m_i


# Backward-compatible alias for the main pipeline
def compute_r_ij(
    parent_units: list[str],
    child_units: list[str],
    embedding_store: EmbeddingStore,
    config: QuantificationConfig,
) -> float:
    """Embedding-based R_ij (legacy, kept for backward compatibility).

    For new code, prefer compute_r_ij_nli() with an NLI client.
    """
    return compute_r_ij_embedding(parent_units, child_units, embedding_store, config)