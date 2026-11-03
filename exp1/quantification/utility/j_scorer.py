"""Answer cross-entropy J(q, A) via remote API logprobs."""

from __future__ import annotations

import re
from typing import Sequence

from quantification.clients.factory import QuantificationClient
from quantification.config import QuantificationConfig
from quantification.graph.data_nodes import DATA_OUTPUT_NODE_ID, is_data_node
from quantification.graph.structure import GraphStructure
from quantification.schemas import ReasoningGraph


def _chunk_output(text: str, max_chunks: int = 8, min_chunk_chars: int = 200) -> list[str]:
    """Split a node output into meaningful chunks.

    Splits by paragraphs first, then by sentences if paragraphs are too few/large.
    Returns at most max_chunks chunks, each at least min_chunk_chars.
    """
    if not text.strip():
        return [text]

    # Split by paragraphs (double newline)
    paragraphs = re.split(r'\n\s*\n', text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if len(paragraphs) <= 1:
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) <= 1:
            # Just split by fixed length
            chunk_size = max(min_chunk_chars, len(text) // max_chunks)
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            return [c for c in chunks if c.strip()][:max_chunks]
        chunks = sentences
    else:
        chunks = paragraphs

    # Merge small chunks to meet min_chunk_chars
    merged = []
    current = ""
    for c in chunks:
        if current and len(current) < min_chunk_chars:
            current += "\n\n" + c
        elif current:
            merged.append(current)
            current = c
        else:
            current = c
    if current:
        merged.append(current)

    return merged[:max_chunks]


class JScorer:
    """Compute J_0 and J(A_i) for each prefix along execution order."""

    def __init__(self, client: QuantificationClient, config: QuantificationConfig) -> None:
        self.client = client
        self.config = config

    def compute_j_baseline(self, graph: ReasoningGraph) -> float:
        return self.client.compute_answer_cross_entropy(graph.question, graph.ground_truth)

    def compute_prefix_cross_entropies(
        self,
        graph: ReasoningGraph,
        structure: GraphStructure,
    ) -> dict[str, float]:
        outputs = graph.output_by_id()
        order = [nid for nid in structure.execution_order() if not is_data_node(nid)]
        prefix_outputs: list[str] = []
        j_by_node: dict[str, float] = {}

        for node_id in order:
            prefix_outputs.append(outputs.get(node_id, ""))
            context = self.client.format_context(graph.question, prefix_outputs)
            j_by_node[node_id] = self.client.compute_answer_cross_entropy(
                context, graph.ground_truth
            )

        return j_by_node

    def compute_dag_aware_cross_entropies(
        self,
        graph: ReasoningGraph,
        structure: GraphStructure,
        j_baseline: float | None = None,
    ) -> dict[str, float]:
        """DAG-aware: J(C_i^+) = J(question + ancestors' outputs + node's own output).

        Returns a dict mapping node_id → J(C_i^+).
        Also returns the J(C_i) "before" values used for utility computation.
        """
        outputs = graph.output_by_id()
        order = [nid for nid in structure.execution_order() if not is_data_node(nid)]

        # Reuse the already-computed J(question, answer) baseline for every root
        # context_before.  Previously it was sent to Qwen30B a second time.
        j_cache: dict[str, float] = (
            {graph.question: float(j_baseline)} if j_baseline is not None else {}
        )
        j_before: dict[str, float] = {}
        j_after: dict[str, float] = {}

        def _build_context(node_ids: list[str]) -> str:
            parts = [outputs.get(nid, "") for nid in node_ids if nid not in (DATA_OUTPUT_NODE_ID,)]
            return self.client.format_context(graph.question, parts)

        for node_id in order:
            ancestors = sorted(structure.ancestors(node_id) - {DATA_OUTPUT_NODE_ID})
            ancestors = [a for a in ancestors if not is_data_node(a)]

            # J(C_i): context without this node's output
            context_before = _build_context(ancestors)
            if context_before not in j_cache:
                j_cache[context_before] = self.client.compute_answer_cross_entropy(
                    context_before, graph.ground_truth
                )
            j_before[node_id] = j_cache[context_before]

            # J(C_i^+): context with ancestors + this node's output
            context_after = _build_context(ancestors + [node_id])
            if context_after not in j_cache:
                j_cache[context_after] = self.client.compute_answer_cross_entropy(
                    context_after, graph.ground_truth
                )
            j_after[node_id] = j_cache[context_after]

        return j_after, j_before

    def compute_dag_aware_chunked_cross_entropies(
        self,
        graph: ReasoningGraph,
        structure: GraphStructure,
        max_chunks: int = 8,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, list[float]]]:
        """DAG-aware with chunked output: J(C_i^+) for each chunk of node output.

        Returns:
            j_after: dict node_id → J(ancestors + full_output)
            j_before: dict node_id → J(ancestors)
            j_by_chunk: dict node_id → list of J(ancestors + chunk) for each chunk
        """
        outputs = graph.output_by_id()
        order = [nid for nid in structure.execution_order() if not is_data_node(nid)]

        j_cache: dict[str, float] = {}
        j_before: dict[str, float] = {}
        j_after: dict[str, float] = {}
        j_by_chunk: dict[str, list[float]] = {}

        def _build_context(node_ids: list[str]) -> str:
            parts = [outputs.get(nid, "") for nid in node_ids if nid not in (DATA_OUTPUT_NODE_ID,)]
            return self.client.format_context(graph.question, parts)

        for node_id in order:
            ancestors = sorted(structure.ancestors(node_id) - {DATA_OUTPUT_NODE_ID})
            ancestors = [a for a in ancestors if not is_data_node(a)]

            # J(C_i): context without this node's output
            context_before = _build_context(ancestors)
            if context_before not in j_cache:
                j_cache[context_before] = self.client.compute_answer_cross_entropy(
                    context_before, graph.ground_truth
                )
            j_before[node_id] = j_cache[context_before]

            # J(C_i^+): context with ancestors + full node output
            context_after = _build_context(ancestors + [node_id])
            if context_after not in j_cache:
                j_cache[context_after] = self.client.compute_answer_cross_entropy(
                    context_after, graph.ground_truth
                )
            j_after[node_id] = j_cache[context_after]

            # Per-chunk J scores (cumulative: marginal contribution of each chunk)
            node_output = outputs.get(node_id, "")
            chunks = _chunk_output(node_output, max_chunks=max_chunks)
            chunk_j_scores: list[float] = []
            cumulative_parts = [outputs.get(a, "") for a in ancestors if a not in (DATA_OUTPUT_NODE_ID,)]
            for chunk in chunks:
                cumulative_parts.append(chunk)
                cum_context = self.client.format_context(graph.question, cumulative_parts)
                if cum_context not in j_cache:
                    j_cache[cum_context] = self.client.compute_answer_cross_entropy(
                        cum_context, graph.ground_truth
                    )
                chunk_j_scores.append(j_cache[cum_context])
            j_by_chunk[node_id] = chunk_j_scores

        return j_after, j_before, j_by_chunk

    def compute_j_for_prefixes(
        self,
        question: str,
        ground_truth: str,
        prefix_sequences: Sequence[Sequence[str]],
    ) -> list[float]:
        results: list[float] = []
        for outputs in prefix_sequences:
            context = self.client.format_context(question, list(outputs))
            results.append(self.client.compute_answer_cross_entropy(context, ground_truth))
        return results
