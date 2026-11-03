"""Orchestrator for full MAS node quantification pipeline."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
from typing import Any

from quantification.adapters.trace_adapter import load_reasoning_graph_from_trace
from quantification.clients.factory import create_quantification_client
from quantification.communication.embedding import EmbeddingStore
from quantification.communication.propagation import (
    compute_agent_propagation_scope,
    compute_baseline_shifted_communication_strengths,
    compute_communication_strengths,
    compute_communication_strengths_no_sink,
    compute_legacy_canonical_communication_strengths,
    compute_propagation_scope,
)
from quantification.communication.segmentation import segment_output
from quantification.communication.nli import NLIClient, get_nli_client, reset_nli_client
from quantification.communication.utilization import (
    compute_r_ij,
    compute_r_ij_nli,
)
from quantification.config import QuantificationConfig
from quantification.graph.data_nodes import (
    DATA_OUTPUT_NODE_ID,
    augment_graph_with_data_nodes,
    is_data_node,
)
from quantification.graph.structure import GraphStructure
from quantification.schemas import GraphMetrics, JPhaseState, NodeMetrics, ReasoningGraph
from quantification.utility.j_scorer import JScorer
from quantification.utility.utility import compute_utilities, compute_utilities_dag_aware


class QuantificationEngine:
    """Compute unified node metrics (U, K, M) for a reasoning graph."""

    def __init__(self, config: QuantificationConfig) -> None:
        self.config = config
        self.client = create_quantification_client(config)
        self.j_scorer = JScorer(self.client, config)
        self.embedding_store = EmbeddingStore(self.client)
        self._nli_client: NLIClient | None = None

    def begin_j_phase(self) -> None:
        begin = getattr(self.client, "begin_scoring_phase", None)
        if callable(begin):
            begin()

    def begin_k_phase(self) -> None:
        begin = getattr(self.client, "begin_embedding_phase", None)
        if callable(begin):
            begin()

    def begin_nli_phase(self) -> None:
        """Load the NLI model for R_ij computation.

        If the embedding engine is loaded (vLLM in-process), release it
        first to free GPU memory, then load the lightweight NLI model.
        """
        release = getattr(self.client, "release_all_engines", None)
        if callable(release):
            release()
        self._nli_client = get_nli_client()

    def release_gpu_engines(self) -> None:
        release = getattr(self.client, "release_all_engines", None)
        if callable(release):
            release()
        reset_nli_client()
        self._nli_client = None

    def compute_j_phase(self, graph: ReasoningGraph) -> JPhaseState:
        if not graph.ground_truth.strip():
            raise ValueError("ground_truth is required for Utility computation")

        graph = augment_graph_with_data_nodes(graph)
        structure = GraphStructure(graph)
        order = [nid for nid in structure.execution_order() if not is_data_node(nid)]

        j_baseline = self.j_scorer.compute_j_baseline(graph)

        if self.config.use_dag_aware_utility:
            # DAG-aware J scoring: j_before = J(ancestors) for correct delta baseline
            j_after, j_before = self.j_scorer.compute_dag_aware_cross_entropies(
                graph, structure, j_baseline=j_baseline
            )
            utilities = compute_utilities_dag_aware(
                structure, j_baseline, j_before, j_after, self.config
            )
            j_by_node = j_after
            j_by_chunk = None
        else:
            j_by_node = self.j_scorer.compute_prefix_cross_entropies(graph, structure)
            utilities = compute_utilities(structure, j_baseline, j_by_node, self.config)
            j_by_chunk = None
            j_before = {}

        return JPhaseState(
            graph=graph,
            j_baseline=j_baseline,
            j_before=j_before if self.config.use_dag_aware_utility else {},
            j_by_node=j_by_node,
            utilities=utilities,
            execution_order=order,
            j_by_chunk=j_by_chunk,
        )

    def finalize_metrics(self, j_state: JPhaseState) -> GraphMetrics:
        graph = j_state.graph
        structure = GraphStructure(graph)
        order = j_state.execution_order
        j_baseline = j_state.j_baseline
        j_by_node = j_state.j_by_node
        utilities = j_state.utilities

        outputs = graph.output_by_id()

        units_by_node = {
            nid: segment_output(outputs.get(nid, "")) for nid in structure.node_ids
        }

        r_matrix: dict[tuple[str, str], float] = {}
        edge_pairs = [
            (parent_id, child_id)
            for parent_id in structure.node_ids
            for child_id in structure.children(parent_id)
        ]

        use_nli = self._nli_client is not None

        if use_nli:
            # NLI-based R_ij: batch all edge pairs
            parent_outputs = [outputs.get(p, "") for p, _ in edge_pairs]
            child_outputs = [outputs.get(c, "") for _, c in edge_pairs]
            pairs = list(zip(parent_outputs, child_outputs))

            entailment_scores = self._nli_client.entailment_probabilities(pairs)

            for (parent_id, child_id), score in zip(edge_pairs, entailment_scores):
                r_matrix[(parent_id, child_id)] = float(score)
        else:
            # Embedding-based R_ij (fallback)
            def _compute_edge(pair: tuple[str, str]) -> tuple[tuple[str, str], float]:
                parent_id, child_id = pair
                return pair, compute_r_ij(
                    units_by_node[parent_id],
                    units_by_node[child_id],
                    self.embedding_store,
                    self.config,
                )

            if edge_pairs:
                workers = min(self.config.embedding_concurrency, len(edge_pairs))
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(_compute_edge, pair) for pair in edge_pairs]
                    for future in as_completed(futures):
                        pair, value = future.result()
                        r_matrix[pair] = value

        inter_node_mode = self.config.inter_node_mode
        if inter_node_mode == "unified_baseline":
            g_values = compute_agent_propagation_scope(
                structure,
                self.config,
                use_depth_fair=self.config.use_depth_fair_g,
            )
            k_raw, k_normalized, edge_utils = (
                compute_baseline_shifted_communication_strengths(
                    structure, r_matrix, g_values
                )
            )
        elif inter_node_mode == "legacy_canonical":
            g_values = compute_agent_propagation_scope(
                structure,
                self.config,
                use_depth_fair=self.config.use_depth_fair_g,
            )
            k_raw, k_normalized, edge_utils = (
                compute_legacy_canonical_communication_strengths(
                    structure, r_matrix, g_values
                )
            )
        elif inter_node_mode in {"legacy_no_sink", "legacy_with_sink"}:
            # Preserve both historical normalized implementations as options.
            g_values = compute_propagation_scope(structure, self.config)
            k_function = (
                compute_communication_strengths
                if inter_node_mode == "legacy_with_sink"
                else compute_communication_strengths_no_sink
            )
            k_raw, k_normalized, edge_utils = k_function(
                structure, r_matrix, g_values, self.config
            )
        else:
            raise ValueError(f"Unsupported inter_node_mode: {inter_node_mode!r}")

        node_metrics: list[NodeMetrics] = []
        prev_j = j_baseline
        all_zero = True

        for node_id in order:
            current_j = j_by_node.get(node_id, prev_j)
            u = utilities.get(node_id, 0.0)
            k = k_raw.get(node_id, 0.0)
            kn = k_normalized.get(node_id, 0.0)

            risk = max(0.0, -u) * kn
            opp = max(0.0, u) * kn

            node_metrics.append(
                NodeMetrics(
                    id=node_id,
                    utility=u,
                    communication=k,
                    communication_normalized=kn,
                    main_value=u * kn,
                    risk_score=risk,
                    opportunity_score=opp,
                    j_prefix=current_j,
                    j_previous=prev_j,
                    propagation_scope=g_values.get(node_id, 0.0),
                    edge_utilizations=edge_utils.get(node_id, []),
                )
            )
            k_has_signal = (
                k != 1.0 if inter_node_mode == "unified_baseline" else k != 0.0
            )
            if current_j != j_baseline or u != 0.0 or k_has_signal:
                all_zero = False
            prev_j = current_j

        if all_zero:
            execution_status = "no_execution"
            failure_reason = "MAS sub-agents produced no output; all j_prefix values equal j_baseline"
        else:
            execution_status = "executed"
            failure_reason = ""

        metadata = dict(graph.metadata)
        metadata.update({
            "inter_node_mode": inter_node_mode,
            "g_type": (
                "depth_fair" if self.config.use_depth_fair_g else "simple_descendants"
            ),
            "k_formula": (
                "1 + sum_j R_ij * (1 + G_j)"
                if inter_node_mode == "unified_baseline"
                else inter_node_mode
            ),
            "terminal_nodes_included": True,
            "terminal_k_behavior": (
                "K=1 baseline"
                if inter_node_mode == "unified_baseline"
                else inter_node_mode
            ),
        })

        return GraphMetrics(
            question=graph.question,
            ground_truth=graph.ground_truth,
            j_baseline=j_baseline,
            execution_order=order,
            node_metrics=node_metrics,
            metadata=metadata,
            execution_status=execution_status,
            failure_reason=failure_reason,
        )

    def compute(self, graph: ReasoningGraph) -> GraphMetrics:
        j_state = self.compute_j_phase(graph)
        use_nli = os.getenv("QUANTIFICATION_USE_NLI", "1").strip().lower() in {"1", "true", "yes", "on"}
        if use_nli:
            self.begin_nli_phase()
        else:
            # MAS-Gate full local-vLLM mode: use the BGE /embeddings endpoint instead
            # of loading the BGE/NLI transformer directly in this Ray reward process.
            self.begin_k_phase()
        return self.finalize_metrics(j_state)

    def compute_from_trace(
        self,
        trace_path: Path | str,
        ground_truth: str | None = None,
        ground_truth_field: str | None = None,
    ) -> GraphMetrics:
        graph = load_reasoning_graph_from_trace(
            trace_path,
            ground_truth=ground_truth,
            ground_truth_field=ground_truth_field,
        )
        return self.compute(graph)
