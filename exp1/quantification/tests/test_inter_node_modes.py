"""Regression tests for selectable inter-node strength formulas."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quantification.schemas import JPhaseState
from quantification.config import QuantificationConfig
from quantification.unified import (
    _compute_g_values,
    _compute_k_values,
    quantify_from_j_state,
)


class _FixedNLI:
    def __init__(self, score: float) -> None:
        self.score = score

    def entailment_probabilities(self, pairs):
        return [self.score for _ in pairs]


class InterNodeModeTest(unittest.TestCase):
    def test_unified_baseline_is_the_config_default(self) -> None:
        self.assertEqual(QuantificationConfig().inter_node_mode, "unified_baseline")

    def test_unified_baseline_covers_terminal_and_terminal_child(self) -> None:
        # a -> b, where b is terminal. Raw G_b remains zero.
        adj = {"a": ["b"], "b": []}
        topo = ["a", "b"]
        g_values = _compute_g_values(adj, topo)
        self.assertEqual(g_values["b"], 0.0)

        k_raw, k_values = _compute_k_values(
            adj,
            topo,
            {("a", "b"): 0.75},
            g_values,
            ["a", "b"],
            inter_node_mode="unified_baseline",
        )
        self.assertAlmostEqual(k_raw["a"], 1.75)
        self.assertAlmostEqual(k_values["a"], 1.75)
        self.assertEqual(k_values["b"], 1.0)

    def test_unified_baseline_uses_direct_plus_downstream_reach(self) -> None:
        # a -> b -> c. Descendant counts are 2, 1, 0, hence G_b=0.5.
        adj = {"a": ["b"], "b": ["c"], "c": []}
        topo = ["a", "b", "c"]
        g_values = _compute_g_values(adj, topo)
        self.assertAlmostEqual(g_values["a"], 1.0)
        self.assertAlmostEqual(g_values["b"], 0.5)
        self.assertEqual(g_values["c"], 0.0)

        _, k_values = _compute_k_values(
            adj,
            topo,
            {("a", "b"): 1.0, ("b", "c"): 1.0},
            g_values,
            ["a", "b", "c"],
            inter_node_mode="unified_baseline",
        )
        self.assertAlmostEqual(k_values["a"], 2.5)
        self.assertAlmostEqual(k_values["b"], 2.0)
        self.assertEqual(k_values["c"], 1.0)

    def test_legacy_canonical_remains_available(self) -> None:
        adj = {"a": ["b"], "b": []}
        topo = ["a", "b"]
        g_values = _compute_g_values(adj, topo)
        _, k_values = _compute_k_values(
            adj,
            topo,
            {("a", "b"): 0.75},
            g_values,
            ["a", "b"],
            inter_node_mode="legacy_canonical",
        )
        self.assertAlmostEqual(k_values["a"], 0.75)
        self.assertEqual(k_values["b"], 0.0)

    def test_default_quantification_includes_terminal_node(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mas_gate_quant_inter_node_") as td:
            trace = Path(td) / "0__case.json"
            trace.write_text(
                json.dumps({
                    "question": "q",
                    "graph_edges": [{"from": "a", "to": "b"}],
                    "sub_agents": [
                        {"agent_name": "a", "output": "evidence"},
                        {"agent_name": "b", "output": "answer"},
                    ],
                }),
                encoding="utf-8",
            )
            state = JPhaseState(
                graph=None,
                j_baseline=0.5,
                j_by_node={"a": 0.4, "b": 0.3},
                j_before={"a": 0.5, "b": 0.4},
                execution_order=["a", "b"],
            )
            result = quantify_from_j_state(
                state,
                trace,
                judge_score=1.0,
                threshold=0.5,
                nli_client=_FixedNLI(0.8),
            )
            self.assertIsNotNone(result)
            self.assertEqual(result["inter_node_mode"], "unified_baseline")
            self.assertTrue(result["terminal_nodes_included"])
            self.assertEqual([node["id"] for node in result["nodes"]], ["a", "b"])
            self.assertAlmostEqual(result["nodes"][0]["K"], 1.8)
            self.assertEqual(result["nodes"][1]["K"], 1.0)

    def test_single_terminal_node_is_supported_by_default(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mas_gate_quant_single_node_") as td:
            trace = Path(td) / "0__single.json"
            trace.write_text(
                json.dumps({
                    "question": "q",
                    "sub_agents": [{"agent_name": "only", "output": "answer"}],
                }),
                encoding="utf-8",
            )
            state = JPhaseState(
                graph=None,
                j_baseline=0.5,
                j_by_node={"only": 0.4},
                j_before={"only": 0.5},
                execution_order=["only"],
            )
            result = quantify_from_j_state(
                state,
                trace,
                judge_score=1.0,
                threshold=0.5,
                nli_client=_FixedNLI(0.8),
            )
            self.assertIsNotNone(result)
            self.assertNotIn("error", result)
            self.assertEqual(result["num_edges"], 0)
            self.assertEqual(result["nodes"][0]["K"], 1.0)


if __name__ == "__main__":
    unittest.main()
