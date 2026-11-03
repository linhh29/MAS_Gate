"""Regression guard: correctness is diagnostic, not directly fused into mean_impact."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quantification.schemas import JPhaseState
from quantification.unified import quantify_from_j_state


class _FixedNLI:
    def entailment_probabilities(self, pairs):
        return [0.8 for _ in pairs]


class JudgeScoreIndependenceTest(unittest.TestCase):
    def test_mean_impact_is_identical_for_judge_zero_and_one(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mas_gate_quant_judge_independence_") as td:
            trace = Path(td) / "0__case.json"
            trace.write_text(json.dumps({
                "question": "q",
                "graph_edges": [{"from": "a", "to": "b", "type": "dependency"}],
                "sub_agents": [
                    {"agent_name": "a", "output": "evidence"},
                    {"agent_name": "b", "output": "answer"},
                ],
            }), encoding="utf-8")
            state = JPhaseState(
                graph=None,  # quantify_from_j_state only consumes the scalar state below
                j_baseline=0.5,
                j_by_node={"a": 0.4, "b": 0.3},
                j_before={"a": 0.5, "b": 0.4},
                utilities={},
                execution_order=["a", "b"],
            )
            wrong = quantify_from_j_state(state, trace, judge_score=0.0, threshold=0.5, nli_client=_FixedNLI())
            correct = quantify_from_j_state(state, trace, judge_score=1.0, threshold=0.5, nli_client=_FixedNLI())
            self.assertIsNotNone(wrong)
            self.assertIsNotNone(correct)
            self.assertEqual(wrong["mean_impact"], correct["mean_impact"])
            self.assertEqual(wrong["nodes"], correct["nodes"])
            self.assertEqual(wrong["edges"], correct["edges"])
            self.assertEqual(wrong["judge_score"], 0.0)
            self.assertEqual(correct["judge_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
