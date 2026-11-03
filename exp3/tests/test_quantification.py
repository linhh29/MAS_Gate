import math
import unittest

from mas_gate.attribution import Example, Step
from mas_gate.quantification import finalize_scores, score_utility


class QuantificationTests(unittest.TestCase):
    def setUp(self):
        self.example = Example("synthetic", "Question", "Answer", (
            Step(0, "Planner", "First"), Step(1, "Solver", "Second"), Step(2, "Solver", "Third"),
        ))

    def test_chain_formulas(self):
        state = {"id": "synthetic", "j_baseline": 2.0, "nodes": [
            {"step": 0, "j_before": 2.0, "j_after": 3.0},
            {"step": 1, "j_before": 3.0, "j_after": 2.5},
            {"step": 2, "j_before": 2.5, "j_after": 3.5},
        ]}
        result = finalize_scores(self.example, state, [0.5, 0.25])
        self.assertEqual([r["propagation_scope"] for r in result["nodes"]], [1.0, 0.5, 0.0])
        self.assertEqual([r["communication"] for r in result["nodes"]], [0.75, 0.25, 0.0])
        self.assertEqual([r["influence_multiplier"] for r in result["nodes"]], [1.75, 1.25, 1.0])
        self.assertAlmostEqual(result["nodes"][0]["utility"], math.tanh(-1 / (2 + 1e-8)))
        self.assertEqual(result["nodes"][1]["risk_score"], 0.0)
        self.assertGreater(result["nodes"][2]["risk_score"], 0.0)
        for row in result["nodes"]:
            self.assertEqual(row["risk_score"], max(0, -row["impact"]))

    def test_context_and_baseline_readout(self):
        calls = []

        def score(context, answer):
            calls.append((context, answer))
            return float(len(context))

        score_utility(self.example, score)
        self.assertEqual(calls, [("Question", "Answer"), ("Question", "Answer"),
                                 ("Question\nFirst", "Answer"),
                                 ("Question\nFirst\nSecond", "Answer"),
                                 ("Question\nFirst\nSecond\nThird", "Answer")])

    def test_aegis_context_only_turns_stay_in_graph(self):
        example = Example("synthetic", "Question", "Answer", (
            Step(1, "User", "Input", selectable=False), Step(2, "Solver", "Output"),
        ), benchmark="aegis")
        state = score_utility(example, lambda context, answer: float(len(context)))
        result = finalize_scores(example, state, [0.5])
        self.assertEqual(len(result["nodes"]), 2)
        self.assertFalse(result["nodes"][0]["selectable"])
        self.assertEqual(result["nodes"][0]["influence_multiplier"], 1.5)

    def test_single_turn_and_zero_baseline(self):
        example = Example("one", "Question", "Answer", (Step(0, "Solver", "Output"),))
        state = {"id": "one", "j_baseline": 0, "nodes": [
            {"step": 0, "j_before": 0, "j_after": 1},
        ]}
        row = finalize_scores(example, state, [])["nodes"][0]
        self.assertEqual(row["propagation_scope"], 0)
        self.assertEqual(row["influence_multiplier"], 1)
        self.assertEqual(row["risk_score"], 1)

    def test_invalid_scores_rejected(self):
        state = score_utility(self.example, lambda context, answer: 1.0)
        for values in ([0.5], [float("nan"), 0.1], [1.1, 0.1], [True, 0.1], [-0.1, 0.1]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                finalize_scores(self.example, state, values)


if __name__ == "__main__":
    unittest.main()
