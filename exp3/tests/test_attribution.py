"""Synthetic tests for dataset isolation and benchmark attribution contracts."""

import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

from mas_gate.attribution import (
    Example, Label, Step, build_messages, evaluate, load_aegis, load_who_when,
    parse_prediction, rank_candidates,
)


class AttributionTests(unittest.TestCase):
    def ww(self):
        return Example("synthetic/example", "Which value is larger?", "9", (
            Step(0, "Researcher", "The values are 3 and 8."),
            Step(1, "Checker", "8 is larger than 3."),
            Step(2, "Reporter", "8"),
        ))

    def aegis(self):
        return Example("private_record_identifier", "What is 3 + 6?", "9", (
            Step(1, "User", "Please calculate.", selectable=False),
            Step(2, "Writer", "I obtain 8.", "analysis"),
            Step(3, "Checker", "I agree."),
            Step(4, "Writer", "Answer: 8", "final"),
            Step(5, "System", "Done", selectable=False),
        ), "8", "aegis")

    def test_who_when_risk_ties_and_soft_selection(self):
        example = self.ww()
        candidates = rank_candidates(example, {0: 0.5, 1: 0.5, 2: 0.0}, 2)
        self.assertEqual([c["step"] for c in candidates], [0, 1])
        direct = build_messages(example)[1]["content"]
        guided = build_messages(example, candidates)[1]["content"]
        self.assertTrue(guided.startswith(direct + "\n\nAdditional MAS-Gate"))
        self.assertIn("Rank 1: Step 0\nRank 2: Step 1\n\nUse", guided)
        prediction = parse_prediction(
            "Agent Name: Reporter\nStep Number: 2\nReason for Mistake: Wrong result.", example
        )
        self.assertTrue(prediction["parse_ok"])
        self.assertTrue(evaluate(prediction, Label("Reporter", 2), "who_when")["exact_step"])

    def test_aegis_agent_max_and_representative_tie_breaking(self):
        example = self.aegis()
        candidates = rank_candidates(example, {1: 100, 2: 0.1, 3: 0.8, 4: 0.8, 5: 100}, 4)
        self.assertEqual([c["agent"] for c in candidates], ["Checker", "Writer"])
        self.assertEqual(candidates[1]["step"], 4)
        self.assertEqual(candidates[1]["source_steps"], [2, 4])
        tied = rank_candidates(example, {2: 0.8, 3: 0.8, 4: 0.8}, 4)
        self.assertEqual([c["step"] for c in tied], [2, 3])
        prompt = build_messages(example, candidates)[1]["content"]
        self.assertIn("Top-2 most suspicious agents", prompt)
        self.assertIn("Rank 1: Checker\nRank 2: Writer", prompt)
        self.assertIn("Step 1\nAgent: User\nSelection status: context only; not selectable", prompt)
        self.assertNotIn("0.8", prompt)
        self.assertNotIn(example.id, prompt)

    def test_controls_are_repeatable_and_shuffled_changes_order(self):
        example = self.ww()
        risks = {0: 0.7, 1: 0.5, 2: 0.1}
        shuffled = rank_candidates(example, risks, 2, "shuffled")
        self.assertEqual({c["step"] for c in shuffled}, {0, 1})
        self.assertEqual([c["step"] for c in shuffled], [1, 0])
        random = rank_candidates(example, risks, 2, "random")
        self.assertEqual(random, rank_candidates(example, dict(reversed(list(risks.items()))), 2, "random"))
        with self.assertRaises(ValueError):
            rank_candidates(self.aegis(), {2: 0.1, 3: 0.1, 4: 0.1}, 4, "random")

    def test_invalid_risks_and_candidates_fail(self):
        for value in (True, -0.1, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                rank_candidates(self.ww(), {0: value, 1: 0, 2: 0}, 8)
        with self.assertRaises(ValueError):
            rank_candidates(self.ww(), {0: 0}, 8)
        with self.assertRaises(ValueError):
            build_messages(self.ww(), [{"rank": 1, "step": 99}])

    def test_parser_formats_and_evaluation_are_benchmark_specific(self):
        ww = self.ww()
        # WW's published parser rejects even a trailing newline.
        text = "Agent Name: Researcher\nStep Number: 0\nReason for Mistake: Wrong premise."
        self.assertTrue(parse_prediction(text, ww)["parse_ok"])
        self.assertFalse(parse_prediction(text + "\n", ww)["parse_ok"])
        self.assertFalse(parse_prediction(text.replace("Number: 0", "Number: 3"), ww)["parse_ok"])
        inconsistent = parse_prediction(text.replace("Researcher", "Checker"), ww)
        self.assertFalse(inconsistent["agent_step_consistent"])
        self.assertTrue(evaluate(inconsistent, Label("Researcher", 0), "who_when")["exact_step"])
        aegis = self.aegis()
        good = parse_prediction("Agent Name: Writer\nReason for Mistake: Wrong sum.\n", aegis)
        self.assertTrue(evaluate(good, Label("Writer"), "aegis")["agent_hit"])
        for name in ("User", "System", "writer", "Unknown"):
            bad = parse_prediction(f"Agent Name: {name}\nReason for Mistake: Wrong sum.", aegis)
            self.assertFalse(bad["parse_ok"])

    def test_who_when_loader_separates_labels_and_ignores_sidecars(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            group = root / "Algorithm-Generated"
            group.mkdir()
            raw = {"question": "Q", "ground_truth": "A", "history": [
                {"name": "Solver", "content": "Observable", "is_injected": True}
            ], "mistake_agent": "Solver", "mistake_step": "0", "mistake_reason": "PRIVATE_LABEL"}
            (group / "1.json").write_text(json.dumps(raw), encoding="utf-8")
            (group / "._1.json").write_text("invalid sidecar", encoding="utf-8")
            (root / "manifest.json").write_text("{}", encoding="utf-8")
            example, label = load_who_when(root, require_complete=False)[0]
            self.assertEqual(label, Label("Solver", 0))
            self.assertNotIn("PRIVATE_LABEL", json.dumps(asdict(example)))
            with self.assertRaises(ValueError):
                load_who_when(root)

    def test_aegis_allowlist_and_unique_exact_agent_filter(self):
        raw = {
            "id": "private_record_identifier", "num_injected_agents": 2,
            "input": {"query": "Q", "final_output": "", "conversation_history": [
                {"step": 1, "agent_name": "User", "content": "Question", "is_injected": False},
                {"step": 2, "agent_name": "Solver", "phase": "final", "content": "Observable", "is_injected": True},
            ]},
            "ground_truth": {"correct_answer": "A", "injected_agents": [
                {"agent_name": "Solver", "strategy": "PRIVATE_STRATEGY"},
                {"agent_name": "Solver", "faulty_output": "PRIVATE_OUTPUT"},
            ]},
            "metadata": {"error_type": "PRIVATE_ERROR"},
        }
        multi = json.loads(json.dumps(raw))
        multi["ground_truth"]["injected_agents"][1]["agent_name"] = "solver"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.jsonl"
            path.write_text(json.dumps(raw) + "\n" + json.dumps(multi) + "\n", encoding="utf-8")
            pairs = load_aegis(path, require_complete=False)
            self.assertEqual(len(pairs), 1)
            example, label = pairs[0]
            self.assertEqual(label, Label("Solver"))
            self.assertEqual(example.observed_final_output, "Observable")
            self.assertEqual([s.index for s in example.steps], [1, 2])
            serialized = json.dumps(asdict(example))
            for excluded in ("PRIVATE_STRATEGY", "PRIVATE_OUTPUT", "PRIVATE_ERROR", "is_injected", "num_injected_agents"):
                self.assertNotIn(excluded, serialized)
            self.assertNotIn(example.id, json.dumps(build_messages(example)))
            self.assertEqual(build_messages(example), build_messages(replace(example, id="another_id")))
            with self.assertRaises(ValueError):
                load_aegis(path)


if __name__ == "__main__":
    unittest.main()
