import tempfile
import unittest
import argparse
import contextlib
import io
from pathlib import Path
from unittest.mock import patch

from mas_gate.__main__ import load_matching, prepare_manifest, record_path, write_once, digest
from mas_gate import __main__ as cli, models
from mas_gate.attribution import Example, Label, Step
from dataclasses import asdict


class WorkflowTests(unittest.TestCase):
    def test_atomic_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_once(path, {"value": 1})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_once(path, {"value": 2})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).glob(".pending-*")), [])

    def test_manifest_safe_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = {"settings": [("mas_gate", 8)], "temperature": 0}
            first = prepare_manifest(path, config)
            self.assertEqual(prepare_manifest(path, config), first)
            with self.assertRaises(ValueError):
                prepare_manifest(path, {**config, "temperature": 1})

    def test_record_input_integrity(self):
        example = Example("synthetic", "Question", "Answer", (Step(0, "Solver", "Output"),))
        with tempfile.TemporaryDirectory() as directory:
            path = record_path(Path(directory), example.id)
            write_once(path, {"id": example.id, "fingerprint": "config",
                              "example_sha256": digest(asdict(example))})
            self.assertEqual(load_matching(path, "config", example)["id"], "synthetic")
            with self.assertRaises(ValueError):
                load_matching(path, "different", example)

    def test_end_to_end_resume_and_model_cleanup(self):
        example = Example("synthetic", "Question", "Answer", (
            Step(0, "Planner", "First"), Step(1, "Solver", "Second"),
        ))
        opened, closed = [], []

        class FakeScorer:
            def __init__(self, *args, **kwargs):
                opened.append("utility")

            def __call__(self, context, answer):
                return float(len(context))

            def close(self):
                closed.append("utility")

        class FakeNLI:
            def __init__(self, *args, **kwargs):
                opened.append("risk")

            def probabilities(self, pairs):
                return [0.5] * len(pairs)

            def close(self):
                closed.append("risk")

        class FakeLocalizer:
            def __init__(self, *args, **kwargs):
                opened.append("localize")

            def generate(self, messages):
                return {"text": "Agent Name: Solver\nStep Number: 1\nReason for Mistake: Incorrect step.",
                        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}}

            def close(self):
                closed.append("localize")

        with tempfile.TemporaryDirectory() as temporary, contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(cli, "load_who_when", return_value=[(example, Label("Solver", 1))]))
            stack.enter_context(patch.object(cli, "checkpoint_identity", return_value={"name": "synthetic"}))
            stack.enter_context(patch.object(models, "AnswerScorer", FakeScorer))
            stack.enter_context(patch.object(models, "NLI", FakeNLI))
            stack.enter_context(patch.object(models, "Localizer", FakeLocalizer))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            root = Path(temporary)
            common = dict(benchmark="who_when", data="synthetic", model="synthetic",
                          limit=None, max_model_len=81920, memory=0.9, device="cpu")
            stages = [
                argparse.Namespace(**common, command="utility", output=str(root / "utility")),
                argparse.Namespace(**common, command="risk", output=str(root / "risk"), utility=str(root / "utility")),
                argparse.Namespace(**common, command="localize", output=str(root / "localize"),
                                   scores=str(root / "risk"), profile="qwen30", k=[2],
                                   arms=["direct", "mas_gate", "random", "shuffled"]),
            ]
            for stage in stages:
                cli.run(stage)
            self.assertEqual(opened, ["utility", "risk", "localize"])
            self.assertEqual(closed, opened)
            before = {p: p.read_bytes() for p in root.rglob("*.json")}
            for stage in stages:
                cli.run(stage)
            self.assertEqual(len(opened), 3)
            self.assertEqual(before, {p: p.read_bytes() for p in root.rglob("*.json")})
            for path in (root / "localize").glob("*/records/*.json"):
                self.assertTrue(cli.read_json(path)["metrics"]["exact_step"])


if __name__ == "__main__":
    unittest.main()
