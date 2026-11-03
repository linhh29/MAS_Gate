import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
JUDGE = ROOT / "datasets" / "BrowseComp-Plus" / "judge.py"
spec = importlib.util.spec_from_file_location("bcp_judge_under_test", JUDGE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_parse_json_yes():
    out = mod._parse_judge_text(json.dumps({
        "extracted_final_answer": "Ada Lovelace", "reasoning": "same entity",
        "correct": "yes", "confidence": "97%",
    }))
    assert out["correct"] == "yes"
    assert out["extracted_final_answer"] == "Ada Lovelace"
    assert out["confidence"] == 97.0


def test_parse_key_value_no():
    out = mod._parse_judge_text("extracted_final_answer: X\nreasoning: mismatch\ncorrect: no\nconfidence: 80%")
    assert out["correct"] == "no"
    assert out["confidence"] == 80.0


def test_parse_empty_is_no():
    out = mod._parse_judge_text("")
    assert out["correct"] == "no"
    assert out["extracted_final_answer"] == "None"


def test_hotpotqa_prompt_is_semantic_and_bcp_prompt_is_unchanged():
    hotpot, hotpot_version = mod.grader_template_for("hotpotqa_test")
    bcp, bcp_version = mod.grader_template_for("bcp")
    assert hotpot_version == "hotpotqa_semantic_judge_v1"
    assert "Do not solve the question" in hotpot
    assert "yes/no sentence" in hotpot
    assert bcp_version == "BrowseComp-Plus/judge.py::GRADER_TEMPLATE"
    assert "small margin of error for numerical problems" in bcp
    assert "For HotpotQA" not in bcp


def test_missing_correct_field_is_parse_error_not_incorrect_answer():
    out = mod._parse_judge_text(json.dumps({
        "extracted_final_answer": "Paris",
        "reasoning": "looks right",
        "confidence": 100,
    }))
    assert out["correct"] == "no"
    assert out["parse_error"] is True
