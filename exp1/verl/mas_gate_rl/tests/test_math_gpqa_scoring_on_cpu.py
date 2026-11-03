from __future__ import annotations

from aime.gpqa.gpqa_score import extract_gpqa_answer
from aime.score import AIMEScorer
from mas_gate_rl.reward import _deepscaler_score


def test_aime_and_deepscaler_use_strict_math_equivalence() -> None:
    scorer = AIMEScorer()
    cases = [
        ("1", r"\boxed{1}", 1),
        ("1", "10", 0),
        ("1/2", r"\boxed{\frac{1}{2}}", 1),
        ("-2/3", r"\boxed{-\frac{2}{3}}", 1),
        ("0.8", r"\boxed{80\%}", 1),
        ("073", r"\boxed{73}", 1),
        ("504", r"\boxed{r^2+s^2=504}", 1),
        ("8", r"\boxed{2^3}", 1),
        ("2", r"\boxed{\sqrt{4}}", 1),
        ("2*pi", r"\boxed{2\pi}", 1),
        ("23", r"\boxed{\text{23}}", 1),
        (r"336^\circ", r"\boxed{336^\circ}", 1),
        ("504", r"\boxed{x=505}", 0),
        ("26", r"\boxed{\textbf{(A)} 26}", 0),
    ]
    for gold, output, expected in cases:
        assert scorer.calculate_score(gold, output)[0] == expected
        assert _deepscaler_score(gold, output)[0] == float(expected)


def test_last_box_wins_and_malformed_box_does_not_crash() -> None:
    scorer = AIMEScorer()
    assert scorer.calculate_score("2", r"work \boxed{1}; corrected \boxed{2}")[0] == 1
    assert scorer.calculate_score("2", r"unfinished \boxed{2")[0] == 0
    assert scorer.calculate_score("2", r"\boxed{2}; unfinished \boxed{")[0] == 0


def test_gpqa_extracts_only_explicit_final_option() -> None:
    cases = {
        r"\boxed{A}": "A",
        r"\boxed{\text{b}}": "B",
        "Final answer: c\n": "C",
        "reasoning mentions A, B, C\n(D)": "D",
        "reasoning mentions A, B, C, and D": "",
        r"\boxed{6}": "",
        r"first \boxed{A}; final \boxed{6}": "",
        r"\boxed{A}; unfinished \boxed{": "",
        r"\boxed{No valid option}": "",
    }
    for output, expected in cases.items():
        assert extract_gpqa_answer(output) == expected


def test_gpqa_prompt_contract_requires_uppercase_boxed_final_line() -> None:
    from aime.gpqa.prepare_gpqa import clean_question
    from building.skill_mas.build import _with_hard_instruction_constraints

    prompt = clean_question("Question?\nA. one\nB. two\nC. three\nD. four")
    constraints = _with_hard_instruction_constraints(
        "Compare the choices.", is_final=True, dataset_name="gpqa", role_name="final"
    )
    for text in (prompt, constraints):
        assert r"\boxed{A}" in text
        assert "uppercase" in text
    assert "option text" in prompt
    assert "No valid option" in constraints
