from pathlib import Path

import pytest

from building.formalize.orchestra import assemble_graph
from building.formalize.schema import AgentSpec, AgentSpecs, SubTask, TaskDecomposition
from mas_gate_rl.evaluate_orchestra import _question_sha256, _sha256_file, _validate_artifact_identity
from mas_gate_rl.xml_policy import build_orchestra_user_prompt, graph_from_rollout_xml


def _decomposition() -> TaskDecomposition:
    return TaskDecomposition(
        goal="Resolve two linked facts and answer briefly.",
        sub_tasks=[
            SubTask(idx=0, desc="Retrieve supporting documents for both hops."),
            SubTask(idx=1, desc="Synthesize the exact short answer."),
        ],
    )


def _specs(first_mode: str | None = "multi_turn_search") -> AgentSpecs:
    return AgentSpecs(
        specs=[
            AgentSpec(
                id="agent_0", role="Evidence retriever", sub_task_idx=0,
                dependencies=[], is_final=False, execution_mode=first_mode,
            ),
            AgentSpec(
                id="agent_1", role="Final synthesizer", sub_task_idx=1,
                dependencies=["agent_0"], is_final=True, execution_mode="llm_only",
            ),
        ]
    )


def test_hotpotqa_preserves_explicit_search_and_aime_remains_llm_only():
    hotpot = assemble_graph(_decomposition(), _specs(), "hotpotqa")
    assert [node.execution_mode for node in hotpot.nodes] == ["multi_turn_search", "llm_only"]

    aime = assemble_graph(_decomposition(), _specs(), "aime")
    assert {node.execution_mode for node in aime.nodes} == {"llm_only"}


def test_hotpotqa_legacy_graph_gets_search_without_changing_bcp_contract():
    hotpot = assemble_graph(_decomposition(), _specs(first_mode=None), "hotpotqa")
    bcp = assemble_graph(_decomposition(), _specs(first_mode=None), "bcp")
    assert any(node.execution_mode == "multi_turn_search" for node in hotpot.nodes)
    assert any(node.execution_mode == "multi_turn_search" for node in bcp.nodes)
    assert hotpot.nodes[-1].execution_mode == "llm_only"
    assert bcp.nodes[-1].execution_mode == "llm_only"


def test_direct_mas_graph_is_sanitized_by_dataset_capability():
    xml = """<mas_graph topology="dag"><nodes>
    <agent id="agent_0" is_final="false" execution_mode="multi_turn_search"><role>Retrieve evidence</role><task>Search documents</task><goal>Answer</goal></agent>
    <agent id="agent_1" is_final="true" execution_mode="llm_only"><role>Answer</role><task>Synthesize</task><goal>Answer</goal></agent>
    </nodes><edges><edge f="agent_0" t="agent_1" type="data" /></edges></mas_graph>"""
    hotpot = graph_from_rollout_xml(xml, "question", "hotpotqa")
    aime = graph_from_rollout_xml(xml, "question", "aime")
    assert hotpot.nodes[0].execution_mode == "multi_turn_search"
    assert {node.execution_mode for node in aime.nodes} == {"llm_only"}


def test_dataset_prompts_are_capability_specific():
    hotpot = build_orchestra_user_prompt("Who is linked to whom?", "hotpotqa")
    bcp = build_orchestra_user_prompt("Find an entity", "bcp")
    aime = build_orchestra_user_prompt("Compute x", "aime")
    assert "HotpotQA routing constraints" in hotpot
    assert "multi-hop" in hotpot and "exact short answer" in hotpot
    assert "Math constraints" not in hotpot and "\\boxed" not in hotpot
    assert "BrowseComp-Plus routing constraints" in bcp
    assert "Math constraints" in aime and "\\boxed" in aime


def test_phase_artifact_identity_and_file_hash_fail_closed(tmp_path: Path):
    dataset = tmp_path / "test.jsonl"
    dataset.write_text('{"id":"q1","question":"Question one","answer":"A"}\n', encoding="utf-8")
    first_hash = _sha256_file(dataset)
    artifact = {
        "index": 0,
        "task_id": "q1",
        "question": "Question one",
        "question_sha256": _question_sha256("Question one"),
    }
    _validate_artifact_identity(
        artifact, idx=0, task_id="q1", question="Question one", path=tmp_path / "0.json"
    )
    with pytest.raises(RuntimeError, match="question mismatch"):
        _validate_artifact_identity(
            artifact, idx=0, task_id="q1", question="Different question", path=tmp_path / "0.json"
        )
    dataset.write_text('{"id":"q2","question":"Question two","answer":"B"}\n', encoding="utf-8")
    assert _sha256_file(dataset) != first_hash


def test_hotpotqa_semantic_judge_rejects_remote_endpoint(monkeypatch: pytest.MonkeyPatch):
    import asyncio
    from mas_gate_rl.reward import _run_semantic_llm_judge

    monkeypatch.setenv("MAS_GATE_HOTPOTQA_JUDGE_BASE_URL", "https://example.com/v1")
    with pytest.raises(RuntimeError, match="loopback"):
        asyncio.run(_run_semantic_llm_judge(
            dataset_name="hotpotqa",
            question="Question?",
            response="Answer",
            correct_answer="Answer",
        ))


def test_hotpotqa_judge_parser_defaults_and_fail_closed_contract():
    from mas_gate_rl.evaluate_orchestra import build_parser

    args = build_parser().parse_args([
        "--dataset", "hotpotqa",
        "--test-file", "/tmp/test.jsonl",
        "--output-dir", "/tmp/out",
    ])
    assert args.orchestra_concurrency == 1024
    assert args.execution_concurrency == 1024
    assert args.judge_concurrency == 1024

    source = (Path(__file__).resolve().parents[1] / "evaluate_orchestra.py").read_text(encoding="utf-8")
    assert '"score": (None if semantic_judge_dataset else int(score))' in source
    assert '"exact_score": int(score) if dataset in {"bcp", "hotpotqa"} else None' in source
    assert '"scoring_method": f"{dataset}_llm_judge_failed"' in source
    assert '"score": None' in source
    assert 'if not semantic_judge_dataset:' in source


def test_hotpotqa_launchers_match_training_client_and_server_capacity():
    root = Path(__file__).resolve().parents[3]
    launcher = (root / "datasets" / "run_eval_hotpotqa_orchestra.sh").read_text(encoding="utf-8")
    fresh = (root / "verl" / "mas_gate_rl" / "run_orchestra_eval_fresh_phases.sh").read_text(encoding="utf-8")
    training = (root / "verl" / "mas_gate_rl" / "run_hotpotqa_train.sh").read_text(encoding="utf-8")
    expected = {
        "ORCHESTRA_CONCURRENCY": 1024, "EXECUTION_CONCURRENCY": 1024, "JUDGE_CONCURRENCY": 1024,
        "ORCHESTRA_MAX_NUM_SEQS": 128, "SUBAGENT_MAX_NUM_SEQS": 1024, "JUDGE_MAX_NUM_SEQS": 1024,
    }
    for name, value in expected.items():
        assert f"{name}=${{{name}:-{value}}}" in launcher
    for name in ("ORCHESTRA", "SUBAGENT", "JUDGE"):
        assert f"{name}_MAX_NUM_BATCHED_TOKENS=${{{name}_MAX_NUM_BATCHED_TOKENS:-196608}}" in launcher
    assert '[[ "$MAS_GATE_EVAL_DATASET" == bcp || "$MAS_GATE_EVAL_DATASET" == hotpotqa ]]' in fresh
    assert "START_BCP_JUDGE_SERVER=${START_BCP_JUDGE_SERVER:-1}" in training
    assert "MAS_GATE_BCP_LLM_JUDGE_ENABLED=${MAS_GATE_BCP_LLM_JUDGE_ENABLED:-1}" in training
