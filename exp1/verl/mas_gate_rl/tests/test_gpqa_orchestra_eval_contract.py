from pathlib import Path

from mas_gate_rl.evaluate_orchestra import build_parser
from mas_gate_rl.xml_policy import build_orchestra_user_prompt


def test_gpqa_parser_prompt_and_launcher_contract():
    args = build_parser().parse_args([
        "--dataset", "gpqa", "--test-file", "/tmp/gpqa.jsonl", "--output-dir", "/tmp/out"
    ])
    assert args.orchestra_concurrency == 1024
    assert args.execution_concurrency == 1024
    prompt = build_orchestra_user_prompt("Question\nA. a\nB. b\nC. c\nD. d", "gpqa")
    assert "GPQA constraints" in prompt
    assert r"\boxed{A}" in prompt and r"\boxed{D}" in prompt
    root = Path(__file__).resolve().parents[3]
    launcher = (root / "datasets" / "run_eval_gpqa_orchestra.sh").read_text()
    assert "MAS_GATE_EVAL_DATASET=gpqa" in launcher
    assert "--evaluation-dataset gpqa --train-dataset deepscaler" in launcher
    assert "ORCHESTRA_CONCURRENCY=${ORCHESTRA_CONCURRENCY:-128}" in launcher
    assert "EXECUTION_CONCURRENCY=${EXECUTION_CONCURRENCY:-128}" in launcher
    assert "SUBAGENT_MAX_NUM_SEQS=${SUBAGENT_MAX_NUM_SEQS:-128}" in launcher
    assert "SUBAGENT_MAX_NUM_BATCHED_TOKENS=${SUBAGENT_MAX_NUM_BATCHED_TOKENS:-8192}" in launcher
    assert "SUBAGENT_GPU_MEMORY_UTILIZATION=${SUBAGENT_GPU_MEMORY_UTILIZATION:-0.68}" in launcher


def test_aime_launcher_uses_deepscaler_training_capacity():
    root = Path(__file__).resolve().parents[3]
    launcher = (root / "datasets" / "run_eval_deepscaler_orchestra.sh").read_text()
    assert "MAS_GATE_EVAL_DATASET=aime" in launcher
    assert "--evaluation-dataset aime --train-dataset deepscaler" in launcher
    assert "ORCHESTRA_CONCURRENCY=${ORCHESTRA_CONCURRENCY:-128}" in launcher
    assert "EXECUTION_CONCURRENCY=${EXECUTION_CONCURRENCY:-128}" in launcher
    assert "SUBAGENT_MAX_NUM_SEQS=${SUBAGENT_MAX_NUM_SEQS:-128}" in launcher
    assert "SUBAGENT_MAX_NUM_BATCHED_TOKENS=${SUBAGENT_MAX_NUM_BATCHED_TOKENS:-8192}" in launcher
    assert "SUBAGENT_GPU_MEMORY_UTILIZATION=${SUBAGENT_GPU_MEMORY_UTILIZATION:-0.68}" in launcher


def test_core_eval_capacity_contract_is_dataset_specific():
    root = Path(__file__).resolve().parents[3]
    launcher = (root / "verl" / "mas_gate_rl" / "run_orchestra_eval.sh").read_text()
    assert 'if [[ "$MAS_GATE_EVAL_DATASET" == aime || "$MAS_GATE_EVAL_DATASET" == gpqa ]]' in launcher
    assert "[ORCHESTRA_CONCURRENCY]=256 [EXECUTION_CONCURRENCY]=256" in launcher
    assert "[ORCHESTRA_MAX_NUM_SEQS]=256 [SUBAGENT_MAX_NUM_SEQS]=256" in launcher
    assert "SUBAGENT_MAX_NUM_BATCHED_TOKENS must be 12288 for DeepScaler-aligned evaluation" in launcher
    assert "SUBAGENT_GPU_MEMORY_UTILIZATION must be 0.76 for DeepScaler-aligned evaluation" in launcher
    assert "[ORCHESTRA_CONCURRENCY]=256 [EXECUTION_CONCURRENCY]=256 [JUDGE_CONCURRENCY]=256" in launcher
    assert "[ORCHESTRA_MAX_NUM_SEQS]=256 [SUBAGENT_MAX_NUM_SEQS]=256 [JUDGE_MAX_NUM_SEQS]=256" in launcher
