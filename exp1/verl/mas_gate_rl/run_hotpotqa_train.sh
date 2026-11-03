#!/usr/bin/env bash
# Default to the latest unified inter-node quantification; set this variable explicitly to select a legacy mode.
export QUANTIFICATION_INTER_NODE_MODE="${QUANTIFICATION_INTER_NODE_MODE:-unified_baseline}"
set -euo pipefail
ROOT=${MAS_GATE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "${ROOT}"
export DATASET=${DATASET:-hotpotqa}
export HOTPOTQA_JSONL=${HOTPOTQA_JSONL:-${ROOT}/datasets/BrowseComp-Plus/hotpotqa/data/hotpotqa_train.jsonl}
export MAS_GATE_HOTPOTQA_BM25_INDEX=${MAS_GATE_HOTPOTQA_BM25_INDEX:-${ROOT}/datasets/BrowseComp-Plus/hotpotqa/indexes/bm25}
export DATA_DIR=${DATA_DIR:-${ROOT}/verl/data/mas_gate_hotpotqa_b256_r16}
export VAL_RATIO=${VAL_RATIO:-0}
# Fixed HotpotQA full-run contract.  Older startup shells may export the old
# BCP values; do not let those stale values silently change this experiment.
if [[ -n "${TRAIN_BATCH_SIZE:-}" && "${TRAIN_BATCH_SIZE}" != "256" ]]; then
  echo "[HOTPOTQA] ignoring stale TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}; fixed value is 256" >&2
fi
if [[ -n "${ROLLOUT_N:-}" && "${ROLLOUT_N}" != "16" ]]; then
  echo "[HOTPOTQA] ignoring stale ROLLOUT_N=${ROLLOUT_N}; fixed value is 16" >&2
fi
export TRAIN_BATCH_SIZE=256
export ROLLOUT_N=16
if [[ -n "${DATA_PAD_TO_MULTIPLE:-}" && "${DATA_PAD_TO_MULTIPLE}" != "${TRAIN_BATCH_SIZE}" ]]; then
  echo "[HOTPOTQA] ignoring stale DATA_PAD_TO_MULTIPLE=${DATA_PAD_TO_MULTIPLE}; fixed value is ${TRAIN_BATCH_SIZE}" >&2
fi
export DATA_PAD_TO_MULTIPLE=${TRAIN_BATCH_SIZE}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-256}
export CHECKPOINT_INTERMEDIATE_COUNT=${CHECKPOINT_INTERMEDIATE_COUNT:-10}
export SAVE_FREQ=${SAVE_FREQ:--1}
export REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-4}
export MAS_GATE_MAX_CONCURRENCY=${MAS_GATE_MAX_CONCURRENCY:-1024}
export MAS_GATE_PHASE_CONCURRENCY=${MAS_GATE_PHASE_CONCURRENCY:-1024}
export SUB_AGENT_ENABLE_PREFIX_CACHING=0
export MAS_GATE_SUB_AGENT_CONCURRENCY_PER_WORKER=${MAS_GATE_SUB_AGENT_CONCURRENCY_PER_WORKER:-128}
export MAS_GATE_REWARD_EXECUTOR_MAX_WORKERS=${MAS_GATE_REWARD_EXECUTOR_MAX_WORKERS:-128}
export VERL_REWARD_LOOP_EXECUTOR_MAX_WORKERS=${VERL_REWARD_LOOP_EXECUTOR_MAX_WORKERS:-128}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-128}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-196608}
export SUB_AGENT_MAX_NUM_SEQS=${SUB_AGENT_MAX_NUM_SEQS:-1024}
export SUB_AGENT_VLLM_MAX_NUM_SEQS=${SUB_AGENT_VLLM_MAX_NUM_SEQS:-1024}
export SUB_AGENT_MAX_NUM_BATCHED_TOKENS=${SUB_AGENT_MAX_NUM_BATCHED_TOKENS:-196608}
export EMBEDDING_MAX_NUM_SEQS=${EMBEDDING_MAX_NUM_SEQS:-1024}
export EMBEDDING_MAX_NUM_BATCHED_TOKENS=${EMBEDDING_MAX_NUM_BATCHED_TOKENS:-196608}
export JUDGE_MAX_NUM_SEQS=${JUDGE_MAX_NUM_SEQS:-1024}
export JUDGE_MAX_NUM_BATCHED_TOKENS=${JUDGE_MAX_NUM_BATCHED_TOKENS:-196608}
export QUANTIFICATION_SCORING_CONCURRENCY=${QUANTIFICATION_SCORING_CONCURRENCY:-128}
export QUANTIFICATION_EMBEDDING_CONCURRENCY=${QUANTIFICATION_EMBEDDING_CONCURRENCY:-128}
# Train on the group-normalized dense MAS impact plus the authoritative
# HotpotQA semantic/source correctness signal.
export MAS_GATE_REWARD_FUSION_ENABLED=${MAS_GATE_REWARD_FUSION_ENABLED:-1}
export MAS_GATE_REWARD_FUSION_LAMBDA_CORRECT=1.0
# HotpotQA uses the same Qwen3-30B semantic Judge contract as BCP; EM/F1 remain diagnostics.
export START_BCP_JUDGE_SERVER=${START_BCP_JUDGE_SERVER:-1}
export MAS_GATE_BCP_LLM_JUDGE_ENABLED=${MAS_GATE_BCP_LLM_JUDGE_ENABLED:-1}
[[ -f "${HOTPOTQA_JSONL}" ]] || { echo "[HOTPOTQA] missing train file: ${HOTPOTQA_JSONL}" >&2; exit 2; }
[[ -d "${MAS_GATE_HOTPOTQA_BM25_INDEX}" ]] || { echo "[HOTPOTQA] missing BM25 index: ${MAS_GATE_HOTPOTQA_BM25_INDEX}" >&2; exit 2; }
read -r _hotpot_rows < <(wc -l < "${HOTPOTQA_JSONL}")
[[ "${_hotpot_rows}" == "10000" ]] || { echo "[HOTPOTQA] expected the seed=0 10k train subset, got ${_hotpot_rows}" >&2; exit 2; }
echo "[HOTPOTQA] train_rows=${_hotpot_rows} batch=${TRAIN_BATCH_SIZE} rollout_n=${ROLLOUT_N} expected_steps=$((_hotpot_rows / TRAIN_BATCH_SIZE + (_hotpot_rows % TRAIN_BATCH_SIZE != 0))) reward=group_z(mean_impact)+1.0*group_z(semantic_source_correctness) test_file=${ROOT}/datasets/BrowseComp-Plus/hotpotqa/data/hotpotqa_test.jsonl"
exec bash "${ROOT}/verl/mas_gate_rl/run_mas_gate_grpo.sh" "$@"
