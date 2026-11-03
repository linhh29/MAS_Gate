#!/usr/bin/env bash
# Default to the latest unified inter-node quantification; set this variable explicitly to select a legacy mode.
export QUANTIFICATION_INTER_NODE_MODE="${QUANTIFICATION_INTER_NODE_MODE:-unified_baseline}"
set -euo pipefail
MAS_GATE_ROOT=${MAS_GATE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
# shellcheck source=runtime_limits.sh
source "${MAS_GATE_ROOT}/verl/mas_gate_rl/runtime_limits.sh"
cd "${MAS_GATE_ROOT}"
export DATASET=${DATASET:-deepscaler}
export DEEPSCALER_JSON=${DEEPSCALER_JSON:-${MAS_GATE_ROOT}/datasets/DeepScaleR/deepscaler_train.json}
export VAL_RATIO=${VAL_RATIO:-0}
export MAX_SAMPLES=${MAX_SAMPLES:-0}
# Fixed cross-dataset training contract. Ignore stale values inherited from
# older shells so all three formal datasets use exactly 256 x 16.
if [[ -n "${TRAIN_BATCH_SIZE:-}" && "${TRAIN_BATCH_SIZE}" != "256" ]]; then
  echo "[DEEPSCALER] ignoring stale TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}; fixed value is 256" >&2
fi
if [[ -n "${ROLLOUT_N:-}" && "${ROLLOUT_N}" != "16" ]]; then
  echo "[DEEPSCALER] ignoring stale ROLLOUT_N=${ROLLOUT_N}; fixed value is 16" >&2
fi
export TRAIN_BATCH_SIZE=256
export DATA_PAD_TO_MULTIPLE=256
export PPO_MINI_BATCH_SIZE=256
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}
export ROLLOUT_N=16
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-16}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-196608}
export REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-8}
export MAS_GATE_MAX_CONCURRENCY=${MAS_GATE_MAX_CONCURRENCY:-128}
export MAS_GATE_PHASE_CONCURRENCY=${MAS_GATE_PHASE_CONCURRENCY:-128}
export SUB_AGENT_ENABLE_PREFIX_CACHING=0
export ROLLOUT_TP=${ROLLOUT_TP:-1}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-15000}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export CHECKPOINT_INTERMEDIATE_COUNT=${CHECKPOINT_INTERMEDIATE_COUNT:-10}
export SAVE_FREQ=${SAVE_FREQ:--1}
export TEST_FREQ=${TEST_FREQ:--1}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export MAS_GATE_REWARD_USE_QUANT=${MAS_GATE_REWARD_USE_QUANT:-1}
# Train on the group-normalized dense MAS impact plus strict mathematical
# source-answer correctness.
export MAS_GATE_REWARD_FUSION_ENABLED=1
export MAS_GATE_REWARD_FUSION_LAMBDA_CORRECT=1.0
export SUB_AGENT_MAX_MODEL_LEN=${SUB_AGENT_MAX_MODEL_LEN:-32768}
export MAS_GATE_SUB_AGENT_MAX_TOKENS=${MAS_GATE_SUB_AGENT_MAX_TOKENS:-8192}
export MAS_GATE_SUB_AGENT_MIN_COMPLETION_TOKENS=${MAS_GATE_SUB_AGENT_MIN_COMPLETION_TOKENS:-8192}
# Keep every asynchronous request phase on the eight-GPU lane at aggregate
# concurrency 128. Per-worker values are 16 because eight reward workers share
# the same lane (8 * 16 = 128). Qwen-J prompt log-probabilities materialize
# float32 logits, so the previous 1024-way contract exhausted H800 memory.
export MAS_GATE_SUB_AGENT_CONCURRENCY_PER_WORKER=${MAS_GATE_SUB_AGENT_CONCURRENCY_PER_WORKER:-16}
export MAS_GATE_REWARD_EXECUTOR_MAX_WORKERS=${MAS_GATE_REWARD_EXECUTOR_MAX_WORKERS:-16}
export VERL_REWARD_LOOP_EXECUTOR_MAX_WORKERS=${VERL_REWARD_LOOP_EXECUTOR_MAX_WORKERS:-${MAS_GATE_REWARD_EXECUTOR_MAX_WORKERS}}
export SUB_AGENT_VLLM_MAX_NUM_SEQS=${SUB_AGENT_VLLM_MAX_NUM_SEQS:-128}
export SUB_AGENT_MAX_NUM_SEQS=${SUB_AGENT_MAX_NUM_SEQS:-128}
export SUB_AGENT_MAX_NUM_BATCHED_TOKENS=${SUB_AGENT_MAX_NUM_BATCHED_TOKENS:-8192}
# Qwen-J computes prompt log-probabilities in float32. Bound only its scheduler
# workspace so a large 256 x 16 reward batch cannot admit enough long prompts
# simultaneously to exhaust device memory. This does not change output tokens.
export MAS_GATE_QWEN_PHASE_MAX_NUM_BATCHED_TOKENS=${MAS_GATE_QWEN_PHASE_MAX_NUM_BATCHED_TOKENS:-8192}
export MAS_GATE_QWEN_J_AGGREGATE_CONCURRENCY=${MAS_GATE_QWEN_J_AGGREGATE_CONCURRENCY:-128}
export MAS_GATE_QWEN_J_SERVER_MAX_NUM_SEQS=${MAS_GATE_QWEN_J_SERVER_MAX_NUM_SEQS:-128}
export SUB_AGENT_DISABLE_LOG_REQUESTS=${SUB_AGENT_DISABLE_LOG_REQUESTS:-1}
export QUANTIFICATION_SCORING_CONCURRENCY=${QUANTIFICATION_SCORING_CONCURRENCY:-16}
export QUANTIFICATION_EMBEDDING_CONCURRENCY=${QUANTIFICATION_EMBEDDING_CONCURRENCY:-16}
export EMBEDDING_GPU_MEMORY_UTILIZATION=${EMBEDDING_GPU_MEMORY_UTILIZATION:-0.90}
export EMBEDDING_MAX_NUM_SEQS=${EMBEDDING_MAX_NUM_SEQS:-128}
export EMBEDDING_MAX_NUM_BATCHED_TOKENS=${EMBEDDING_MAX_NUM_BATCHED_TOKENS:-196608}
export SUB_AGENT_GPU_MEMORY_UTILIZATION=${SUB_AGENT_GPU_MEMORY_UTILIZATION:-0.68}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.86}
export MAS_GATE_EMPTY_CACHE_BEFORE_ROLLOUT_WAKE=${MAS_GATE_EMPTY_CACHE_BEFORE_ROLLOUT_WAKE:-1}
export QUANTIFICATION_BACKEND=${QUANTIFICATION_BACKEND:-openai}
export QUANTIFICATION_SCORING_SERVED_MODEL=${QUANTIFICATION_SCORING_SERVED_MODEL:-Qwen3-30B-A3B-Instruct-2507}
export MAS_GATE_NLI_MODEL_PATH=${MAS_GATE_NLI_MODEL_PATH:-./models/bge-m3-zeroshot-v2.0}
# Normal training commits synchronously at each save point. Never inherit
# a stale shell setting that would trigger an unnecessary end-of-run scan.
# RAY_TMPDIR intentionally defaults to a short MAS-Gate-only /tmp path; Ray logs are copied to shared local storage at exit.
export RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8279}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-mas_gate_deepscaler_train_$(date +%Y%m%d_%H%M%S)}
exec bash verl/mas_gate_rl/run_mas_gate_grpo.sh "$@"
