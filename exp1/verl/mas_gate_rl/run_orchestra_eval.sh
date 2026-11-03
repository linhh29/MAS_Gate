#!/usr/bin/env bash
# Default to the latest unified inter-node quantification; set this variable explicitly to select a legacy mode.
export QUANTIFICATION_INTER_NODE_MODE="${QUANTIFICATION_INTER_NODE_MODE:-unified_baseline}"
# Common launcher for MAS-Gate evaluation. Do not invoke directly; use one of:
#   datasets/run_eval_bcp_orchestra.sh
#   datasets/run_eval_deepscaler_orchestra.sh
set -Eeuo pipefail
# Prevent multi-process Python/vLLM/Ray/NCCL crashes from filling shared /tmp with core dumps.
ulimit -c 0

ROOT=${MAS_GATE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
THIS_VERL="$ROOT/verl"
LAUNCHER_ROOT_PHYSICAL=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
CONFIGURED_ROOT_PHYSICAL=$(cd "$ROOT" 2>/dev/null && pwd -P) || {
  echo "[FATAL] MAS_GATE_ROOT does not exist: $ROOT" >&2
  exit 2
}
if [[ "$LAUNCHER_ROOT_PHYSICAL" != "$CONFIGURED_ROOT_PHYSICAL" ]]; then
  echo "[FATAL] launcher is not the configured MAS-Gate checkout: launcher=$LAUNCHER_ROOT_PHYSICAL root=$CONFIGURED_ROOT_PHYSICAL" >&2
  exit 2
fi
unset LAUNCHER_ROOT_PHYSICAL CONFIGURED_ROOT_PHYSICAL
: "${MAS_GATE_EVAL_DATASET:?MAS_GATE_EVAL_DATASET must be bcp, hotpotqa, aime, or gpqa}"
: "${MAS_GATE_EVAL_TEST_FILE:?MAS_GATE_EVAL_TEST_FILE is required}"
[[ "$MAS_GATE_EVAL_DATASET" == bcp || "$MAS_GATE_EVAL_DATASET" == hotpotqa || "$MAS_GATE_EVAL_DATASET" == aime || "$MAS_GATE_EVAL_DATASET" == gpqa ]] || { echo "bad dataset" >&2; exit 2; }
[[ -f "$MAS_GATE_EVAL_TEST_FILE" ]] || { echo "missing test file: $MAS_GATE_EVAL_TEST_FILE" >&2; exit 2; }

# Use the caller-selected GPU set; historical evaluations default to 0-7.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MAS_GATE_ROOT="$ROOT"
export PYTHONPATH="$THIS_VERL:$ROOT:$ROOT/building:$ROOT/datasets"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export NO_PROXY=127.0.0.1,localhost,::1
export no_proxy="$NO_PROXY"
export VLLM_SERVER_DEV_MODE=1
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Accept exactly one local HuggingFace directory, either positionally or through ORCHESTRA_MODEL_PATH.
if (( $# > 0 )) && [[ -z "${ORCHESTRA_MODEL_PATH:-}" ]]; then
  export ORCHESTRA_MODEL_PATH="$1"
  shift
fi
if (( $# > 0 )); then
  echo "[FATAL] unknown positional arguments: $*" >&2
  exit 2
fi
: "${ORCHESTRA_MODEL_PATH:?set ORCHESTRA_MODEL_PATH to a local HuggingFace directory}"
[[ -d "$ORCHESTRA_MODEL_PATH" ]] || { echo "[FATAL] local Orchestra model is not a directory: $ORCHESTRA_MODEL_PATH" >&2; exit 2; }

ORCHESTRA_PORT=${ORCHESTRA_PORT:-8201}
SUBAGENT_PORT=${SUBAGENT_PORT:-8200}
JUDGE_PORT=${JUDGE_PORT:-8203}
ORCHESTRA_MODEL_NAME=${ORCHESTRA_MODEL_NAME:-qwen3-8b}
SUBAGENT_MODEL_NAME=${SUBAGENT_MODEL_NAME:-Qwen3-30B-A3B-Instruct-2507}
SUBAGENT_MODEL_PATH=${SUBAGENT_MODEL_PATH:-./models/Qwen3-30B-A3B}
JUDGE_MODEL_NAME=${JUDGE_MODEL_NAME:-Qwen3-30B-A3B-Instruct-2507}
JUDGE_MODEL_PATH=${JUDGE_MODEL_PATH:-./models/Qwen3-30B-A3B}
# HotpotQA/BCP training and formal evaluation share one resident Qwen3-30B
# endpoint across execution and semantic-judge roles. Resolve the alias before
# port validation, monitoring, and cleanup so it is treated as one service.
if [[ "$MAS_GATE_EVAL_DATASET" == bcp || "$MAS_GATE_EVAL_DATASET" == hotpotqa ]]; then
  JUDGE_PORT="$SUBAGENT_PORT"
  JUDGE_MODEL_PATH="$SUBAGENT_MODEL_PATH"
  JUDGE_MODEL_NAME="$SUBAGENT_MODEL_NAME"
fi
ORCHESTRA_TP=${ORCHESTRA_TP:-8}
SUBAGENT_TP=${SUBAGENT_TP:-8}
JUDGE_TP=${JUDGE_TP:-8}
ORCHESTRA_CONCURRENCY=${ORCHESTRA_CONCURRENCY:-128}
EXECUTION_CONCURRENCY=${EXECUTION_CONCURRENCY:-128}
JUDGE_CONCURRENCY=${JUDGE_CONCURRENCY:-128}
ORCHESTRA_MAX_NUM_SEQS=${ORCHESTRA_MAX_NUM_SEQS:-128}
SUBAGENT_MAX_NUM_SEQS=${SUBAGENT_MAX_NUM_SEQS:-128}
JUDGE_MAX_NUM_SEQS=${JUDGE_MAX_NUM_SEQS:-128}
# BrowseComp-Plus uses long multi-turn retrieval workflows. 256 simultaneous
# workflows saturated the PPU runtime: most cases reached the per-sample 1800s
# deadline even though vLLM stayed alive. Keep 64-way fan-out (still high on
# four GPUs) and let the BCP-specific contract below override the generic lane.
BCP_CONCURRENCY=${BCP_CONCURRENCY:-64}
BCP_MAX_NUM_SEQS=${BCP_MAX_NUM_SEQS:-64}
BCP_SAMPLE_TIMEOUT=${BCP_SAMPLE_TIMEOUT:-3600}
# Inactive phase-exclusive servers must release both KV cache and model weights.
# The lane sweep may override memory fractions, but these defaults match the
# stable 30B settings used by BCP training.
VLLM_SLEEP_LEVEL=${VLLM_SLEEP_LEVEL:-1}
export VLLM_SLEEP_LEVEL
export ORCHESTRA_CONCURRENCY EXECUTION_CONCURRENCY JUDGE_CONCURRENCY
export ORCHESTRA_MAX_NUM_SEQS SUBAGENT_MAX_NUM_SEQS JUDGE_MAX_NUM_SEQS

# One evaluation process owns exactly one isolated eight-GPU lane. Capacity is
# selected by the training dataset contract: HotpotQA/BCP use 1024-way reward
# fan-out, while AIME/GPQA checkpoints trained on DeepScaler use 128-way fan-out
# and a 128-sequence resident Qwen scheduler.
_eval_die() { echo "[FATAL] evaluation capacity contract: $*" >&2; exit 2; }
IFS=',' read -ra _eval_gpus <<< "$CUDA_VISIBLE_DEVICES"
declare -A _eval_seen=()
for _eval_gpu in "${_eval_gpus[@]}"; do
  [[ "$_eval_gpu" =~ ^[0-9]+$ ]] || _eval_die "invalid CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  [[ -z "${_eval_seen[$_eval_gpu]:-}" ]] || _eval_die "duplicate GPU $_eval_gpu"
  _eval_seen[$_eval_gpu]=1
done
if [[ "${MAS_GATE_ALLOW_NON_TP8:-0}" == 1 ]]; then
  [[ ${#_eval_seen[@]} -ge 1 ]] || _eval_die "CUDA_VISIBLE_DEVICES must contain at least 1 GPU"
else
  [[ ${#_eval_seen[@]} -eq 8 ]] || _eval_die "CUDA_VISIBLE_DEVICES must contain exactly 8 GPUs, got $CUDA_VISIBLE_DEVICES"
fi
for _eval_name in ORCHESTRA_TP SUBAGENT_TP JUDGE_TP; do
  [[ ${!_eval_name} -eq ${#_eval_seen[@]} ]] || _eval_die "${_eval_name} must match visible GPU count ${#_eval_seen[@]}, got ${!_eval_name}"
done
if [[ "$MAS_GATE_EVAL_DATASET" == aime || "$MAS_GATE_EVAL_DATASET" == gpqa ]]; then
  declare -A _eval_expected=(
    [ORCHESTRA_CONCURRENCY]=256 [EXECUTION_CONCURRENCY]=256
    [ORCHESTRA_MAX_NUM_SEQS]=256 [SUBAGENT_MAX_NUM_SEQS]=256
  )
elif [[ "$MAS_GATE_EVAL_DATASET" == bcp ]]; then
  # BCP is latency-bound rather than throughput-bound. 64 concurrent workflows
  # avoids the systematic per-case timeout observed at 256 while retaining
  # substantial parallelism across all four visible GPUs.
  ORCHESTRA_CONCURRENCY="$BCP_CONCURRENCY"
  EXECUTION_CONCURRENCY="$BCP_CONCURRENCY"
  JUDGE_CONCURRENCY="$BCP_CONCURRENCY"
  ORCHESTRA_MAX_NUM_SEQS="$BCP_MAX_NUM_SEQS"
  SUBAGENT_MAX_NUM_SEQS="$BCP_MAX_NUM_SEQS"
  JUDGE_MAX_NUM_SEQS="$BCP_MAX_NUM_SEQS"
  export ORCHESTRA_CONCURRENCY EXECUTION_CONCURRENCY JUDGE_CONCURRENCY
  export ORCHESTRA_MAX_NUM_SEQS SUBAGENT_MAX_NUM_SEQS JUDGE_MAX_NUM_SEQS
  declare -A _eval_expected=(
    [ORCHESTRA_CONCURRENCY]="$BCP_CONCURRENCY" [EXECUTION_CONCURRENCY]="$BCP_CONCURRENCY" [JUDGE_CONCURRENCY]="$BCP_CONCURRENCY"
    [ORCHESTRA_MAX_NUM_SEQS]="$BCP_MAX_NUM_SEQS" [SUBAGENT_MAX_NUM_SEQS]="$BCP_MAX_NUM_SEQS" [JUDGE_MAX_NUM_SEQS]="$BCP_MAX_NUM_SEQS"
  )
else
  declare -A _eval_expected=(
    [ORCHESTRA_CONCURRENCY]=256 [EXECUTION_CONCURRENCY]=256 [JUDGE_CONCURRENCY]=256
    [ORCHESTRA_MAX_NUM_SEQS]=256 [SUBAGENT_MAX_NUM_SEQS]=256 [JUDGE_MAX_NUM_SEQS]=256
  )
fi
for _eval_name in "${!_eval_expected[@]}"; do
  [[ ${!_eval_name} -eq ${_eval_expected[$_eval_name]} ]] || \
    _eval_die "${_eval_name} must be ${_eval_expected[$_eval_name]}, got ${!_eval_name}"
done
if [[ "$MAS_GATE_EVAL_DATASET" == aime || "$MAS_GATE_EVAL_DATASET" == gpqa ]]; then
  [[ ${SUBAGENT_MAX_NUM_BATCHED_TOKENS:-0} -eq 12288 ]] || \
    _eval_die "SUBAGENT_MAX_NUM_BATCHED_TOKENS must be 12288 for DeepScaler-aligned evaluation, got ${SUBAGENT_MAX_NUM_BATCHED_TOKENS:-unset}"
  [[ ${SUBAGENT_GPU_MEMORY_UTILIZATION:-unset} == 0.76 ]] || \
    _eval_die "SUBAGENT_GPU_MEMORY_UTILIZATION must be 0.76 for DeepScaler-aligned evaluation, got ${SUBAGENT_GPU_MEMORY_UTILIZATION:-unset}"
fi
unset _eval_expected
unset _eval_gpus _eval_seen _eval_gpu _eval_name
[[ -f "$SUBAGENT_MODEL_PATH/config.json" ]] || { echo "missing local sub-agent: $SUBAGENT_MODEL_PATH" >&2; exit 2; }

STAMP=${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${EVAL_OUTPUT_DIR:-$ROOT/datasets/eval_results/mas_gate_${MAS_GATE_EVAL_DATASET}_${STAMP}}
mkdir -p "$OUTPUT_DIR/server_logs"
RUNTIME_LOG_DIR=${EVAL_RUNTIME_LOG_DIR:-/tmp/mas_gate_eval_runtime_${MAS_GATE_EVAL_DATASET}_${STAMP}_$$}
mkdir -p "$RUNTIME_LOG_DIR"
RESOLVED_JSON="$OUTPUT_DIR/resolved_orchestra_model.json"

python "$THIS_VERL/mas_gate_rl/eval_model_resolver.py" \
  --model-path "$ORCHESTRA_MODEL_PATH" --output-file "$RESOLVED_JSON"
ORCHESTRA_MODEL_PATH_RESOLVED=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_path"])' "$RESOLVED_JSON")
ORCHESTRA_MODEL_SOURCE=$(python -c 'import json,sys; x=json.load(open(sys.argv[1])); print(x["source"]+":"+x["source_id"])' "$RESOLVED_JSON")

echo "[MAS-Gate eval] dataset=$MAS_GATE_EVAL_DATASET"
echo "[MAS-Gate eval] test=$MAS_GATE_EVAL_TEST_FILE"
echo "[MAS-Gate eval] orchestra=$ORCHESTRA_MODEL_PATH_RESOLVED ($ORCHESTRA_MODEL_SOURCE)"
echo "[MAS-Gate eval] sub-agent=$SUBAGENT_MODEL_PATH"
echo "[MAS-Gate eval] GPUs=$CUDA_VISIBLE_DEVICES output=$OUTPUT_DIR"

port_free() {
  python - "$1" <<'PY'
import socket,sys
s=socket.socket(); s.settimeout(.3)
rc=s.connect_ex(('127.0.0.1',int(sys.argv[1])))
s.close(); raise SystemExit(0 if rc != 0 else 1)
PY
}
PORTS=("$ORCHESTRA_PORT" "$SUBAGENT_PORT" "$JUDGE_PORT")
declare -A _eval_port_seen=()
UNIQUE_PORTS=()
for p in "${PORTS[@]}"; do
  [[ -n "${_eval_port_seen[$p]:-}" ]] && continue
  _eval_port_seen[$p]=1
  UNIQUE_PORTS+=("$p")
  port_free "$p" || { echo "[FATAL] localhost:$p is already occupied; choose another port" >&2; exit 3; }
done
unset _eval_port_seen

# A candidate keeps one Orchestra/sub-agent vLLM process alive and uses
# /sleep + /wake_up between design, execution, and judge phases.  This avoids
# reloading the same checkpoint for every phase.  Set FRESH_PHASES=1 only as a
# compatibility fallback for a runtime where sleep/wake semantic health fails.
MAS_GATE_EVAL_FRESH_PHASES=${MAS_GATE_EVAL_FRESH_PHASES:-1}
[[ "$MAS_GATE_EVAL_FRESH_PHASES" == 0 || "$MAS_GATE_EVAL_FRESH_PHASES" == 1 ]] || \
  _eval_die "MAS_GATE_EVAL_FRESH_PHASES must be 0 or 1, got $MAS_GATE_EVAL_FRESH_PHASES"
if [[ "$MAS_GATE_EVAL_FRESH_PHASES" == 1 ]]; then
  echo "[MAS-Gate eval] lifecycle=fresh_phase_restart (design -> execute -> optional judge)"
  # Source rather than exec: the fresh backend consumes the validated shell
  # variables above, including resolved model paths and phase capacity.
  source "$THIS_VERL/mas_gate_rl/run_orchestra_eval_fresh_phases.sh"
  exit 0
fi
echo "[MAS-Gate eval] lifecycle=sleep_wake (explicit legacy opt-in)"

ORCH_PID=""; SUB_PID=""; JUDGE_PID=""; MON_PID=""
cleanup() {
  set +e
  echo "[cleanup] stopping MAS-Gate evaluation capacity monitor and vLLM servers"
  [[ -n "$MON_PID" ]] && kill -TERM "$MON_PID" 2>/dev/null || true
  [[ -n "$MON_PID" ]] && wait "$MON_PID" 2>/dev/null || true
  [[ -n "$JUDGE_PID" ]] && kill -TERM -- "-$JUDGE_PID" 2>/dev/null
  [[ -n "$SUB_PID" ]] && kill -TERM -- "-$SUB_PID" 2>/dev/null
  [[ -n "$ORCH_PID" ]] && kill -TERM -- "-$ORCH_PID" 2>/dev/null
  sleep 3
  [[ -n "$JUDGE_PID" ]] && kill -KILL -- "-$JUDGE_PID" 2>/dev/null
  [[ -n "$SUB_PID" ]] && kill -KILL -- "-$SUB_PID" 2>/dev/null
  [[ -n "$ORCH_PID" ]] && kill -KILL -- "-$ORCH_PID" 2>/dev/null
  [[ -n "$JUDGE_PID" ]] && wait "$JUDGE_PID" 2>/dev/null
  [[ -n "$SUB_PID" ]] && wait "$SUB_PID" 2>/dev/null
  [[ -n "$ORCH_PID" ]] && wait "$ORCH_PID" 2>/dev/null

  # A completed candidate must not race the next candidate while sockets or
  # spawned vLLM workers are still being reaped.
  local deadline=$((SECONDS + ${VLLM_CLEANUP_TIMEOUT:-120}))
  while (( SECONDS < deadline )); do
    local all_free=1
    for p in "${UNIQUE_PORTS[@]}"; do
      port_free "$p" || { all_free=0; break; }
    done
    if (( all_free )); then
      sleep "${VLLM_CLEANUP_STABILIZE_SEC:-15}"
      mkdir -p "$OUTPUT_DIR/server_logs"
      cp -f "$RUNTIME_LOG_DIR"/*.log "$OUTPUT_DIR/server_logs/" 2>/dev/null || true
      return 0
    fi
    sleep 1
  done
  echo "[cleanup] warning: evaluation ports did not all become free" >&2
  mkdir -p "$OUTPUT_DIR/server_logs"
  cp -f "$RUNTIME_LOG_DIR"/*.log "$OUTPUT_DIR/server_logs/" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_ready() {
  local port=$1 pid=$2 name=$3 log=$4 timeout=${5:-1800}
  local start=$SECONDS
  while (( SECONDS - start < timeout )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[FATAL] $name vLLM exited during startup" >&2
      tail -200 "$log" >&2 || true
      return 1
    fi
    if curl -fsS --max-time 5 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "[ready] $name port=$port pid=$pid"
      return 0
    fi
    sleep 5
  done
  echo "[FATAL] timeout waiting for $name vLLM" >&2
  tail -200 "$log" >&2 || true
  return 1
}

semantic_health() {
  local port=$1 model=$2 role=$3 stage=$4
  python3 "$THIS_VERL/mas_gate_rl/vllm_semantic_health.py" \
    --base-url "http://127.0.0.1:${port}/v1" --model "$model" \
    --role "$role" --stage "$stage"
}

sleep_server() {
  local port=$1 name=$2
  curl -fsS -X POST --max-time 600 "http://127.0.0.1:${port}/sleep?level=${VLLM_SLEEP_LEVEL:-1}" >/dev/null
  local i
  for i in $(seq 1 120); do
    if curl -fsS --max-time 5 "http://127.0.0.1:${port}/is_sleeping" | grep -q 'true'; then
      echo "[sleep] $name released GPU memory"
      return 0
    fi
    sleep 2
  done
  echo "[FATAL] $name did not enter sleep mode" >&2
  return 1
}

# Start Orchestra first, then sleep it before loading Qwen30B on the same eight GPUs.
setsid python -m vllm.entrypoints.openai.api_server \
  --model "$ORCHESTRA_MODEL_PATH_RESOLVED" \
  --served-model-name "$ORCHESTRA_MODEL_NAME" \
  --host 127.0.0.1 --port "$ORCHESTRA_PORT" \
  --tensor-parallel-size "${ORCHESTRA_TP}" \
  --dtype bfloat16 \
  --gpu-memory-utilization "${ORCHESTRA_GPU_MEMORY_UTILIZATION:-0.92}" \
  --max-model-len "${ORCHESTRA_MAX_MODEL_LEN:-12288}" \
  --max-num-seqs "${ORCHESTRA_MAX_NUM_SEQS:-128}" \
  --max-num-batched-tokens "${ORCHESTRA_MAX_NUM_BATCHED_TOKENS:-196608}" \
  --enable-sleep-mode --enforce-eager \
  --max-num-partial-prefills 1 --max-long-partial-prefills 1 --long-prefill-token-threshold 1024 \
  >"$RUNTIME_LOG_DIR/orchestra_vllm.log" 2>&1 &
ORCH_PID=$!
wait_ready "$ORCHESTRA_PORT" "$ORCH_PID" orchestra "$RUNTIME_LOG_DIR/orchestra_vllm.log"
semantic_health "$ORCHESTRA_PORT" "$ORCHESTRA_MODEL_NAME" orchestra initial
sleep_server "$ORCHESTRA_PORT" orchestra

setsid python -m vllm.entrypoints.openai.api_server \
  --model "$SUBAGENT_MODEL_PATH" \
  --served-model-name "$SUBAGENT_MODEL_NAME" \
  --host 127.0.0.1 --port "$SUBAGENT_PORT" \
  --tensor-parallel-size "${SUBAGENT_TP}" \
  --dtype bfloat16 \
  --gpu-memory-utilization "${SUBAGENT_GPU_MEMORY_UTILIZATION:-0.81}" \
  --max-model-len "${SUB_AGENT_MAX_MODEL_LEN:-32768}" \
  --max-num-seqs "${SUBAGENT_MAX_NUM_SEQS:-128}" \
  --max-num-batched-tokens "${SUBAGENT_MAX_NUM_BATCHED_TOKENS:-196608}" \
  --enable-sleep-mode --enforce-eager \
  --max-num-partial-prefills 1 --max-long-partial-prefills 1 --long-prefill-token-threshold 1024 \
  >"$RUNTIME_LOG_DIR/subagent_vllm.log" 2>&1 &
SUB_PID=$!
wait_ready "$SUBAGENT_PORT" "$SUB_PID" subagent "$RUNTIME_LOG_DIR/subagent_vllm.log"
semantic_health "$SUBAGENT_PORT" "$SUBAGENT_MODEL_NAME" sub_agent initial

# These envs are also consumed by training-consistent BCP/reward helpers.
export MAS_GATE_ORCHESTRA_BASE_URL="http://127.0.0.1:${ORCHESTRA_PORT}/v1"
export MAS_GATE_SUB_AGENT_BASE_URL="http://127.0.0.1:${SUBAGENT_PORT}/v1"
export MAS_GATE_BCP_JUDGE_BASE_URL="http://127.0.0.1:${JUDGE_PORT}/v1"
export MAS_GATE_BCP_JUDGE_MODEL_PATH="$JUDGE_MODEL_PATH"
export MAS_GATE_BCP_JUDGE_MAX_MODEL_LEN=${JUDGE_MAX_MODEL_LEN:-12288}
export MAS_GATE_BCP_JUDGE_MAX_TOKENS=${JUDGE_MAX_TOKENS:-1536}
export MAS_GATE_SUB_AGENT_MODEL_PATH="$SUBAGENT_MODEL_PATH"
export MAS_GATE_SUB_AGENT_MAX_MODEL_LEN=${SUB_AGENT_MAX_MODEL_LEN:-32768}
export MAS_GATE_SUB_AGENT_MAX_TOKENS=${SUB_AGENT_MAX_TOKENS:-8192}
export MAS_GATE_SUB_AGENT_MIN_COMPLETION_TOKENS=${SUB_AGENT_MIN_COMPLETION_TOKENS:-8192}
export MAS_GATE_BCP_BM25_INDEX=${MAS_GATE_BCP_BM25_INDEX:-$ROOT/datasets/BrowseComp-Plus/scripts_build_index/indexes/bm25}
export MAS_GATE_BCP_CORPUS_DIR=${MAS_GATE_BCP_CORPUS_DIR:-$ROOT/datasets/BrowseComp-Plus/browsecomp-plus-corpus}
export MAS_GATE_BCP_BM25_CACHE=${MAS_GATE_BCP_BM25_CACHE:-/tmp/mas_gate_bcp_bm25_cache.pkl}
export MAS_GATE_BCP_RETRIEVAL_TOPK=${MAS_GATE_BCP_RETRIEVAL_TOPK:-5}
export MAS_GATE_BCP_DOC_MAX_TOKENS=${MAS_GATE_BCP_DOC_MAX_TOKENS:-512}
export MAS_GATE_BCP_RETRIEVAL_MAX_ROUNDS=${MAS_GATE_BCP_RETRIEVAL_MAX_ROUNDS:-5}

# Persist capacity evidence per candidate. The process is lane/port scoped and
# cleanup stops it before the next checkpoint starts.
export ALL_GPUS="${ALL_GPUS:-${CUDA_VISIBLE_DEVICES}}"
export MAS_GATE_EVAL_CONCURRENCY="${MAS_GATE_EVAL_CONCURRENCY:-${ORCHESTRA_CONCURRENCY}}"
export MAS_GATE_ORCHESTRA_METRICS_URL="http://127.0.0.1:${ORCHESTRA_PORT}"
export MAS_GATE_SUB_AGENT_METRICS_URL="http://127.0.0.1:${SUBAGENT_PORT}"
export MAS_GATE_JUDGE_METRICS_URL="http://127.0.0.1:${JUDGE_PORT}"
export MAS_GATE_EVAL_MONITOR_ROOT="${MAS_GATE_EVAL_MONITOR_ROOT:-${OUTPUT_DIR}/monitor}"
python3 "$THIS_VERL/mas_gate_rl/monitor_eval_capacity.py" >>"$RUNTIME_LOG_DIR/capacity_monitor.log" 2>&1 &
MON_PID=$!

EXTRA_ARGS=()
[[ "${MAX_SAMPLES:-0}" != 0 ]] && EXTRA_ARGS+=(--max-samples "$MAX_SAMPLES")
[[ -n "${EVAL_INDICES:-}" ]] && EXTRA_ARGS+=(--indices "$EVAL_INDICES")
if [[ "${EVAL_RESUME:-1}" == 0 ]]; then
  EXTRA_ARGS+=(--no-resume)
else
  EXTRA_ARGS+=(--resume)
fi

python "$THIS_VERL/mas_gate_rl/evaluate_orchestra.py" \
  --dataset "$MAS_GATE_EVAL_DATASET" \
  --test-file "$MAS_GATE_EVAL_TEST_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --orchestra-model "$ORCHESTRA_MODEL_NAME" \
  --orchestra-model-source "$ORCHESTRA_MODEL_SOURCE" \
  --orchestra-base-url "$MAS_GATE_ORCHESTRA_BASE_URL" \
  --subagent-model "$SUBAGENT_MODEL_NAME" \
  --subagent-base-url "$MAS_GATE_SUB_AGENT_BASE_URL" \
  --judge-model "$JUDGE_MODEL_NAME" \
  --judge-base-url "$MAS_GATE_BCP_JUDGE_BASE_URL" \
  --judge-concurrency "${JUDGE_CONCURRENCY:-1024}" \
  --judge-max-tokens "${JUDGE_MAX_TOKENS:-1536}" \
  --judge-attempts "${JUDGE_ATTEMPTS:-5}" \
  --orchestra-concurrency "${ORCHESTRA_CONCURRENCY:-1024}" \
  --execution-concurrency "${EXECUTION_CONCURRENCY:-1024}" \
  --orchestra-max-tokens "${ORCHESTRA_MAX_TOKENS:-4096}" \
  --orchestra-attempts "${ORCHESTRA_ATTEMPTS:-3}" \
  --execution-attempts "${EXECUTION_ATTEMPTS:-2}" \
  --sample-timeout "${SAMPLE_TIMEOUT:-${BCP_SAMPLE_TIMEOUT:-1800}}" \
  --sleep-level "${VLLM_SLEEP_LEVEL:-1}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$RUNTIME_LOG_DIR/eval.log"

cp -f "$RUNTIME_LOG_DIR"/*.log "$OUTPUT_DIR/server_logs/" 2>/dev/null || true
echo "[DONE] summary: $OUTPUT_DIR/summary.json"
