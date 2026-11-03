#!/usr/bin/env bash
# Default to the latest unified inter-node quantification; set this variable explicitly to select a legacy mode.
export QUANTIFICATION_INTER_NODE_MODE="${QUANTIFICATION_INTER_NODE_MODE:-unified_baseline}"
# Phase-local MAS-Gate evaluator for PPU TP8 Qwen3.
#
# This backend deliberately does not call vLLM /sleep or /wake_up.  On the
# deployed PPU vLLM 0.10 stack both Qwen3-8B and Qwen3-MoE have produced
# deterministic semantic corruption after sleep/wake (healthy prompts become
# repeated punctuation or random tokens).  Each phase is therefore:
#   fresh model load -> deterministic semantic health gate -> phase work ->
#   terminate the complete process group -> verify the port/GPU owner is gone.
set -Eeuo pipefail
ulimit -c 0

required=(ROOT THIS_VERL OUTPUT_DIR ORCHESTRA_MODEL_PATH_RESOLVED ORCHESTRA_MODEL_SOURCE \
  ORCHESTRA_MODEL_NAME ORCHESTRA_PORT ORCHESTRA_TP SUBAGENT_MODEL_PATH SUBAGENT_MODEL_NAME \
  SUBAGENT_PORT SUBAGENT_TP JUDGE_MODEL_PATH JUDGE_MODEL_NAME JUDGE_PORT JUDGE_TP \
  MAS_GATE_EVAL_DATASET MAS_GATE_EVAL_TEST_FILE CUDA_VISIBLE_DEVICES RUNTIME_LOG_DIR)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "[FATAL] fresh phase evaluator missing $name" >&2; exit 2; }
done

export MAS_GATE_ORCHESTRA_BASE_URL="http://127.0.0.1:${ORCHESTRA_PORT}/v1"
export MAS_GATE_SUB_AGENT_BASE_URL="http://127.0.0.1:${SUBAGENT_PORT}/v1"
export MAS_GATE_BCP_JUDGE_BASE_URL="http://127.0.0.1:${JUDGE_PORT}/v1"
export MAS_GATE_BCP_JUDGE_MODEL_PATH="$JUDGE_MODEL_PATH"
export MAS_GATE_BCP_JUDGE_MAX_MODEL_LEN=${JUDGE_MAX_MODEL_LEN:-12288}
export MAS_GATE_BCP_JUDGE_MAX_TOKENS=${JUDGE_MAX_TOKENS:-1536}
export MAS_GATE_HOTPOTQA_JUDGE_BASE_URL="$MAS_GATE_BCP_JUDGE_BASE_URL"
export MAS_GATE_HOTPOTQA_JUDGE_MODEL_PATH="$JUDGE_MODEL_PATH"
export MAS_GATE_HOTPOTQA_JUDGE_MAX_MODEL_LEN="$MAS_GATE_BCP_JUDGE_MAX_MODEL_LEN"
export MAS_GATE_HOTPOTQA_JUDGE_MAX_TOKENS="$MAS_GATE_BCP_JUDGE_MAX_TOKENS"
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

ACTIVE_PID=""
ACTIVE_PORT=""
ACTIVE_NAME=""
MON_PID=""

port_free() {
  python - "$1" <<'PY'
import socket,sys
s=socket.socket(); s.settimeout(.3)
rc=s.connect_ex(('127.0.0.1',int(sys.argv[1])))
s.close(); raise SystemExit(0 if rc != 0 else 1)
PY
}

wait_ready() {
  local port=$1 pid=$2 name=$3 log=$4 timeout=${5:-1800}
  local start=$SECONDS
  while (( SECONDS - start < timeout )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      local rc=0
      wait "$pid" || rc=$?
      echo "[FATAL] $name vLLM exited during startup rc=$rc" >&2
      python - "$log" <<'PY' >&2 || true
from pathlib import Path
import sys
p=Path(sys.argv[1]); print(p.read_bytes()[-30000:].decode('utf-8','replace') if p.exists() else 'missing log')
PY
      return 1
    fi
    if curl -fsS --max-time 5 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "[ready:fresh] $name port=$port pid=$pid"
      return 0
    fi
    sleep 5
  done
  echo "[FATAL] timeout waiting for fresh $name vLLM" >&2
  return 1
}

sync_runtime_logs() {
  mkdir -p "$OUTPUT_DIR/server_logs"
  cp -f "$RUNTIME_LOG_DIR"/*.log "$OUTPUT_DIR/server_logs/" 2>/dev/null || true
}

stop_active() {
  set +e
  if [[ -n "$ACTIVE_PID" ]]; then
    echo "[fresh-stop] terminating $ACTIVE_NAME pid=$ACTIVE_PID port=$ACTIVE_PORT"
    kill -TERM -- "-$ACTIVE_PID" 2>/dev/null
    local deadline=$((SECONDS + ${VLLM_CLEANUP_TIMEOUT:-180}))
    while (( SECONDS < deadline )); do
      if port_free "$ACTIVE_PORT" && ! kill -0 "$ACTIVE_PID" 2>/dev/null; then break; fi
      sleep 2
    done
    kill -KILL -- "-$ACTIVE_PID" 2>/dev/null
    wait "$ACTIVE_PID" 2>/dev/null
    if ! port_free "$ACTIVE_PORT"; then
      echo "[FATAL] $ACTIVE_NAME port $ACTIVE_PORT remained occupied after targeted shutdown" >&2
      return 1
    fi
    sleep "${FRESH_PHASE_SETTLE_SEC:-15}"
  fi
  sync_runtime_logs
  ACTIVE_PID=""; ACTIVE_PORT=""; ACTIVE_NAME=""
  set -e
}

cleanup() {
  set +e
  [[ -n "$MON_PID" ]] && kill -TERM "$MON_PID" 2>/dev/null
  [[ -n "$MON_PID" ]] && wait "$MON_PID" 2>/dev/null
  stop_active
  sync_runtime_logs
}
trap cleanup EXIT INT TERM

semantic_health() {
  local port=$1 model=$2 role=$3
  python "$THIS_VERL/mas_gate_rl/vllm_semantic_health.py" \
    --base-url "http://127.0.0.1:${port}/v1" --model "$model" \
    --role "$role" --stage fresh_start | tee -a "$OUTPUT_DIR/server_logs/semantic_health.jsonl"
}

start_orchestra() {
  local log="$RUNTIME_LOG_DIR/orchestra_vllm.log"
  : > "$log"
  setsid python -m vllm.entrypoints.openai.api_server \
    --model "$ORCHESTRA_MODEL_PATH_RESOLVED" --served-model-name "$ORCHESTRA_MODEL_NAME" \
    --host 127.0.0.1 --port "$ORCHESTRA_PORT" --tensor-parallel-size "$ORCHESTRA_TP" \
    --dtype bfloat16 --gpu-memory-utilization "${ORCHESTRA_GPU_MEMORY_UTILIZATION:-0.92}" \
    --max-model-len "${ORCHESTRA_MAX_MODEL_LEN:-12288}" \
    --max-num-seqs "${ORCHESTRA_MAX_NUM_SEQS:-128}" \
    --max-num-batched-tokens "${ORCHESTRA_MAX_NUM_BATCHED_TOKENS:-196608}" \
    --enforce-eager --max-num-partial-prefills 1 --max-long-partial-prefills 1 \
    --long-prefill-token-threshold 1024 >"$log" 2>&1 &
  ACTIVE_PID=$!; ACTIVE_PORT=$ORCHESTRA_PORT; ACTIVE_NAME=orchestra
  wait_ready "$ACTIVE_PORT" "$ACTIVE_PID" "$ACTIVE_NAME" "$log"
  semantic_health "$ACTIVE_PORT" "$ORCHESTRA_MODEL_NAME" orchestra
}

start_subagent() {
  local log="$RUNTIME_LOG_DIR/subagent_vllm.log"
  : > "$log"
  setsid python -m vllm.entrypoints.openai.api_server \
    --model "$SUBAGENT_MODEL_PATH" --served-model-name "$SUBAGENT_MODEL_NAME" \
    --host 127.0.0.1 --port "$SUBAGENT_PORT" --tensor-parallel-size "$SUBAGENT_TP" \
    --dtype bfloat16 --gpu-memory-utilization "${SUBAGENT_GPU_MEMORY_UTILIZATION:-0.76}" \
    --max-model-len "${SUB_AGENT_MAX_MODEL_LEN:-32768}" \
    --max-num-seqs "${SUBAGENT_MAX_NUM_SEQS:-128}" \
    --max-num-batched-tokens "${SUBAGENT_MAX_NUM_BATCHED_TOKENS:-12288}" \
    --enforce-eager --max-num-partial-prefills 1 --max-long-partial-prefills 1 \
    --long-prefill-token-threshold 1024 >"$log" 2>&1 &
  ACTIVE_PID=$!; ACTIVE_PORT=$SUBAGENT_PORT; ACTIVE_NAME=subagent
  wait_ready "$ACTIVE_PORT" "$ACTIVE_PID" "$ACTIVE_NAME" "$log"
  semantic_health "$ACTIVE_PORT" "$SUBAGENT_MODEL_NAME" subagent
}

start_judge() {
  local log="$RUNTIME_LOG_DIR/judge_vllm.log"
  : > "$log"
  setsid python -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_MODEL_PATH" --served-model-name "$JUDGE_MODEL_NAME" \
    --host 127.0.0.1 --port "$JUDGE_PORT" --tensor-parallel-size "$JUDGE_TP" \
    --dtype bfloat16 --gpu-memory-utilization "${JUDGE_GPU_MEMORY_UTILIZATION:-0.76}" \
    --max-model-len "${JUDGE_MAX_MODEL_LEN:-12288}" \
    --max-num-seqs "${JUDGE_MAX_NUM_SEQS:-128}" \
    --max-num-batched-tokens "${JUDGE_MAX_NUM_BATCHED_TOKENS:-12288}" \
    --enforce-eager --disable-log-requests >"$log" 2>&1 &
  ACTIVE_PID=$!; ACTIVE_PORT=$JUDGE_PORT; ACTIVE_NAME=judge
  wait_ready "$ACTIVE_PORT" "$ACTIVE_PID" "$ACTIVE_NAME" "$log"
  semantic_health "$ACTIVE_PORT" "$JUDGE_MODEL_NAME" judge
}

EXTRA_ARGS=()
[[ "${MAX_SAMPLES:-0}" != 0 ]] && EXTRA_ARGS+=(--max-samples "$MAX_SAMPLES")
[[ -n "${EVAL_INDICES:-}" ]] && EXTRA_ARGS+=(--indices "$EVAL_INDICES")
if [[ "${EVAL_RESUME:-1}" == 0 ]]; then EXTRA_ARGS+=(--no-resume); else EXTRA_ARGS+=(--resume); fi

run_phase() {
  local phase=$1
  echo "[fresh-phase] begin $phase"
  python "$THIS_VERL/mas_gate_rl/evaluate_orchestra.py" \
    --dataset "$MAS_GATE_EVAL_DATASET" --test-file "$MAS_GATE_EVAL_TEST_FILE" \
    --output-dir "$OUTPUT_DIR" --orchestra-model "$ORCHESTRA_MODEL_NAME" \
    --orchestra-model-source "$ORCHESTRA_MODEL_SOURCE" \
    --orchestra-base-url "$MAS_GATE_ORCHESTRA_BASE_URL" \
    --subagent-model "$SUBAGENT_MODEL_NAME" --subagent-base-url "$MAS_GATE_SUB_AGENT_BASE_URL" \
    --judge-model "$JUDGE_MODEL_NAME" --judge-base-url "$MAS_GATE_BCP_JUDGE_BASE_URL" \
    --judge-concurrency "${JUDGE_CONCURRENCY:-128}" --judge-max-tokens "${JUDGE_MAX_TOKENS:-1536}" \
    --judge-attempts "${JUDGE_ATTEMPTS:-5}" --orchestra-concurrency "${ORCHESTRA_CONCURRENCY:-128}" \
    --execution-concurrency "${EXECUTION_CONCURRENCY:-128}" \
    --orchestra-max-tokens "${ORCHESTRA_MAX_TOKENS:-4096}" \
    --orchestra-attempts "${ORCHESTRA_ATTEMPTS:-3}" --execution-attempts "${EXECUTION_ATTEMPTS:-2}" \
    --sample-timeout "${SAMPLE_TIMEOUT:-1800}" --sleep-level "${VLLM_SLEEP_LEVEL:-2}" \
    --phase "$phase" --service-lifecycle fresh_phase_restart "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "$OUTPUT_DIR/eval.log"
  echo "[fresh-phase] complete $phase"
}

# The monitor tolerates inactive phase endpoints and records the active TP8
# server's queue/memory throughout all three fresh processes.
export ALL_GPUS="${ALL_GPUS:-$CUDA_VISIBLE_DEVICES}"
export MAS_GATE_EVAL_CONCURRENCY="${MAS_GATE_EVAL_CONCURRENCY:-${ORCHESTRA_CONCURRENCY}}"
export MAS_GATE_ORCHESTRA_METRICS_URL="http://127.0.0.1:${ORCHESTRA_PORT}"
export MAS_GATE_SUB_AGENT_METRICS_URL="http://127.0.0.1:${SUBAGENT_PORT}"
export MAS_GATE_JUDGE_METRICS_URL="http://127.0.0.1:${JUDGE_PORT}"
export MAS_GATE_EVAL_MONITOR_ROOT="${MAS_GATE_EVAL_MONITOR_ROOT:-${OUTPUT_DIR}/monitor}"
python "$THIS_VERL/mas_gate_rl/monitor_eval_capacity.py" >>"$OUTPUT_DIR/server_logs/capacity_monitor.log" 2>&1 &
MON_PID=$!

start_orchestra
run_phase design
stop_active

start_subagent
run_phase execute
stop_active

if [[ "$MAS_GATE_EVAL_DATASET" == bcp || "$MAS_GATE_EVAL_DATASET" == hotpotqa ]]; then
  start_judge
  run_phase judge
  stop_active
fi

echo "[DONE:fresh-phases] summary: $OUTPUT_DIR/summary.json"
