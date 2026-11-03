#!/usr/bin/env bash
# Isolated HotpotQA GRPO ablation: binary accuracy only. Default launcher untouched.
set -euo pipefail
ROOT=${MAS_GATE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$ROOT"
export DATASET=hotpotqa
export HOTPOTQA_JSONL=${HOTPOTQA_JSONL:-$ROOT/datasets/BrowseComp-Plus/hotpotqa/data/hotpotqa_train.jsonl}
export DATA_DIR=${DATA_DIR:-$ROOT/verl/data/mas_gate_hotpotqa_b256_r16}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-256}
export ROLLOUT_N=${ROLLOUT_N:-16}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-15000}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-20}
# Ablation target is exact optimizer-update count; keep default training untouched.
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-20}
export MAS_GATE_ABLATION_TAG=MAS_GATE_ABLATION_HOTPOTQA_ACC_01
export MAS_GATE_REWARD_FUSION_ENABLED=0
export MAS_GATE_REWARD_METRIC_KEY=hotpotqa_accuracy_01
export VLLM_USE_V1=${VLLM_USE_V1:-1}
[[ -f "$DATA_DIR/train.parquet" ]] || python3 "$ROOT/verl/mas_gate_rl/data_preprocess.py" --dataset hotpotqa --jsonl "$HOTPOTQA_JSONL" --local_save_dir "$DATA_DIR" --val-ratio 0 --pad-to-multiple "$TRAIN_BATCH_SIZE"
export PYTHONPATH="$ROOT/quantification/vllm_patches:$ROOT/verl:$ROOT:$ROOT/building:$ROOT/datasets:${PYTHONPATH:-}"
exec python3 -m verl.trainer.main_ppo \
  hydra.run.dir="${HYDRA_RUN_DIR:-/tmp/mas_gate_hydra_hotpotqa_acc_01}" hydra.output_subdir=null \
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
  "data.train_files=[$DATA_DIR/train.parquet]" "data.val_files=[$DATA_DIR/test.parquet]" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" data.val_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" data.max_response_length="$MAX_RESPONSE_LENGTH" \
  +data.apply_chat_template_kwargs.enable_thinking=False data.filter_overlong_prompts=True data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH:-./models/Qwen3-8B}" \
  actor_rollout_ref.model.lora_rank=0 actor_rollout_ref.model.lora_adapter_path=null \
  actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-1e-6}" actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}" actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.clip_ratio=0.2 actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}" \
  actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.entropy_coeff=0 \
  'actor_rollout_ref.actor.checkpoint.save_contents=[hf_model]' actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.25}" \
  actor_rollout_ref.rollout.enforce_eager=True actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS:-16}" actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-32768}" \
  actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN:-24576}" \
  actor_rollout_ref.rollout.enable_prefix_caching=False actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  reward.custom_reward_function.path="$ROOT/verl/mas_gate_rl/reward.py" reward.custom_reward_function.name=compute_score \
  reward.reward_manager.name=naive reward.num_workers="${REWARD_NUM_WORKERS:-4}" trainer.balance_batch=True \
  trainer.val_before_train=False 'trainer.logger=["console","file"]' trainer.project_name=mas_gate_verl \
  trainer.experiment_name="${EXPERIMENT_NAME:-hotpotqa_acc_01_8gpu_20epoch}" trainer.n_gpus_per_node=8 trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ:-20}" trainer.test_freq=-1 trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.resume_mode=disable trainer.default_local_dir="${RUN_ARTIFACT_ROOT:-$ROOT/outputs/hotpotqa_acc_01_8gpu_20epoch}"
