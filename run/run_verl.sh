#!/bin/bash
set -eu

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_ROOT"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    N_GPUS=$(nvidia-smi -L | wc -l)
else
    N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi

unset ROCR_VISIBLE_DEVICES
set -x

# The model you want to train
MODEL_PATH="/workspace/models/qwen_3b_sft"
MODEL_PATH="/workspace/models/qwen25_3b_instruct"
# If you don't have the model downloaded, you can use the HF Hub ID: Qwen/Qwen2.5-3B-Instruct

# Log file
LOG_FILE="logs/run_verl_grpo_sum_mixed_$(date +%Y%m%d_%H%M%S).log"

# Run the command
{
    python3 -m verl.trainer.main_ppo \
      algorithm.adv_estimator=grpo \
      data.train_files=data/sum/sum_train_verl_rl_mixed_70_30.parquet \
      data.val_files=data/sum/sum_test_verl_rl.parquet \
      data.train_batch_size=256 \
      data.val_batch_size=512 \
      data.max_prompt_length=1024 \
      data.max_response_length=4096 \
      data.filter_overlong_prompts=True \
      data.truncation=error \
      actor_rollout_ref.model.path="$MODEL_PATH" \
      actor_rollout_ref.actor.optim.lr=1e-6 \
      actor_rollout_ref.model.use_remove_padding=True \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      actor_rollout_ref.actor.ppo_mini_batch_size=16 \
      actor_rollout_ref.actor.use_dynamic_bsz=True \
      actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20000 \
      actor_rollout_ref.actor.use_kl_loss=False \
      actor_rollout_ref.actor.kl_loss_coef=0.001 \
      actor_rollout_ref.actor.kl_loss_type=low_var_kl \
      actor_rollout_ref.actor.fsdp_config.param_offload=True \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
      actor_rollout_ref.ref.fsdp_config.param_offload=True \
      actor_rollout_ref.rollout.name=vllm \
      actor_rollout_ref.rollout.n=5 \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
      actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
      +actor_rollout_ref.rollout.add_bo_think_token=False \
      algorithm.kl_ctrl.kl_coef=0.001 \
      trainer.project_name=sum-grpo-rl4 \
      trainer.experiment_name=sft \
      trainer.total_epochs=1 \
      trainer.total_training_steps=200 \
      trainer.logger=['console','wandb'] \
      trainer.save_freq=20 \
      trainer.test_freq=5 \
      trainer.nnodes=1 \
      trainer.n_gpus_per_node=$N_GPUS \
      trainer.device=cuda
} 2>&1 | tee "$LOG_FILE"

JOB_EXIT_CODE=${PIPESTATUS[0]}

exit $JOB_EXIT_CODE
