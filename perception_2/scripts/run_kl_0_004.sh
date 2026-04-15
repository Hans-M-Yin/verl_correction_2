set -x
ENGINE=${1:-vllm}

source "$(dirname "$0")/../configs/rl_config.sh"
export $(grep -v '^#' ../configs/reward_function_hyperparam.env | xargs)


DATE=$(date +%m%d)


EXP_NAME="${DATE}_kl_coef_0.004"
ROLLOUT_SAVE_PATH="./rollouts_saved/${EXP_NAME}"

KL_COEF=0.004

VLLM_USE_V1=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.actor.strategy="fsdp2" \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_PARALLEL_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTIL \
    actor_rollout_ref.rollout.multi_stage_wake_up=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    reward_model.use_reward_loop=True \
    reward_model.enable=True \
    reward_model.model.path=$RM_MODEL_PATH \
    reward_model.rollout.name=vllm \
    reward_model.rollout.prompt_length=1500 \
    reward_model.rollout.response_length=2000 \
    reward_model.rollout.gpu_memory_utilization=$RM_GPU_MEMORY_UTIL \
    reward_model.rollout.tensor_model_parallel_size=$TENSOR_PARALLEL_SIZE \
    reward_model.num_workers=4 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$N_GPU \
    trainer.nnodes=1 \
    trainer.total_epochs=$EPOCHS \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    +trainer.enable_trigger=True \
    +trainer.trigger_enable_schedule=True \
    +trainer.trigger_mode='part' \
    +trainer.trigger_warmup_ratio=$TRIGGER_WARMUP_RATIO \
    +trainer.trigger_start_ratio=$TRIGGER_START_RATIO \
    +trainer.trigger_final_ratio=$TRIGGER_FINAL_RATIO \
    +trainer.trigger_file_path=$TRIGGER_FILE \
    trainer.validation_data_dir="${ROLLOUT_SAVE_PATH}/val" \
    trainer.rollout_data_dir="${ROLLOUT_SAVE_PATH}/train" \
    custom_reward_function.path=perception_2/reward_function_loop.py \
    custom_reward_function.name=compute_score \
    $@