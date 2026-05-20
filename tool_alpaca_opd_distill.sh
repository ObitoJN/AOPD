#!/bin/bash

# 环境配置
PROJECT_PATH=$(pwd)

# 分布式环境参数（按需修改）
NNODES=${NNODES:-1}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
GPUS_PER_NODE=${GPUS_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$(($NNODES*$GPUS_PER_NODE))
PORT=6379

# 模型路径配置
POLICY_MODEL_PATH=/your/local/path
TEACHER_MODEL_NAME=Qwen3-32B
IP_POOL="['<teacher_ip_1>','<teacher_ip_2>']"

# 数据路径配置
TRAIN_DATA=/your/local/path
VAL_DATA=/your/local/path

# 实验名称与保存路径
EXTRA_NAME=Qwen3_8B_sft3000_CL_ToolAlpaca_EXOPD_Distill
SAVE_DIR=/your/local/path${EXTRA_NAME}

export RAY_memory_monitor_refresh_ms=0
export VLLM_USE_V1=1

TENSORBOARD_DIR="${SAVE_DIR}/tensorboard"
export TENSORBOARD_DIR="${TENSORBOARD_DIR}"
mkdir -p ${TENSORBOARD_DIR}



if [ "$NODE_RANK" == "0" ]; then
    echo "HEAD NODE"
    ray start --head --node-ip-address $MASTER_ADDR --port=${PORT} --num-gpus ${GPUS_PER_NODE}
    sleep 30

    ray job submit --runtime-env-json="{\"working_dir\": \"${PROJECT_PATH}\"}" \
    -- python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=none \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=${POLICY_MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    actor_rollout_ref.actor.distill_signal_in_loss=False \
    actor_rollout_ref.actor.distill_loss_type=low_var_kl \
    actor_rollout_ref.actor.distill_mode=opd \
    actor_rollout_ref.actor.distill.opd_weight=1.25 \
    actor_rollout_ref.actor.distill.gkd_weight=0.0 \
    actor_rollout_ref.actor.checkpoint.contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    +actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    ++actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20000 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
    actor_rollout_ref.rollout.max_num_batched_tokens=20000 \
    distill.enable=True \
    distill.model_name=${TEACHER_MODEL_NAME} \
    distill.ip_pool=${IP_POOL} \
    distill.top_k=1 \
    reward_model.enable=False \
    algorithm.kl_ctrl.kl_coef=0.0001 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='tool_alpaca_opd' \
    trainer.experiment_name=${EXTRA_NAME} \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.val_before_train=False \
    trainer.save_freq=10 \
    trainer.test_freq=-1 \
    trainer.total_epochs=6 \
    trainer.default_local_dir=${SAVE_DIR}

else
    echo "WORKER NODE"
    sleep 10
    ray start --block --address=${MASTER_ADDR}:${PORT} --num-gpus ${GPUS_PER_NODE}
fi
