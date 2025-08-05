#!/bin/bash
set -ex

# ==================================
#       MODIFICATION 1: OpenAI API Key
# ==================================
export OPENAI_API_KEY=${OPENAI_API_KEY:-"sk-your-key-here"}
if [[ "$OPENAI_API_KEY" == "sk-your-key-here" ]]; then
    echo "Please set your OPENAI_API_KEY environment variable."
    exit 1
fi

DATA_HOME="/data/post_train/data/"
MODEL_HOME="/data/post_train/models/"

MEGATRON_CKPT_PATH="${MODEL_HOME}/Qwen3-4B_slime_yenting"

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-4B.sh"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_HOME}/Qwen3-4B"
   #--hf-checkpoint /root/Qwen3-4B-FP8
   --ref-load "${MODEL_HOME}/Qwen3_4B_yenting_torch_dist"
   --load "${MEGATRON_CKPT_PATH}"
   --save "${MEGATRON_CKPT_PATH}"
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data "${DATA_HOME}/dapo-math-17k/dapo-math-17k.jsonl"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --custom-rm-path slime.rollout.rm_hub.judge_rm.get_llm_as_judge_reward
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 0.8

   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime "${DATA_HOME}/aime-2024/aime-2024.jsonl"
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3-4B-test
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.5
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

# MASTER_ADDR="147.185.41.214"
# MASTER_PORT="30265"
RAY_HEAD_ADDRESS="http://147.185.41.214:30265"
ROUTER_IP="147.185.41.214"
ROUTER_PORT="30000"
# '127.0.1.1', port=3330
# export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# export MASTER_ADDR=${RAY_HEAD_ADDRESS:-"127.0.0.1"}
# export MASTER_PORT=${MASTER_PORT:-"12345"}

export no_proxy="localhost,0.0.0.0,127.0.1.1,127.0.0.1,${ROUTER_IP}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"no_proxy\": \"${no_proxy}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"MASTER_PORT\": \"${MASTER_PORT}\",
    \"PYTHONPATH\": \"/data/post_train/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"OPENAI_API_KEY\": \"${OPENAI_API_KEY}\"
  }
}"

   # --sglang-router-ip ${MASTER_ADDR} \
   # --sglang-router-port ${MASTER_PORT} \
ray job submit --address="${RAY_HEAD_ADDRESS}" \
   --working-dir="." \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 4 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}