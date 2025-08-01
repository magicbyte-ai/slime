#!/bin/bash
set -ex

source scripts/models/qwen3-235B-A22B.sh

MODEL_HOME="/data/post_train/models/"
MEGATRON_CKPT_PATH="${MODEL_HOME}/Qwen3-235B-A22B_slime"

RAY_HEAD_ADDRESS="http://147.185.41.214:30265"

# RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING=1
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/data/post_train/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING\": \"1\"
  }
}"

ray job submit --address="${RAY_HEAD_ADDRESS}" \
   --working-dir="." \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint "${MODEL_HOME}/Qwen3-235B-A22B" \
    --save "${MEGATRON_CKPT_PATH}"