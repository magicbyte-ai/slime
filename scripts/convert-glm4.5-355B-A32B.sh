#!/bin/bash
# Usage: bash scripts/convert-qwen3-235B-A22B.sh 0
# Usage: bash scripts/convert-qwen3-235B-A22B.sh 1

set -ex

source scripts/models/qwen3-235B-A22B.sh

MODEL_HOME="/data/post_train/models/"
MEGATRON_CKPT_PATH="${MODEL_HOME}/Qwen3-235B-A22B_torch_dist"

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

NS=default
POD0=raycluster-gpu-gpu-workers-worker-g2dp4
POD1=raycluster-gpu-gpu-workers-worker-jnrth
POD2=raycluster-gpu-gpu-workers-worker-554zb
POD3=raycluster-gpu-gpu-workers-worker-2kcj8
MASTER_ADDR=$(sudo kubectl get pod -n $NS $POD0 -o jsonpath='{.status.podIP}')
MASTER_PORT=23456

REPO=/data/post_train/yenting/slime
BASE_DIR=/data/post_train/models/
HF="$BASE_DIR/GLM-4.5-355B-A32B"
SAVE="$BASE_DIR/GLM-4.5-355B-A32B_torch_dist/"

# 0, 1 is determined by $1. User will pass 0 or 1 or 2 or 3.
POD_NUM=$1

if [ $POD_NUM -eq 0 ]; then
  POD=$POD0
elif [ $POD_NUM -eq 1 ]; then
  POD=$POD1
elif [ $POD_NUM -eq 2 ]; then
  POD=$POD2
elif [ $POD_NUM -eq 3 ]; then
  POD=$POD3
else
  echo "Invalid pod number: $POD_NUM"
  exit 1
fi


# sudo kubectl exec -n $NS -it $POD -- bash -lc 'ls -al /data/post_train/'
# sudo kubectl exec -n $NS -it $POD -- bash -lc 'ls -al /data/post_train/yenting/'
# sudo kubectl exec -n $NS -it $POD -- bash -lc 'echo hello-from-pod0-$(date +%s) > /data/post_train/yenting/tmp/.touch_from_pod0'
# sudo kubectl exec -n $NS -it $POD -- bash -lc 'cat /data/post_train/yenting/tmp/.touch_from_pod0 || echo "NOT_SHARED"'
# # see torch is installed
# sudo kubectl exec -n $NS -it $POD -- bash -lc 'python3 -c "import torch; print(f\"PyTorch version: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"CUDA devices: {torch.cuda.device_count()}\")"'
# # see torchrun is working
# sudo kubectl exec -n $NS -it $POD -- bash -lc 'torchrun --help | head -5'

sudo kubectl exec -n $NS -it $POD -- bash -lc "
set -e
cd $REPO
ls tools/convert_hf_to_torch_dist.py

source scripts/models/glm4.5-355B-A32B.sh

# 建議：緩解記憶體碎片、設定網卡
export PYTHONPATH=/data/post_train/Megatron-LM/
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME=\$(ip -o -4 route show to default | awk '{print \$5}')

pip install -e .

huggingface-cli download zai-org/GLM-4.5 --local-dir $BASE_DIR/GLM-4.5-355B-A32B

torchrun \
  --nproc-per-node 8 \
  --nnodes 2 \
   --node-rank $POD_NUM \
  --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
  tools/convert_hf_to_torch_dist.py \
  \${MODEL_ARGS[@]} \
  --hf-checkpoint $HF \
  --save $SAVE
"