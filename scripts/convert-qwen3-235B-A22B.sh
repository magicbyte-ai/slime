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
MASTER_ADDR=$(sudo kubectl get pod -n $NS $POD0 -o jsonpath='{.status.podIP}')
MASTER_PORT=23456

REPO=/data/post_train/yenting/slime
HF=/data/post_train/models/Qwen3-235B-A22B-Thinking-2507
SAVE=/data/post_train/models/Qwen3-235B-A22B-Thinking-2507_torch_dist

# 0, 1 is determined by $1. User will pass 0 or 1.
POD_NUM=$1

if [ $POD_NUM -eq 0 ]; then
  POD=$POD0
elif [ $POD_NUM -eq 1 ]; then
  POD=$POD1
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
# 確認轉檔腳本存在；若不存在，請把 repo 掛進來或用 kubectl cp 進來
ls tools/convert_hf_to_torch_dist.py

# 載入 MODEL_ARGS（來自你的 qwen 設定檔）
source scripts/models/qwen3-235B-A22B.sh

# 建議：緩解記憶體碎片、設定網卡
export PYTHONPATH=/data/post_train/Megatron-LM/
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME=\$(ip -o -4 route show to default | awk '{print \$5}')

pip install -e .

torchrun \
  --nproc-per-node 8 \
  --nnodes 2 --node-rank $POD_NUM \
  --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
  tools/convert_hf_to_torch_dist.py \
  \${MODEL_ARGS[@]} \
  --hf-checkpoint $HF \
  --save $SAVE
"