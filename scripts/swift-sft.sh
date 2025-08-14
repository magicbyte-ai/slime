#!/bin/bash
# Usage: bash swift-sft.sh 0

set -ex

# --- Getting pod status ---
# sudo kubectl get pods -o wide
# NAME                                      READY   STATUS             RESTARTS         AGE     IP             NODE   NOMINATED NODE   READINESS GATES
# kuberay-operator-6d74c6dc6c-7r428         1/1     Running            0                7d1h    10.42.24.94    g262   <none>           <none>
# raycluster-gpu-gpu-workers-worker-2kcj8   0/1     Running            31 (6m22s ago)   3d10h   10.42.28.93    g265   <none>           <none>
# raycluster-gpu-gpu-workers-worker-78pth   0/1     Init:1/2           0                4h7m    10.42.22.52    g258   <none>           <none>
# raycluster-gpu-gpu-workers-worker-lkz97   0/1     Running            30 (6m18s ago)   3d10h   10.42.24.108   g262   <none>           <none>
# raycluster-gpu-gpu-workers-worker-plwr2   0/1     Running            29 (6m47s ago)   3d10h   10.42.36.117   g337   <none>           <none>
# raycluster-gpu-head-ww88v                 0/1     CrashLoopBackOff   56 (4m39s ago)   4h33m   10.15.31.1     g255   <none>           <none>
RAY_HEAD_ADDRESS="http://147.185.41.214:30265"
NS=default
POD0=raycluster-gpu-gpu-workers-worker-2kcj8
POD1=raycluster-gpu-gpu-workers-worker-78pth
POD2=raycluster-gpu-gpu-workers-worker-lkz97
POD3=raycluster-gpu-gpu-workers-worker-plwr2

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


sudo kubectl exec -n $NS -it $POD -- bash -lc "
set -e

pip install "/data/post_train/yenting/ms-swift/[all]" -U
pip install "deepspeed" -U
pip install "transformers==4.55.0"

# huggingface-cli download openai/gpt-oss-20b --local-dir /data/post_train/yenting/models/gpt-oss-20b
hf download openai/gpt-oss-20b --local-dir /data/post_train/yenting/models/gpt-oss-20b

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export NPROC_PER_NODE=8

swift sft \
    --use_hf 1 \
    --model /data/post_train/yenting/models/gpt-oss-20b \
    --train_type full \
    --dataset /data/post_train/yenting/data/sample.jsonl \
    --torch_dtype bfloat16 \
    --max_steps 1000 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 2 \
    --packing true \
    --eval_steps 10 \
    --save_steps 10 \
    --logging_steps 1 \
    --max_length 8192 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --save_total_limit 2 \
    --save_only_model true \
    --output_dir /data/post_train/yenting/models/gpt-oss-20b-sft \
    --deepspeed zero3 \
    --attn_impl flash_attention_2 \
"