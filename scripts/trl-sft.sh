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

export HF_HOME=/data/post_train/yenting/hf_home/

cd /data/post_train/yenting/gpt-oss-recipes
pwd

# pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu128
pip install git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels
pip install -r requirements.txt

accelerate launch \
  --config_file configs/zero3.yaml \
  sft.py \
    --config configs/sft_full.yaml \
    --attn_implementation kernels-community/vllm-flash-attn3 \
    --output_dir /data/post_train/yenting/models/gpt-oss-20b-multilingual-reasoner-1
"