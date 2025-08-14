#!/bin/bash
# Usage: bash trl-sft-multinode.sh
set -ex

# --- Configuration ---
RAY_HEAD_ADDRESS="http://147.185.41.214:30265"
NS=default
POD0=raycluster-gpu-gpu-workers-worker-2kcj8
POD1=raycluster-gpu-gpu-workers-worker-78pth
POD2=raycluster-gpu-gpu-workers-worker-lkz97
POD3=raycluster-gpu-gpu-workers-worker-plwr2

# Array of all pods
PODS=($POD0 $POD1 $POD2 $POD3)
NUM_NODES=${#PODS[@]}

# Master node configuration (using POD0 as master)
MASTER_POD=$POD0
MASTER_PORT=29500

# Get the master node's IP address
MASTER_ADDR=$(sudo kubectl get pod -n $NS $MASTER_POD -o jsonpath='{.status.podIP}')
echo "Master node address: $MASTER_ADDR:$MASTER_PORT"

# Function to launch training on a single pod
launch_on_pod() {
    local POD=$1
    local NODE_RANK=$2
    
    echo "Launching on pod $POD with rank $NODE_RANK"
    
    sudo kubectl exec -n $NS -it $POD -- bash -lc "
    set -e
    export HF_HOME=/data/post_train/yenting/hf_home/
    cd /data/post_train/yenting/gpt-oss-recipes
    pwd
    
    # Install dependencies (consider doing this once in the image instead)
    pip install git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels
    pip install -r requirements.txt
    
    # Set distributed training environment variables
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT
    export WORLD_SIZE=$NUM_NODES
    export NODE_RANK=$NODE_RANK
    export LOCAL_RANK=0
    
    # For multi-GPU per node, you might need to adjust these
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Adjust based on your GPU configuration
    
    echo 'Starting training on node rank: $NODE_RANK'
    echo 'Master address: $MASTER_ADDR:$MASTER_PORT'
    echo 'World size: $WORLD_SIZE'
    
    # Launch accelerate with distributed configuration
    accelerate launch \
      --config_file configs/zero3.yaml \
      --num_processes \$(nvidia-smi -L | wc -l) \
      --num_machines $NUM_NODES \
      --machine_rank $NODE_RANK \
      --main_process_ip $MASTER_ADDR \
      --main_process_port $MASTER_PORT \
      --mixed_precision bf16 \
      --dynamo_backend no \
      sft.py \
        --config configs/sft_full.yaml \
        --attn_implementation kernels-community/vllm-flash-attn3 \
        --output_dir /data/post_train/yenting/models/gpt-oss-20b-multilingual-reasoner-1
    " &
}

# Launch training on all nodes
for i in "${!PODS[@]}"; do
    launch_on_pod "${PODS[$i]}" "$i"
done

echo "Launched training on all $NUM_NODES nodes"
echo "Waiting for all processes to complete..."
wait

echo "Training completed on all nodes"