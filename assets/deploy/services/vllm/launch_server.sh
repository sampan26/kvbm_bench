#!/bin/bash

# Launch vLLM with NVIDIA Triton Inference Server optimized build
# This should have proper support for compute capability 12.1 (DGX Spark)

# Enable unified memory usage for DGX Spark
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Enable CUDA unified memory and oversubscription
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_NO_CUDA_MEMORY_CACHING=0

# Force vLLM to use CPU offloading for large models
export VLLM_CPU_OFFLOAD_GB=50
export VLLM_ALLOW_RUNTIME_LORA_UPDATES_WITH_SGD_LORA=1
export VLLM_SKIP_WARMUP=0

# Optimized environment for performance
export VLLM_LOGGING_LEVEL=INFO
export PYTHONUNBUFFERED=1

# Enable CUDA optimizations
export VLLM_USE_MODELSCOPE=false

# Enable unified memory in vLLM
export VLLM_USE_V1=0

# First, test basic CUDA functionality
echo "=== Testing CUDA functionality ==="
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name} (compute capability {props.major}.{props.minor})')
        # Try basic CUDA operation
        try:
            x = torch.randn(10, 10).cuda(i)
            y = torch.matmul(x, x.T)
            print(f'GPU {i}: Basic CUDA operations work')
        except Exception as e:
            print(f'GPU {i}: CUDA operation failed: {e}')
"

echo "=== Starting optimized vLLM server ==="

# Ensure HuggingFace token is set and clear any cached tokens
if [ -n "$HF_TOKEN" ]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    echo "HuggingFace token is set (length: ${#HF_TOKEN})"
    # Clear any cached tokens and login with the correct token
    rm -rf /root/.cache/huggingface/token
    rm -rf /root/.huggingface/token
    echo "$HF_TOKEN" > /root/.cache/huggingface/token
    # Login using huggingface-cli
    python3 -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"
    echo "Logged in to HuggingFace"
elif [ -n "$HUGGING_FACE_HUB_TOKEN" ]; then
    export HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"
    echo "HuggingFace token is set from HUGGING_FACE_HUB_TOKEN (length: ${#HUGGING_FACE_HUB_TOKEN})"
    # Clear any cached tokens and login with the correct token
    rm -rf /root/.cache/huggingface/token
    rm -rf /root/.huggingface/token
    echo "$HUGGING_FACE_HUB_TOKEN" > /root/.cache/huggingface/token
    # Login using huggingface-cli
    python3 -c "from huggingface_hub import login; login(token='${HUGGING_FACE_HUB_TOKEN}')"
    echo "Logged in to HuggingFace"
else
    echo "WARNING: No HuggingFace token found. Gated models will not work."
fi

# Use model from environment variable, or default to configured model
MODEL_TO_USE="${VLLM_MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
QUANTIZATION_FLAG=""
GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
MAX_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
CPU_OFFLOAD_GB="0"  # No CPU offload for smaller models

echo "Using model: $MODEL_TO_USE"
echo "Quantization: ${QUANTIZATION_FLAG:-'disabled'}"
echo "GPU memory utilization: $GPU_MEMORY_UTIL"
echo "Max model length: $MAX_MODEL_LEN"
echo "Max num seqs: $MAX_NUM_SEQS"
echo "Max batched tokens: $MAX_BATCHED_TOKENS"
echo "CPU Offload: ${CPU_OFFLOAD_GB}GB"

vllm serve "$MODEL_TO_USE" \
  --host 0.0.0.0 \
  --port 8001 \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
  --cpu-offload-gb "$CPU_OFFLOAD_GB" \
  --kv-cache-dtype auto \
  --trust-remote-code \
  --served-model-name "$MODEL_TO_USE" \
  --enable-chunked-prefill \
  --disable-custom-all-reduce \
  --disable-async-output-proc \
  $QUANTIZATION_FLAG