#!/bin/bash

# Modern vLLM launch script that respects environment variables
# Supports all standard vLLM configuration options

set -e

# Enable Python unbuffered output for better logging
export PYTHONUNBUFFERED=1

# Set default values from environment variables or use sensible defaults
MODEL="${VLLM_MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8001}"
TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
QUANTIZATION="${VLLM_QUANTIZATION:-}"
KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-auto}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"

# Log configuration
echo "=== vLLM Configuration ==="
echo "Model: $MODEL"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "Quantization: ${QUANTIZATION:-none}"
echo "KV Cache Dtype: $KV_CACHE_DTYPE"
echo "Max Num Seqs: $MAX_NUM_SEQS"
echo "Max Batched Tokens: $MAX_NUM_BATCHED_TOKENS"
echo "========================="

# Test CUDA availability
echo ""
echo "=== Testing CUDA ==="
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name} (compute capability {props.major}.{props.minor}, {props.total_memory / 1024**3:.1f} GB)')
else:
    print('WARNING: CUDA not available!')
    exit(1)
"
echo ""

# Build vLLM command
VLLM_CMD="python -m vllm.entrypoints.openai.api_server"

# Add model
VLLM_CMD="$VLLM_CMD $MODEL"

# Add server configuration
VLLM_CMD="$VLLM_CMD --host $HOST --port $PORT"

# Add model configuration
VLLM_CMD="$VLLM_CMD --tensor-parallel-size $TENSOR_PARALLEL_SIZE"
VLLM_CMD="$VLLM_CMD --max-model-len $MAX_MODEL_LEN"
VLLM_CMD="$VLLM_CMD --gpu-memory-utilization $GPU_MEMORY_UTILIZATION"
VLLM_CMD="$VLLM_CMD --max-num-seqs $MAX_NUM_SEQS"
VLLM_CMD="$VLLM_CMD --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS"

# Add quantization if specified
if [ -n "$QUANTIZATION" ]; then
    VLLM_CMD="$VLLM_CMD --quantization $QUANTIZATION"
fi

# Add KV cache dtype
VLLM_CMD="$VLLM_CMD --kv-cache-dtype $KV_CACHE_DTYPE"

# Add trust remote code for custom models
VLLM_CMD="$VLLM_CMD --trust-remote-code"

# Enable chunked prefill for better performance
VLLM_CMD="$VLLM_CMD --enable-chunked-prefill"

# Add served model name (use model name from path)
SERVED_MODEL_NAME=$(basename "$MODEL")
VLLM_CMD="$VLLM_CMD --served-model-name $SERVED_MODEL_NAME"

# Log the command
echo "=== Starting vLLM Server ==="
echo "Command: $VLLM_CMD"
echo ""

# Execute vLLM
exec $VLLM_CMD
