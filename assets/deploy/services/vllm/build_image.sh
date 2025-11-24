#!/bin/bash

# Use latest stable vLLM release for better compute capability 12.1 support
# Clone the vLLM GitHub repo and use latest stable release.
git clone https://github.com/vllm-project/vllm.git /tmp/vllm-tutorial
cd /tmp/vllm-tutorial
git checkout $(git describe --tags --abbrev=0)

# Build the docker image using official vLLM Dockerfile.
DOCKER_BUILDKIT=1 docker build . \
        --file docker/Dockerfile \
        --target vllm-openai \
        --build-arg CUDA_VERSION=12.8.1 \
        --build-arg max_jobs=8 \
        --build-arg nvcc_threads=2 \
        --build-arg RUN_WHEEL_CHECK=false \
        --build-arg torch_cuda_arch_list="10.0+PTX;12.1" \
        --build-arg vllm_fa_cmake_gpu_arches="100-real;121-real" \
        -t vllm/vllm-openai:deploy

# Clean up
cd /
rm -rf /tmp/vllm-tutorial
