#!/bin/bash
# source /opt/conda/etc/profile.d/conda.sh
# conda activate verl


# Disable flashinfer to avoid nvcc errors.
export VLLM_DISABLE_FLASHINFER=1
export VLLM_USE_FLASHINFER_SAMPLER=0  # 禁用 flashinfer 采样器（关键！）
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Try disabling the vLLM V1 engine to avoid CUDA fork issues, or force the use of `spawn`.
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# GPU 0 -> Port 8001
CUDA_VISIBLE_DEVICES=0 uvicorn judge_server_vllm:app --host 0.0.0.0 --port 8001 &

# GPU 1 -> Port 8002
CUDA_VISIBLE_DEVICES=1 uvicorn judge_server_vllm:app --host 0.0.0.0 --port 8002 &

# GPU 2 -> Port 8003
CUDA_VISIBLE_DEVICES=2 uvicorn judge_server_vllm:app --host 0.0.0.0 --port 8003 &

# GPU 3 -> Port 8004
CUDA_VISIBLE_DEVICES=3 uvicorn judge_server_vllm:app --host 0.0.0.0 --port 8004 &

# Wait
wait
