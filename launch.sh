#!/bin/bash
# 激活 verl 环境（如果需要）
# source /opt/conda/etc/profile.d/conda.sh
# conda activate verl


# 禁用 flashinfer 以避免 nvcc 错误
export VLLM_DISABLE_FLASHINFER=1
export VLLM_USE_FLASHINFER_SAMPLER=0  # 禁用 flashinfer 采样器（关键！）
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# ============ 路径与参数配置 ============
# 尝试禁用 vLLM V1 引擎以避免 CUDA fork 问题，或强制使用 spawn
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# GPU 0 -> 端口 8001
CUDA_VISIBLE_DEVICES=0 uvicorn judge_server_vllm:app --host 0.0.0.0 --port 8001 &

# GPU 1 -> 端口 8002
CUDA_VISIBLE_DEVICES=1 uvicorn judge_server_vllm:app --host 0.0.0.0 --port 8002 &

# GPU 2 -> 端口 8003
CUDA_VISIBLE_DEVICES=2 uvicorn judge_server_vllm:app --host 0.0.0.0 --port 8003 &

# GPU 3 -> 端口 8004
CUDA_VISIBLE_DEVICES=3 uvicorn judge_server_vllm:app --host 0.0.0.0 --port 8004 &

# 等待所有后台进程
wait
