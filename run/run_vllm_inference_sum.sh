#!/bin/bash
# vLLM inference script - 3B model on 2 GPUs

# Disable flashinfer (works around CUDA 11.8 compatibility issue)
# flashinfer requires CUDA 12.0+, but the system uses CUDA 11.8
export VLLM_DISABLE_FLASHINFER=1
export VLLM_USE_FLASHINFER_SAMPLER=0  # Disable flashinfer sampler (important!)
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Set visible GPUs (needed for vLLM tensor parallel)
# Use the first two GPUs (0 and 1)
export CUDA_VISIBLE_DEVICES=0,1

# Model path
MODEL_PATH="/scratch.global/haoti002/models/qwen25_3b_instruct"

# IO files
INPUT_FILE="/users/9/haoti002/nlpproject/data/sum/sum_test.jsonl"
OUTPUT_DIR="/users/9/haoti002/nlpproject/results"
OUTPUT_FILE="${OUTPUT_DIR}/qwen25-3b-instruct_vllm_result.jsonl"

# Ensure output directory exists
mkdir -p "${OUTPUT_DIR}"

# Run inference (using 2 GPUs)
python /users/9/haoti002/nlpproject/run/run_single_model_batch_vllm.py \
    --model "${MODEL_PATH}" \
    --input "${INPUT_FILE}" \
    --output "${OUTPUT_FILE}" \
    --batch_size 128 \
    --max_new_tokens 4096 \
    --temperature 0 \
    --top_p 1.0 \
    --gpu_memory_utilization 0.9 \
    --dtype float16 \
    --tensor_parallel_size 4 \
    --template structured_strict_answer
