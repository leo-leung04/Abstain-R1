#!/bin/bash
# vLLM 推理脚本 - Abstain 数据集推理

# 禁用 flashinfer（解决 CUDA 11.8 兼容性问题）
# flashinfer 需要 CUDA 12.0+，但系统使用 CUDA 11.8
export VLLM_DISABLE_FLASHINFER=1
export VLLM_USE_FLASHINFER_SAMPLER=0  # 禁用 flashinfer 采样器（关键！）
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Triton 编译相关设置
# 注意：vLLM 使用 Triton 进行 JIT 编译，需要 C 编译器（gcc）
# 如果遇到 "Failed to find C compiler" 错误，需要安装 gcc：
#   sudo apt-get update && sudo apt-get install -y build-essential
# 或者设置 CC 环境变量指向可用的 C 编译器：
#   export CC=/usr/bin/gcc

# 设置可见的 GPU（vLLM tensor parallel 需要）
# 使用 4 张 GPU（0, 1, 2, 3）
export CUDA_VISIBLE_DEVICES=0,1,2,3

# ===============================
# 配置区域 - 根据需要修改
# ===============================

# 模型路径或名称
# 可以是预定义的模型 key（如 qwen25_3b_instruct）或 HuggingFace 模型路径
MODEL_PATH="/scratch.global/haoti002/models/qwen25_7b_instruct"

# 输入输出文件
INPUT_FILE="/users/9/haoti002/nlpproject/data/Abstain/abstain-Test-withoutsum.jsonl"
OUTPUT_DIR="/users/9/haoti002/nlpproject/results/abstain-test"
OUTPUT_FILE="${OUTPUT_DIR}/qwen25_7b_instruct_result.jsonl"

# 推理脚本路径
SCRIPT_PATH="/users/9/haoti002/nlpproject/run/run_abstain_model_batch_vllm.py"

# ===============================
# 推理参数配置
# ===============================

# 批处理大小
BATCH_SIZE=128

# 最大生成 token 数
MAX_NEW_TOKENS=4096

# 采样温度（0 表示贪婪解码）
TEMPERATURE=0

# Top-p 采样
TOP_P=1.0

# GPU 内存利用率
GPU_MEM_UTIL=0.9

# 数据类型
DTYPE="float16"

# 张量并行数（使用的 GPU 数量）
TENSOR_PARALLEL=4

# Prompt 模板
# 可选: structured, structured_strict_answer
TEMPLATE="structured_strict_answer"

# 限制处理数量（留空表示处理全部）
LIMIT=""

# ===============================
# 执行推理
# ===============================

# 确保输出目录存在
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Abstain Dataset vLLM Inference"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Input: ${INPUT_FILE}"
echo "Output: ${OUTPUT_FILE}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Max New Tokens: ${MAX_NEW_TOKENS}"
echo "Temperature: ${TEMPERATURE}"
echo "Template: ${TEMPLATE}"
echo "Tensor Parallel Size: ${TENSOR_PARALLEL}"
echo "=========================================="

# 构建命令
CMD="python ${SCRIPT_PATH} \
    --model \"${MODEL_PATH}\" \
    --input \"${INPUT_FILE}\" \
    --output \"${OUTPUT_FILE}\" \
    --batch_size ${BATCH_SIZE} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --gpu_memory_utilization ${GPU_MEM_UTIL} \
    --dtype ${DTYPE} \
    --tensor_parallel_size ${TENSOR_PARALLEL} \
    --template ${TEMPLATE}"

# 如果设置了 LIMIT，添加 --limit 参数
if [ -n "${LIMIT}" ]; then
    CMD="${CMD} --limit ${LIMIT}"
fi

# 执行命令
echo "Running command:"
echo "${CMD}"
echo ""

eval ${CMD}

echo ""
echo "=========================================="
echo "Inference completed!"
echo "Results saved to: ${OUTPUT_FILE}"
echo "=========================================="

