#!/bin/bash
# 训练脚本 - 自动设置 PYTHONPATH

set -e

# 设置 verl 路径
export PYTHONPATH="/users/9/haoti002:$PYTHONPATH"

# 项目根目录
PROJECT_ROOT="/users/9/haoti002/nlpproject"
CONFIG_DIR="$PROJECT_ROOT/configs"

# 切换到项目目录
cd "$PROJECT_ROOT"

# 检查 verl 是否可以导入
python3 -c "import verl; print(f'verl 模块路径: {verl.__file__}')" || {
    echo "错误: 无法导入 verl 模块"
    echo "请确保 verl 已安装:"
    echo "  cd /users/9/haoti002/verl"
    echo "  pip install -e ."
    exit 1
}

# 检查配置文件是否存在
if [ ! -f "$CONFIG_DIR/verl_sft_abstain.yaml" ]; then
    echo "错误: 配置文件不存在: $CONFIG_DIR/verl_sft_abstain.yaml"
    exit 1
fi

# 检测 GPU 数量
N_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")

echo "=========================================="
echo "训练配置"
echo "=========================================="
echo "项目目录: $PROJECT_ROOT"
echo "配置目录: $CONFIG_DIR"
echo "检测到 $N_GPUS 个 GPU"
echo "PYTHONPATH: $PYTHONPATH"
echo "=========================================="
echo ""

# 根据 GPU 数量选择配置
if [ "$N_GPUS" -ge 4 ]; then
    CONFIG_NAME="verl_sft_abstain"
    echo "使用4卡配置: $CONFIG_NAME"
    
    # 确保在项目根目录运行
    # 使用绝对路径，因为 --config-path 是相对于 Python 脚本位置的
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$N_GPUS \
        -m verl.trainer.fsdp_sft_trainer \
        --config-path="${CONFIG_DIR}" \
        --config-name="${CONFIG_NAME}"
else
    CONFIG_NAME="verl_sft_abstain_local"
    echo "使用单卡配置: $CONFIG_NAME"
    
    python3 -m verl.trainer.fsdp_sft_trainer \
        --config-path="${CONFIG_DIR}" \
        --config-name="${CONFIG_NAME}"
fi

