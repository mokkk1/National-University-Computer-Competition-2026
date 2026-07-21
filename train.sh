#!/bin/bash
# train.sh — 训练沪深300排序模型
# 用法: bash train.sh
# 等同于: python code/src/train.py
set -e

# 自动定位项目根目录（脚本所在目录）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== 沪深300 排序模型训练 ==="
echo "项目目录: $SCRIPT_DIR"

# 如果存在虚拟环境则激活
if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate  # Windows
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate      # Linux/Mac
fi

# 确保输出目录存在
mkdir -p model output logs

# 开始训练
python code/src/train.py

echo "=== 训练完成 ==="
