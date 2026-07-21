#!/bin/bash
# test.sh — 生成 Top-5 选股预测结果
# 用法: bash test.sh
# 等同于: python scripts/predict.py
# 输出: output/result.csv (stock_id, weight 格式)
set -e

# 自动定位项目根目录（脚本所在目录）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== 沪深300 Top-5 选股预测 ==="
echo "项目目录: $SCRIPT_DIR"

# 如果存在虚拟环境则激活
if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate  # Windows
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate      # Linux/Mac
fi

# 确保输出目录存在
mkdir -p output

# 运行预测（输出到 output/result.csv，竞赛标准格式）
python scripts/predict.py --output output/result.csv

# 检查输出
if [ -f "output/result.csv" ]; then
    echo ""
    echo "=== 预测结果 ==="
    cat output/result.csv
    echo ""
    echo "=== 预测完成，结果已保存至 output/result.csv ==="
else
    echo "错误: output/result.csv 未生成！"
    exit 1
fi
