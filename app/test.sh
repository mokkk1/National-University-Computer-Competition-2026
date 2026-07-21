#!/bin/bash
# test.sh — 启动推理，生成 result.csv
set -e
echo "=== 开始预测 ==="
cd /app
python code/src/test.py
echo "=== 预测完成 ==="
