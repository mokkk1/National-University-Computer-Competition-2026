#!/bin/bash
# train.sh — 启动训练
set -e
echo "=== 开始训练 ==="
cd /app
python code/src/train.py
echo "=== 训练完成 ==="
