#!/bin/bash
# init.sh — 环境初始化（非联网操作）
# 依赖已在 Dockerfile 中安装，此处仅创建必要目录

set -e
echo "=== 初始化环境 ==="
cd /app

# 创建必要的目录
mkdir -p /app/model /app/output /app/temp /app/data

echo "环境初始化完成"
