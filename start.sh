#!/bin/bash
# start.sh - 容器启动脚本

set -e

# 切换到项目根目录
cd /app || exit

echo "[INFO] 当前工作目录: $(pwd)"
echo "[INFO] 启动 Streamlit UI (后台运行)..."

# 启动 Streamlit UI
streamlit run ui/ui.py \
    --server.address=${WEBUI__HOST:-0.0.0.0} \
    --server.port=${WEBUI__PORT:-12011} &

echo "[INFO] 启动 FastAPI 应用 (前台运行)..."

# 启动 FastAPI 应用
python3 run.py start \
    --host=${SERVER__HOST:-0.0.0.0} \
    --port=${SERVER__PORT:-12010}