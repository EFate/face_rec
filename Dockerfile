# Dockerfile

# --- Stage 1: 基础镜像 ---
# 使用 NVIDIA 官方的 CUDA 运行时镜像。它比 -devel 镜像更小，但包含运行 GPU 应用所需的所有库。
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# --- Stage 2: 环境配置 ---
# 设置环境变量，避免交互式提示并声明 NVIDIA GPU 可用
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# --- Stage 3: 系统及工具安装 ---
# 更新包列表并一次性安装所有构建依赖
# build-essential (提供 g++ 编译器)
# python3-dev (提供 Python.h 头文件)
# 安装 cuDNN 库
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    build-essential \
    libcudnn9-cuda-12 \
    python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 从清华镜像源安装 uv，以加速 uv 工具本身的下载
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple uv

# --- Stage 4: 项目配置与依赖安装 ---
# 创建并设置工作目录
WORKDIR /app

# 仅复制依赖定义文件，以便利用 Docker 的层缓存机制。如果此文件未更改，则不会重新运行下一步的安装
COPY pyproject.toml .

# 使用 uv 安装 Python 依赖。uv 会自动读取 pyproject.toml 中配置的清华镜像源
RUN uv pip install --system --no-cache -r pyproject.toml

# 将构建上下文（项目根目录）中的所有文件完整地复制到容器的工作目录（/app）中
COPY . .

# 赋予启动脚本执行权限（路径已更新）
RUN chmod +x start.sh

# --- Stage 5: 容器运行配置 ---
# 暴露 FastAPI 和 Streamlit 的端口
EXPOSE 12010
EXPOSE 12011

# 定义容器启动命令，执行我们的启动脚本（路径已更新）
CMD ["./start.sh"]