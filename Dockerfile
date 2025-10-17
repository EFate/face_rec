# Dockerfile
FROM crpi-3syvxpuaq1m8bch1.cn-shanghai.personal.cr.aliyuncs.com/efatenex/cuda:12.4.1-runtime-ubuntu22.04

# 环境变量配置
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        build-essential \
        libcudnn9-cuda-12 \
        python3-dev \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 配置pip镜像源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn

# 设置工作目录
WORKDIR /app

# 复制并安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 设置启动脚本权限
RUN chmod +x start.sh

# 暴露端口
EXPOSE 12010 12011

# 启动命令
CMD ["./start.sh"]