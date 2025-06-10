# Docker 部署指南

------

## ⚙️ 部署先决条件

在开始之前，请确保您的系统中已正确安装并配置了以下所有软件：

- **Git**: [官方网站](https://git-scm.com/downloads)
- **Docker Engine**: [官方安装指南](https://docs.docker.com/engine/install/)
- **Docker Compose**: [官方安装指南](https://docs.docker.com/compose/install/)
- **NVIDIA Container Toolkit**: [官方安装指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

------

## 📝 配置文件详解

部署的核心是以下四个配置文件。理解它们的功能是成功部署的关键。

### 1. `pyproject.toml` (位于项目根目录)

此文件是 Python 项目的现代化管理核心，定义了项目的元数据和所有依赖项。

- **用途**: 替代传统的 `requirements.txt`，提供更丰富的项目管理功能。

```
# 位于项目根目录下的 pyproject.toml
[project]
name = "face-rec"
version = "0.1.0"
description = "A high-performance face recognition system."
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "aiofiles>=24.1.0",
    "fastapi[standard]>=0.115.12",
    "insightface>=0.7.3",
    "loguru>=0.7.3",
    "numpy>=1.26.0",
    "onnx>=1.13.1",
    "onnxruntime-gpu>=1.17.0", # GPU版本的ONNX运行时
    "opencv-python>=4.8.0.76",
    "pandas>=2.0.3",
    "pydantic>=2.5.3",
    "pydantic-settings>=2.1.0",
    "python-dotenv>=1.0.0",
    "pyyaml>=6.0.1",
    "scipy>=1.12.0",
    "sqlalchemy>=2.0.25",
    "streamlit>=1.30.0",
    "typer>=0.9.0",
]

# 为 uv 工具配置全局镜像源
[tool.uv]
[[tool.uv.index]]
url = "https://pypi.tuna.tsinghua.edu.cn/simple/"
default = true
```

### 2. `docker/start.sh`

这是一个 Shell 脚本，作为 Docker 容器的入口点（Entrypoint），负责启动容器内的所有服务。

- **用途**: 解决单个容器需要运行多个进程（FastAPI 和 Streamlit）的问题。
- 关键部分
  - `set -e`: 确保脚本在任何命令出错时立即停止，防止出现意外行为。
  - `cd /app`: 切换到项目根目录，这是确保后续所有命令路径正确的关键一步。
  - `streamlit ... &`: `&` 符号将 Streamlit 进程放入后台运行。
  - `python ...`: FastAPI 应用在前台运行，作为容器的主进程。只有主进程结束，容器才会退出。

```
#!/bin/bash
# 位于 docker/ 目录下的 start.sh

# 当任何命令失败时，立即退出脚本
set -e

# 关键步骤：切换到容器内的项目根目录 /app
# 这能确保后续所有命令的相对路径都是正确的
cd /app || exit

echo "[INFO] Current working directory: $(pwd)"
echo "[INFO] Starting Streamlit UI in background..."

# 以后台模式启动 Streamlit UI，并允许从外部访问
streamlit run ui/ui.py --server.address=0.0.0.0 --server.port=8501 &

echo "[INFO] Starting FastAPI application in foreground..."

# 在前台启动 FastAPI 应用 (作为容器的主进程)
python run.py --env production start
```

### 3. `docker/Dockerfile`

这是构建 Docker 镜像的“设计蓝图”，它定义了从基础环境到最终应用的每一步构建指令。

- **用途**: 创建一个包含所有依赖和代码的、标准化的、可移植的应用镜像。
- 关键部分
  - `FROM nvidia/cuda...`: 选择一个仅包含 CUDA 运行时的官方镜像，以兼顾功能与体积。
  - `pip install -i ... uv`: 从清华源高速安装 `uv` 工具本身。
  - `COPY pyproject.toml .` 和 `RUN uv pip install ...`: 先复制依赖文件再安装，充分利用 Docker 的层缓存机制。
  - `COPY . .`: 将项目所有代码复制到镜像中。
  - `CMD ["./docker/start.sh"]`: 指定容器启动时默认执行的命令。

```
# 位于 docker/ 目录下的 Dockerfile

# --- Stage 1: 基础镜像 ---
# 使用 NVIDIA 官方的 CUDA 运行时镜像。它比 -devel 镜像更小，但包含运行 GPU 应用所需的所有库。
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# --- Stage 2: 环境配置 ---
# 设置环境变量，避免交互式提示并声明 NVIDIA GPU 可用
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# --- Stage 3: 系统及工具安装 ---
# 更新包列表并安装 Python 和 pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.10 python3-pip && \
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

# 赋予启动脚本执行权限
RUN chmod +x docker/start.sh

# --- Stage 5: 容器运行配置 ---
# 暴露 FastAPI 和 Streamlit 的端口
EXPOSE 8000
EXPOSE 8501

# 定义容器启动命令，执行我们的启动脚本
CMD ["./docker/start.sh"]
```

### 4. `docker/docker-compose.yml`

这是一个声明式的服务编排文件，让您能够用更简单、更清晰的方式来定义和管理多容器（或单容器）应用。

- **用途**: 简化 `docker run` 中冗长的参数，使应用的配置、启动、停止和销毁变得极其简单。
- 关键部分
  - `build.context: ..`: 由于此文件在 `docker/` 内，`..` 指示 Docker 将上一级目录（项目根目录）作为构建上下文。
  - `volumes`: 定义了数据卷挂载，将主机的 `data` 和 `logs` 目录映射到容器内，这是实现数据持久化的核心。
  - `deploy.resources`: 这是为容器预留 GPU 资源的现代化标准方式。

```
# 位于 docker/ 目录下的 docker-compose.yml

# 定义 Compose 文件的版本
version: '3.8'

# 定义服务
services:
  # 服务的名称
  face-rec-service:
    # 构建指令
    build:
      # 'context'：构建上下文的路径。'..' 表示项目根目录。
      context: ..
      # 'dockerfile'：Dockerfile 的路径，相对于 context。
      dockerfile: docker/Dockerfile
    
    # 容器名称
    container_name: face_recognition_service
    
    # 端口映射: "主机端口:容器端口"
    ports:
      - "12010:8000"  # FastAPI
      - "12011:8501"  # Streamlit
      
    # 数据卷挂载: "主机路径:容器路径"
    # 这是实现数据持久化的关键，防止容器删除后数据丢失
    volumes:
      - ../data:/app/data
      - ../logs:/app/logs
      
    # GPU 资源配置 (Docker Compose 推荐方式)
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
              
    # 重启策略: 除非手动停止，否则容器总是在退出后重启
    restart: unless-stopped
```

------

## 🚀 部署指令

> **重要提示**: 两种方案只需选择一种即可。**方案二 (Docker Compose)** 因为其便捷性而更被推荐。

### 方案一：使用 Dockerfile（手动挡）

此方法更底层，适用于需要构建独立镜像用于分发的场景。所有操作均在 **项目根目录** 下执行。

1. 构建镜像

   ```
   docker build -t face-rec-app -f docker/Dockerfile .
   ```

2. 运行容器

   ```
   docker run -it --rm \
     --gpus all \
     -p 12010:8000 \
     -p 12011:8501 \
     -v ./data:/app/data \
     -v ./logs:/app/logs \
     --name face-rec-container \
     face-rec-app
   ```

### 方案二：使用 Docker Compose（自动挡 - 推荐）

此方法通过 `docker-compose.yml` 文件管理所有配置，是开发和大多数生产环境的首选。所有操作均在 **`docker/` 目录** 下执行。

1. 导航至 `docker` 目录

   ```
   cd docker
   ```

2. 一键启动服务

   ```
   docker compose up --build -d
   ```

   - `--build`: 强制在启动前重新构建镜像。
   - `-d`: 以分离（后台）模式运行。

------

## ✅ 应用访问与管理

- **访问应用**:
  - API 文档 (Swagger UI): **`http://localhost:12010/docs`**
  - Web 界面 (Streamlit): **`http://localhost:12011`**
- **管理应用**:
  - Docker Compose
    - 查看日志: `docker compose logs -f`
    - 停止并移除: `docker compose down`
  - Dockerfile (手动)
    - 查看日志: `docker logs -f face-rec-container`
    - 停止并移除: `docker stop face-rec-container`

------

## 🤔 常见问题排查 (FAQ)

### Q: 在国内环境下，启动时下载模型文件非常缓慢或失败怎么办？

A: 这是由于国内网络访问 GitHub Releases 不稳定导致的。可以通过预先手动下载模型，再利用数据卷挂载到容器内部来解决。这是最推荐的方案。

**逻辑**：在您的主机（您自己的电脑）上手动下载好模型文件，并放到指定的 `data` 目录下。当容器启动时，由于文件已经存在于挂载的目录中，`insightface` 库会直接使用本地文件，从而跳过下载步骤。

**操作步骤：**

1. **在主机下载文件**

   - **下载地址**: https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
   - **提示**: 在您的电脑上，您可以使用下载工具（如 `wget`、`curl` 或浏览器）、科学上网工具或下载代理来加速下载过程。

2. 在主机项目根目录创建相应文件夹

   根据日志提示的路径，您需要在您的项目根目录（与 docker 文件夹同级）下，手动创建所需的文件夹结构。目录路径如下：

   ```
   ./data/.insightface/models/
   ```

3. 放置模型文件

   将第一步下载好的 buffalo_l.zip 文件，放入刚刚创建的 ./data/.insightface/models/ 目录中。

4. 重新启动服务

   回到 docker/ 目录下，重新启动服务即可。

   ```
   docker compose up -d --force-recreate
   ```

   现在，容器启动时会直接在 `/app/data/.insightface/models/buffalo_l.zip` 找到文件，下载过程将被跳过。