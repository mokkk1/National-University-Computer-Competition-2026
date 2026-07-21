# Dockerfile — 比赛提交镜像
# 构建: docker buildx build --platform linux/amd64 --build-arg IMAGE_NAME=nvidia/cuda -t bdc2026 .

ARG IMAGE_NAME=nvidia/cuda
FROM ${IMAGE_NAME}:12.6.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    python3.11 \
    python3.11-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 创建 python3 软链接
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# 安装 TA-Lib 系统库
RUN wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz -O /tmp/ta-lib.tar.gz && \
    tar -xzf /tmp/ta-lib.tar.gz -C /tmp && \
    cd /tmp/ta-lib && \
    ./configure --prefix=/usr && \
    make -j1 && \
    make install && \
    cd / && rm -rf /tmp/ta-lib /tmp/ta-lib.tar.gz

# 安装 PyTorch (CUDA 版本，匹配基础镜像 CUDA 12.6)
# 使用 cu128 索引与 pyproject.toml 保持一致
RUN pip3 install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu128 \
    "torch>=2.6.0"

# 安装其余 Python 依赖 (从 PyPI)
RUN pip3 install --no-cache-dir \
    "akshare>=1.18.28" \
    "baostock>=0.8.9" \
    "joblib>=1.5.2" \
    "lightgbm>=4.0" \
    "numpy>=2.0.2" \
    "pandas>=2.3.2" \
    "scikit-learn>=1.7.2" \
    "seaborn>=0.13.2" \
    "ta-lib>=0.6.8" \
    "tensorboard>=2.20.0" \
    "tensorboardx>=2.6.4" \
    "tqdm>=4.67.1"

# 复制应用代码
COPY app/ /app/

# 设置工作目录
WORKDIR /app

# 使 shell 脚本可执行
RUN chmod +x /app/init.sh /app/train.sh /app/test.sh

CMD ["/bin/bash"]
