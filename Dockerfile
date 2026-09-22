# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"

ENV CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=4 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/conda/envs/v2x_gaussian/bin:/opt/conda/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /tmp/miniconda.sh \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && conda create -y -n v2x_gaussian python=3.9 pip \
    && conda clean -afy

WORKDIR /workspace/V2X_Gaussian

# Install large, stable dependencies before copying source so normal code edits
# do not invalidate the dependency layers.
COPY requirements.txt ./requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    && pip install -r requirements.txt

COPY . .

# Build the project CUDA extensions into the image. The default architecture
# list supports A100 (8.0), RTX 30/A-series variants (8.6), and RTX 4090 (8.9).
RUN pip install --no-build-isolation ./submodules/depth-diff-gaussian-rasterization \
    && pip install --no-build-isolation ./submodules/simple-knn \
    && python -c "import torch, mmcv, diff_gaussian_rasterization, simple_knn; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

RUN mkdir -p /workspace/V2X_Gaussian/data /workspace/V2X_Gaussian/output

CMD ["bash"]
