#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${CONDA_ENV_NAME:-v2x_gaussian}"
ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9}"

command -v conda >/dev/null 2>&1 || {
  echo "ERROR: conda is not available." >&2
  exit 1
}

command -v nvcc >/dev/null 2>&1 || {
  echo "ERROR: nvcc is required to build the Gaussian CUDA extensions." >&2
  echo "Install a CUDA 12.x toolkit and ensure nvcc is on PATH." >&2
  exit 1
}

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Updating existing environment: ${ENV_NAME}"
  conda env update -n "$ENV_NAME" -f "$REPO_DIR/environment.yml" --prune
else
  echo "Creating environment: ${ENV_NAME}"
  conda env create -n "$ENV_NAME" -f "$REPO_DIR/environment.yml"
fi

CUDA_ROOT="${CUDA_HOME:-$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)}"

echo "Building CUDA extensions for architectures: ${ARCH_LIST}"
CUDA_HOME="$CUDA_ROOT" TORCH_CUDA_ARCH_LIST="$ARCH_LIST" MAX_JOBS="${MAX_JOBS:-4}" \
  conda run -n "$ENV_NAME" pip install --no-build-isolation --force-reinstall \
  "$REPO_DIR/submodules/depth-diff-gaussian-rasterization"

CUDA_HOME="$CUDA_ROOT" TORCH_CUDA_ARCH_LIST="$ARCH_LIST" MAX_JOBS="${MAX_JOBS:-4}" \
  conda run -n "$ENV_NAME" pip install --no-build-isolation --force-reinstall \
  "$REPO_DIR/submodules/simple-knn"

conda run -n "$ENV_NAME" python -c \
  "import torch, mmcv, diff_gaussian_rasterization, simple_knn; print('torch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"

echo "Environment is ready. Activate it with: conda activate ${ENV_NAME}"
