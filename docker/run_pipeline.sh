#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 DATA_DIR OUTPUT_DIR [GPU_IDS] [CONFIG] [SUFFIX]" >&2
  exit 2
fi

DATA_DIR="$(realpath "$1")"
OUTPUT_DIR="$(realpath "$2")"
GPU_IDS="${3:-0,1,2,3}"
CONFIG="${4:-arguments/multi_agents/v2x_gaussian_residual_enhancement.py}"
SUFFIX="${5:-e1_residual_025_4x4090}"
IMAGE_NAME="${IMAGE_NAME:-v2x-gaussian:v3}"

mkdir -p "$OUTPUT_DIR"

docker run --rm \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "$DATA_DIR:/workspace/V2X_Gaussian/data:ro" \
  -v "$OUTPUT_DIR:/workspace/V2X_Gaussian/output" \
  "$IMAGE_NAME" \
  python scripts/run_full_pipeline.py \
    --data-root data \
    --gpus "$GPU_IDS" \
    --config "$CONFIG" \
    --suffix "$SUFFIX"
