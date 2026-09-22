#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-v2x-gaussian:v3}"
ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9}"
OUTPUT_FILE="${1:-$REPO_DIR/v2x-gaussian-v3-docker.tar.gz}"

command -v docker >/dev/null 2>&1 || {
  echo "ERROR: Docker is not installed on this machine." >&2
  exit 1
}

echo "Building ${IMAGE_NAME} for CUDA architectures ${ARCH_LIST}"
docker build \
  --build-arg "TORCH_CUDA_ARCH_LIST=${ARCH_LIST}" \
  -t "$IMAGE_NAME" \
  "$REPO_DIR"

echo "Exporting image to ${OUTPUT_FILE}"
docker save "$IMAGE_NAME" | gzip -1 > "$OUTPUT_FILE"

echo "Done: ${OUTPUT_FILE}"
du -h "$OUTPUT_FILE"
