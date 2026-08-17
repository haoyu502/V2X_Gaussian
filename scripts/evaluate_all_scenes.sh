#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"
CUDA_VISIBLE_DEVICES="${GPUS:-0}" python scripts/evaluate_all_scenes.py "$@"
