#!/usr/bin/env bash
set -euo pipefail

# Run the remaining V3 ablations sequentially because each experiment uses all
# four GPUs. Completed scenes/renders/metrics are skipped by the pipeline, so
# this launcher is safe to restart after an interruption.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_IDS="${1:-0,1,2,3}"

cd "$REPO_DIR"

echo "[E2] residual enhancement lambda=0.50 on GPUs ${GPU_IDS}"
python scripts/run_full_pipeline.py \
  --gpus "$GPU_IDS" \
  --config arguments/multi_agents/v2x_gaussian_residual_enhancement_050.py \
  --suffix e2_residual_050_4x4090

echo "[E3] original collaboration baseline on GPUs ${GPU_IDS}"
python scripts/run_full_pipeline.py \
  --gpus "$GPU_IDS" \
  --config arguments/multi_agents/v2x_gaussian_4090.py \
  --suffix e3_baseline_4x4090

echo "E2 and E3 completed."
echo "  output/evaluation_summary_e2_residual_050_4x4090.csv"
echo "  output/evaluation_summary_e3_baseline_4x4090.csv"
