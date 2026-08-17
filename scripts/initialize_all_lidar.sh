#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_dir="${1:-${repo_dir}/data}"
voxel_size="${2:-0.05}"

for scene_dir in "${data_dir}"/*; do
    [[ -d "${scene_dir}/lidar" ]] || continue
    scene_name="$(basename "${scene_dir}")"
    output_path="${scene_dir}/lidar_downsampled.ply"
    if [[ -s "${output_path}" ]]; then
        echo "[skip] ${scene_name}: lidar_downsampled.ply already exists"
        continue
    fi
    echo "[init] ${scene_name}"
    python "${repo_dir}/V2X_GOTR_Dataset/lidar_initialization_per_scene.py" \
        --lidar_input_path "${scene_dir}/lidar" \
        --combined_output_path "${scene_dir}/lidar_combined.pcd" \
        --voxel_size "${voxel_size}" \
        --downsampled_output_path "${output_path}"
done
