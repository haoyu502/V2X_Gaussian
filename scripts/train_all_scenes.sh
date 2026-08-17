#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_dir="${DATA_DIR:-${repo_dir}/data}"
gpu_list="${GPUS:-0,1}"
gpu_count="${NPROC_PER_NODE:-2}"
config="${CONFIG:-arguments/multi_agents/v2x_gaussian_4090.py}"
experiment_suffix="${EXP_SUFFIX:-2x4090}"
base_port="${BASE_PORT:-6010}"
final_iteration="${ITERATION:-20000}"

cd "${repo_dir}"

echo "Initializing missing LiDAR point clouds..."
"${repo_dir}/scripts/initialize_all_lidar.sh" "${data_dir}" "${VOXEL_SIZE:-0.05}"

mapfile -t scenes < <(
    find "${data_dir}" -mindepth 1 -maxdepth 1 -type d \
        -exec test -d '{}/lidar' ';' -printf '%f\n' | sort
)

if [[ ${#scenes[@]} -eq 0 ]]; then
    echo "No trainable scenes found under ${data_dir}" >&2
    exit 1
fi

echo "Found ${#scenes[@]} scenes; GPUs=${gpu_list}; processes=${gpu_count}"

for index in "${!scenes[@]}"; do
    scene="${scenes[$index]}"
    scene_dir="${data_dir}/${scene}"
    experiment="${scene}_${experiment_suffix}"
    output_dir="${repo_dir}/output/${experiment}"
    final_checkpoint="${output_dir}/chkpnt_fine_${final_iteration}.pth"
    final_point_cloud="${output_dir}/point_cloud/iteration_${final_iteration}/point_cloud.ply"
    port=$((base_port + index))

    if [[ -s "${final_checkpoint}" && -s "${final_point_cloud}" ]]; then
        echo "[skip] ${scene}: completed output exists"
        continue
    fi
    if [[ ! -s "${scene_dir}/lidar_downsampled.ply" ]]; then
        echo "[error] ${scene}: missing lidar_downsampled.ply" >&2
        exit 1
    fi

    echo "[train $((index + 1))/${#scenes[@]}] ${scene} -> output/${experiment}"
    CUDA_VISIBLE_DEVICES="${gpu_list}" torchrun \
        --standalone \
        --nproc_per_node="${gpu_count}" \
        train_v2x_gaussians.py \
        -s "${scene_dir}" \
        --expname "${experiment}" \
        --configs "${config}" \
        --port "${port}" \
        --checkpoint_iterations 2500 5000 10000 15000 "${final_iteration}"
done

echo "All scenes completed."
