#!/usr/bin/env python3
"""One-command preprocessing, multi-GPU training, rendering, and evaluation."""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def stream(command, cwd, env, log_file):
    printable = " ".join(str(value) for value in command)
    print(f"\n+ {printable}", flush=True)
    log_file.write(f"\n+ {printable}\n")
    log_file.flush()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in process.stdout:
        sys.stdout.write(line)
        log_file.write(line)
    return_code = process.wait()
    log_file.flush()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def main():
    parser = argparse.ArgumentParser(
        description="Run LiDAR initialization, train every scene, render videos, and summarize metrics")
    parser.add_argument("--data-root", default="data", help="Directory containing scene folders")
    parser.add_argument("--gpus", default="0,1", help="Comma-separated CUDA device IDs")
    parser.add_argument("--nproc-per-node", type=int, default=None,
                        help="Training processes; defaults to the number of GPU IDs")
    parser.add_argument("--config", default="arguments/multi_agents/v2x_gaussian_residual_enhancement.py")
    parser.add_argument("--suffix", default="residual_enhancement_2x4090",
                        help="Output name suffix and evaluation summary suffix")
    parser.add_argument("--iteration", type=int, default=None,
                        help="Expected final iteration; defaults to OptimizationParams.iterations in the config")
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--base-port", type=int, default=6010)
    parser.add_argument("--evaluation-gpu", default=None,
                        help="GPU used for rendering/metrics; defaults to the first training GPU")
    parser.add_argument("--force-render", action="store_true")
    parser.add_argument("--force-metrics", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = (repo / data_root).resolve()
    config = Path(args.config)
    if not config.is_absolute():
        config = (repo / config).resolve()
    if not data_root.is_dir():
        parser.error(f"data root does not exist: {data_root}")
    if not config.is_file():
        parser.error(f"config does not exist: {config}")
    try:
        import mmcv
        config_iterations = int(mmcv.Config.fromfile(str(config)).OptimizationParams.iterations)
    except (ImportError, AttributeError, KeyError, TypeError, ValueError) as error:
        parser.error(f"cannot read OptimizationParams.iterations from {config}: {error}")
    iteration = args.iteration or config_iterations
    if iteration != config_iterations:
        parser.error(f"--iteration={iteration} does not match config iterations={config_iterations}")

    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpu_ids:
        parser.error("--gpus must contain at least one device ID")
    process_count = args.nproc_per_node or len(gpu_ids)
    if process_count > len(gpu_ids):
        parser.error("--nproc-per-node cannot exceed the number of --gpus")
    evaluation_gpu = args.evaluation_gpu or gpu_ids[0]

    log_dir = repo / "output" / "pipeline_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"pipeline_{args.suffix}_{datetime.now():%Y%m%d_%H%M%S}.log"
    env = os.environ.copy()
    env.update({
        "DATA_DIR": str(data_root),
        "GPUS": ",".join(gpu_ids),
        "NPROC_PER_NODE": str(process_count),
        "CONFIG": str(config),
        "EXP_SUFFIX": args.suffix,
        "VOXEL_SIZE": str(args.voxel_size),
        "BASE_PORT": str(args.base_port),
        "ITERATION": str(iteration),
    })

    print(f"Pipeline log: {log_path}")
    try:
        with log_path.open("a", buffering=1) as log_file:
            stream(["bash", "scripts/train_all_scenes.sh"], repo, env, log_file)
            evaluation = [
                sys.executable,
                "scripts/evaluate_all_scenes.py",
                "--suffix", args.suffix,
                "--iteration", str(iteration),
                "--config", str(config),
            ]
            if args.force_render:
                evaluation.append("--force-render")
            if args.force_metrics:
                evaluation.append("--force-metrics")
            eval_env = env.copy()
            eval_env["CUDA_VISIBLE_DEVICES"] = evaluation_gpu
            stream(evaluation, repo, eval_env, log_file)
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"Pipeline stopped with exit code {error.returncode}. See {log_path}")

    summary_csv = repo / "output" / f"evaluation_summary_{args.suffix}.csv"
    summary_json = repo / "output" / f"evaluation_summary_{args.suffix}.json"
    print("\nPipeline completed successfully.")
    print(f"Metrics CSV:  {summary_csv}")
    print(f"Metrics JSON: {summary_json}")
    print(f"Full log:     {log_path}")


if __name__ == "__main__":
    main()
