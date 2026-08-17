#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def run(command, log_path):
    print("+", " ".join(str(part) for part in command), flush=True)
    with log_path.open("a", buffering=1) as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def main():
    parser = argparse.ArgumentParser(description="Render and evaluate all completed V2X-Gaussians scenes")
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--suffix", default="2x4090")
    parser.add_argument("--iteration", type=int, default=20000)
    parser.add_argument("--config", default="arguments/multi_agents/v2x_gaussian_4090.py")
    parser.add_argument("--force-render", action="store_true")
    parser.add_argument("--force-metrics", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    output_root = (repo / args.output_root).resolve()
    pattern = f"*_{args.suffix}" if args.suffix else "*"
    models = sorted(path for path in output_root.glob(pattern)
                    if (path / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply").is_file())
    if not models:
        raise SystemExit(f"No completed models matching {output_root / pattern}")

    print(f"Found {len(models)} completed scenes", flush=True)
    failures = []
    for index, model in enumerate(models, 1):
        method_dir = model / "test" / f"ours_{args.iteration}"
        test_video = method_dir / "video_rgb.mp4"
        novel_video = model / "video" / f"ours_{args.iteration}" / "video_rgb.mp4"
        results = model / "results.json"
        eval_log = model / "evaluation.log"
        print(f"\n[{index}/{len(models)}] {model.name}", flush=True)
        try:
            if args.force_render or not test_video.is_file() or not novel_video.is_file() \
                    or not (method_dir / "renders").is_dir():
                run([sys.executable, "render.py", "--model_path", str(model),
                     "--iteration", str(args.iteration), "--configs", args.config,
                     "--skip_train"], eval_log)
            else:
                print(f"[skip render] {test_video}; {novel_video}")
            if args.force_metrics or not results.is_file():
                run([sys.executable, "metrics.py", "--model_paths", str(model)], eval_log)
            else:
                print(f"[skip metrics] {results}")
        except subprocess.CalledProcessError as error:
            failures.append((model.name, error.returncode))
            print(f"[failed] {model.name}; see {eval_log}", file=sys.stderr)

    rows = []
    for model in models:
        results_path = model / "results.json"
        if not results_path.is_file():
            continue
        result = json.loads(results_path.read_text())
        method = result.get(f"ours_{args.iteration}")
        if method is None and result:
            method = next(iter(result.values()))
        if method:
            rows.append({
                "scene": model.name,
                **method,
                "test_video": str(model / "test" / f"ours_{args.iteration}" / "video_rgb.mp4"),
                "novel_view_video": str(model / "video" / f"ours_{args.iteration}" / "video_rgb.mp4"),
            })

    metric_keys = ["PSNR", "SSIM", "LPIPS-vgg", "LPIPS-alex", "MS-SSIM", "D-SSIM"]
    if rows:
        average = {key: sum(float(row[key]) for row in rows) / len(rows) for key in metric_keys}
        summary_json = output_root / f"evaluation_summary_{args.suffix}.json"
        summary_csv = output_root / f"evaluation_summary_{args.suffix}.csv"
        summary_json.write_text(json.dumps({"scenes": rows, "average": average}, indent=2))
        with summary_csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=[
                "scene", *metric_keys, "test_video", "novel_view_video"])
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow({"scene": "AVERAGE", **average, "test_video": "", "novel_view_video": ""})
        print("\nFinal metrics")
        print(f"{'scene':45s} {'PSNR':>8s} {'SSIM':>8s} {'LPIPS':>8s}")
        for row in rows:
            print(f"{row['scene']:45s} {row['PSNR']:8.3f} {row['SSIM']:8.4f} {row['LPIPS-vgg']:8.4f}")
        print(f"{'AVERAGE':45s} {average['PSNR']:8.3f} {average['SSIM']:8.4f} {average['LPIPS-vgg']:8.4f}")
        print(f"Summary: {summary_csv}")
        print(f"Test videos: each scene's test/ours_{args.iteration}/video_rgb.mp4")
        print(f"Novel-view videos: each scene's video/ours_{args.iteration}/video_rgb.mp4")
    if failures:
        raise SystemExit("Failed scenes: " + ", ".join(name for name, _ in failures))


if __name__ == "__main__":
    main()
