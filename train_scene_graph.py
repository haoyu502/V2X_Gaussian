"""First milestone: fixed-topology rigid scene graph with sequential view losses.

This entrypoint does not instantiate the global deformation model. It is a
geometry/gradient validation trainer, not yet a quality-equivalent E1 benchmark
(densification, pose refinement and the full evaluation suite are future work).
"""

import argparse
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from gaussian_renderer.scene_graph_renderer import render_scene_graph
from scene.scene_graph.dataset import SceneGraphDataset, load_graph, load_manifest
from utils.loss_utils import ssim


def evaluate(graph, dataset, output, pipe, bg):
    output.mkdir(parents=True, exist_ok=True)
    records = []
    with torch.no_grad():
        for index in range(len(dataset)):
            camera = dataset[index]
            prediction = render_scene_graph(camera, graph, pipe, bg)["render"].clamp(0, 1)
            target = camera.original_image.to(prediction.device)
            mse = (prediction-target).square().mean()
            psnr = float(-10*torch.log10(mse.clamp_min(1e-12)))
            records.append({"image": camera.image_name, "agent": camera.agent,
                            "timestamp_ns": camera.timestamp_ns, "psnr": psnr,
                            "ssim": float(ssim(prediction, target))})
            image = (prediction.permute(1, 2, 0).cpu().numpy()*255).round().astype(np.uint8)
            Image.fromarray(image).save(output / f"{index:05d}_{camera.agent}.png")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--sh-degree", type=int, default=0, choices=range(4))
    parser.add_argument("--initial-scale", type=float, default=0.05)
    parser.add_argument("--position-lr", type=float, default=1.6e-4)
    parser.add_argument("--max-gap-sec", type=float, default=0.5)
    parser.add_argument("--downsample", type=float, default=1)
    parser.add_argument("--collab-weight", type=float, default=1)
    parser.add_argument("--lambda-dssim", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", help="Resume only trusted locally generated checkpoints")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA GPU required for rasterization")
    if args.iterations < 1 or not np.isfinite(args.collab_weight) or args.collab_weight < 0 or not 0 <= args.lambda_dssim <= 1:
        parser.error("Expected positive iterations, finite nonnegative collaborator weight and DSSIM weight in [0,1]")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train = SceneGraphDataset(args.manifest, "train", args.downsample)
    test = SceneGraphDataset(args.manifest, "test", args.downsample)
    if not len(train):
        parser.error("Manifest contains no train cameras")
    graph = load_graph(args.manifest, args.sh_degree, args.initial_scale, args.max_gap_sec)
    manifest, root = load_manifest(args.manifest)
    fingerprint = hashlib.sha256(Path(args.manifest).read_bytes() + (root / manifest["tracks"]).read_bytes()).hexdigest()
    optimizer = torch.optim.Adam(graph.optimizer_groups(args.position_lr), lr=0, eps=1e-15)
    start = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cuda", weights_only=False)
        if checkpoint.get("input_fingerprint") != fingerprint:
            parser.error("Resume manifest/tracklets differ from the checkpoint's inputs")
        saved_args = checkpoint["args"]
        for name in ("sh_degree", "max_gap_sec", "downsample", "collab_weight", "lambda_dssim"):
            if saved_args[name] != getattr(args, name):
                parser.error(f"Resume setting mismatch: {name}")
        graph.load_state_dict(checkpoint["graph"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        random.setstate(checkpoint["random_state"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        torch.cuda.set_rng_state(checkpoint["cuda_rng"].cpu())
        start = checkpoint["iteration"]
        if args.iterations <= start:
            parser.error("--iterations must exceed the resumed checkpoint iteration")
    pipe, bg = SimpleNamespace(debug=False), torch.zeros(3, device="cuda")
    groups = list(train.groups.values())
    history = []
    for iteration in range(start+1, args.iterations+1):
        cameras = [train[i] for i in random.choice(groups)]
        weights = [1.0 if c.agent == "ego" else args.collab_weight for c in cameras]
        weight_sum = sum(weights)
        if weight_sum <= 0:
            raise ValueError("Sampled group has no camera with positive weight")
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for camera, weight in zip(cameras, weights):
            if weight == 0:
                continue
            prediction = render_scene_graph(camera, graph, pipe, bg)["render"]
            target = camera.original_image.to(device="cuda")
            loss = (1-args.lambda_dssim)*(prediction-target).abs().mean()
            if args.lambda_dssim:
                loss = loss + args.lambda_dssim*(1-ssim(prediction, target))
            loss = loss * (weight/weight_sum)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at iteration {iteration}: {camera.image_name}")
            loss.backward()
            total += float(loss.detach())
        optimizer.step()
        history.append({"iteration": iteration, "loss": total})
        if iteration == start+1 or iteration % 100 == 0 or iteration == args.iterations:
            print(f"iteration={iteration} loss={total:.6f}", flush=True)
    # Save topology specification alongside the state; graph is reconstructed from
    # the same manifest before loading. No baseline checkpoint is overwritten.
    torch.save({"graph": graph.state_dict(), "optimizer": optimizer.state_dict(),
                "iteration": max(start, args.iterations), "args": vars(args),
                "input_fingerprint": fingerprint, "manifest": manifest,
                "random_state": random.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state()}, output / "checkpoint.pt")
    metrics = {"train_history": history, "test": evaluate(graph, test, output / "test", pipe, bg)}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output / "run_args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"Saved {output / 'checkpoint.pt'}; held-out cameras={len(test)}", flush=True)


if __name__ == "__main__":
    main()
