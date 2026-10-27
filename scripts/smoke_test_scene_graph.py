"""Generate a tiny THREE-CAMERA SYNTHETIC fixture with CUDA-rendered targets.

Use this to validate plumbing/optimization. Its scores are not research results.
"""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from scene.scene_graph.dataset import SceneGraphDataset, load_graph
from gaussian_renderer.scene_graph_renderer import render_scene_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/scene_graph_smoke")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        parser.error("Output already contains a manifest; choose a fresh directory")
    origin = 1688625741000000000
    rng = np.random.default_rng(0)
    bg_xyz = rng.uniform([-2, -1.5, 5], [2, 1.5, 5.1], size=(150, 3)).astype(np.float32)
    np.savez(root / "background.npz", xyz=bg_xyz, rgb=np.tile([0.2, 0.3, 0.5], (150, 1)))
    xyz = rng.uniform([-0.5, -0.2, -0.2], [0.5, 0.2, 0.2], size=(100, 3)).astype(np.float32)
    rgb = np.tile([0.8, 0.2, 0.1], (100, 1))
    np.savez(root / "car.npz", xyz=xyz, rgb=rgb)
    tracks = {"time_origin_ns": origin, "world_frame": "synthetic_world", "quaternion_order": "wxyz",
              "objects": [{"track_id": "car", "class": "CAR", "observations": [
                  {"timestamp_ns": origin, "position": [-0.3, 0, 4], "rotation_wxyz": [1, 0, 0, 0]},
                  {"timestamp_ns": origin+200000000, "position": [0.3, 0, 4], "rotation_wxyz": [1, 0, 0, 0]}]}]}
    (root / "tracks.json").write_text(json.dumps(tracks, indent=2))
    entries = []
    for group, start in enumerate([20000000, 120000000]):
        for agent, x, delay in [("ego", 0, 0), ("rsu_south1", -0.6, 10000000), ("rsu_south2", 0.6, 20000000)]:
            w2c = np.eye(4)
            w2c[0, 3] = -x
            image = f"{group}_{agent}.png"
            Image.new("RGB", (96, 64)).save(root / image)
            entries.append({"image": image, "agent": agent, "timestamp_ns": origin+start+delay,
                            "sync_group_id": str(group), "split": "train" if group == 0 else "test",
                            "intrinsics": [[90, 0, 48], [0, 90, 32], [0, 0, 1]],
                            "world_to_camera": w2c.tolist()})
    manifest = {"schema_version": 1, "world_frame": "synthetic_world", "time_origin_ns": origin,
                "tracks": "tracks.json", "background": "background.npz", "actors": {"car": "car.npz"},
                "cameras": entries, "synthetic_fixture": True}
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    graph = load_graph(path, initial_scale=0.12)
    for split in ("train", "test"):
        dataset = SceneGraphDataset(path, split)
        with torch.no_grad():
            for i in range(len(dataset)):
                camera = dataset[i]
                image = render_scene_graph(camera, graph, SimpleNamespace(debug=False), torch.zeros(3, device="cuda"))["render"]
                rgb = (image.clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).round().astype(np.uint8)
                Image.fromarray(rgb).save(root / camera.image_name)
    # Start optimization with a different appearance from the target model.
    np.savez(root / "car.npz", xyz=xyz, rgb=np.tile([0.4, 0.4, 0.4], (100, 1)))
    print(f"Created synthetic fixture: {path}")


if __name__ == "__main__":
    main()
