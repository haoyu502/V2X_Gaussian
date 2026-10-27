"""Explicit, calibrated inputs for the independent scene-graph training path."""

import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from utils.graphics_utils import getProjectionMatrix2
from .actor_model import ActorGaussianModel
from .background_model import BackgroundGaussianModel
from .graph_model import GaussianSceneGraph
from .tracklet_manager import TrackletManager


def load_manifest(path):
    path = Path(path).resolve()
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1 or not payload.get("world_frame"):
        raise ValueError("Expected scene-graph manifest schema_version=1 and explicit world_frame")
    if not isinstance(payload.get("time_origin_ns"), int):
        raise ValueError("time_origin_ns must be integer nanoseconds")
    return payload, path.parent


def load_graph(manifest_path, sh_degree=0, initial_scale=0.05, max_gap_sec=0.5, device="cuda"):
    manifest, root = load_manifest(manifest_path)
    tracklets = TrackletManager.from_json(root / manifest["tracks"], max_gap_sec)
    if (tracklets.time_origin_ns != manifest["time_origin_ns"]
            or tracklets.world_frame != manifest["world_frame"]):
        raise ValueError("Tracklets and manifest must have identical world frame and time origin")

    def node(path, model_type):
        with np.load(root / path, allow_pickle=False) as points:
            return model_type(points["xyz"], points["rgb"], sh_degree, initial_scale)

    background = node(manifest["background"], BackgroundGaussianModel)
    actors = {str(key): node(path, ActorGaussianModel) for key, path in manifest["actors"].items()}
    if not len(background.xyz) and not any(len(a.xyz) for a in actors.values()):
        raise ValueError("Scene graph has no initial points")
    return GaussianSceneGraph(background, actors, tracklets).to(device)


class SceneGraphDataset:
    """Camera transforms are explicitly world-to-camera, OpenCV axes.

    A shared sync_group_id is a sampling association only. Each camera retains
    its own filename/manifest time and queries the trajectory independently.
    """

    def __init__(self, manifest_path, split="train", downsample=1):
        self.manifest, self.root = load_manifest(manifest_path)
        if not math.isfinite(downsample) or downsample < 1:
            raise ValueError("downsample must be finite and >=1")
        self.downsample = downsample
        self.entries = [e for e in self.manifest["cameras"] if e["split"] == split]
        self.time_origin_ns = self.manifest["time_origin_ns"]
        seen = set()
        self.groups = {}
        for i, entry in enumerate(self.entries):
            if not isinstance(entry["timestamp_ns"], int):
                raise ValueError("Camera timestamps must be integer nanoseconds")
            key = (entry["agent"], entry["timestamp_ns"])
            if key in seen:
                raise ValueError(f"Duplicate camera observation: {key}")
            seen.add(key)
            self.groups.setdefault(str(entry["sync_group_id"]), []).append(i)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        with Image.open(self.root / entry["image"]) as image:
            width, height = image.size
            out_size = (max(1, int(width/self.downsample)), max(1, int(height/self.downsample)))
            image = image.convert("RGB").resize(out_size, Image.Resampling.LANCZOS)
            rgb = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()/255
        k = np.array(entry["intrinsics"], dtype=np.float32)
        w2c = np.array(entry["world_to_camera"], dtype=np.float32)
        if k.shape != (3, 3) or w2c.shape != (4, 4) or not np.isfinite(k).all() or not np.isfinite(w2c).all():
            raise ValueError("Camera intrinsics/extrinsics must be finite [3,3]/[4,4] matrices")
        if not np.allclose(w2c[3], [0, 0, 0, 1]) or not np.allclose(w2c[:3, :3].T @ w2c[:3, :3], np.eye(3), atol=1e-3):
            raise ValueError("world_to_camera must be a rigid homogeneous transform")
        k[0] *= out_size[0]/width
        k[1] *= out_size[1]/height
        if k[0, 0] <= 0 or k[1, 1] <= 0:
            raise ValueError("Focal lengths must be positive")
        view = torch.from_numpy(w2c.T.copy())
        projection = getProjectionMatrix2(0.01, 1000., k, *out_size).T
        return SimpleNamespace(
            original_image=rgb, image_width=out_size[0], image_height=out_size[1],
            FoVx=2*math.atan(out_size[0]/(2*k[0, 0])), FoVy=2*math.atan(out_size[1]/(2*k[1, 1])),
            world_view_transform=view, full_proj_transform=view @ projection,
            camera_center=view.inverse()[3, :3], image_name=entry["image"],
            timestamp_ns=entry["timestamp_ns"], time_origin_ns=self.time_origin_ns,
            timestamp_sec=(entry["timestamp_ns"]-self.time_origin_ns)*1e-9,
            agent=entry["agent"], sync_group_id=entry["sync_group_id"],
        )
