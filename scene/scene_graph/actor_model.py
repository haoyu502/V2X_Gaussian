"""Trainable fixed-topology local Gaussians; no per-object deformation MLP."""

from dataclasses import dataclass, replace
import math
import torch
from torch import nn
from torch.nn import functional as F
from utils.sh_utils import RGB2SH
from .actor_pose import quaternion_multiply


@dataclass(frozen=True)
class GaussianPack:
    # All fields are ACTIVATED. In particular scaling is not log-scale,
    # rotation is a unit wxyz quaternion, opacity is not a logit.
    xyz: torch.Tensor
    scaling: torch.Tensor
    rotation: torch.Tensor
    opacity: torch.Tensor
    features: torch.Tensor

    def replace(self, **kwargs):
        return replace(self, **kwargs)


def transform_actor_gaussians(pack, rotation, translation, pose_quaternion):
    return pack.replace(
        xyz=pack.xyz @ rotation.T + translation,
        rotation=F.normalize(quaternion_multiply(pose_quaternion, pack.rotation), dim=-1),
    )


class ActorGaussianModel(nn.Module):
    def __init__(self, xyz, rgb, sh_degree=0, initial_scale=0.05, initial_opacity=0.1):
        super().__init__()
        xyz = torch.as_tensor(xyz, dtype=torch.float32)
        rgb = torch.as_tensor(rgb, dtype=torch.float32, device=xyz.device)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape:
            raise ValueError("xyz and rgb must both have shape [N,3]")
        if not torch.isfinite(xyz).all() or not torch.isfinite(rgb).all() or ((rgb < 0) | (rgb > 1)).any():
            raise ValueError("Points must be finite; RGB must be finite and in [0,1]")
        if sh_degree not in range(4) or not math.isfinite(initial_scale) or initial_scale <= 0 or not 0 < initial_opacity < 1:
            raise ValueError("Expected SH degree 0..3, positive scale and opacity in (0,1)")
        self.sh_degree = sh_degree
        self.xyz = nn.Parameter(xyz.clone())
        self.log_scaling = nn.Parameter(torch.full_like(xyz, float(initial_scale)).log())
        rotation = xyz.new_zeros((len(xyz), 4))
        rotation[:, 0] = 1
        self.rotation_raw = nn.Parameter(rotation)
        self.opacity_logits = nn.Parameter(torch.logit(xyz.new_full((len(xyz), 1), initial_opacity)))
        features = xyz.new_zeros((len(xyz), (sh_degree+1)**2, 3))
        features[:, 0] = RGB2SH(rgb)
        self.features = nn.Parameter(features)

    def get_gaussian_pack(self):
        return GaussianPack(self.xyz, self.log_scaling.exp(),
                            F.normalize(self.rotation_raw, dim=-1),
                            self.opacity_logits.sigmoid(), self.features)

    def optimizer_groups(self, prefix, position_lr=1.6e-4):
        return [{"params": [param], "lr": lr, "name": f"{prefix}/{name}"}
                for name, param, lr in (
                    ("xyz", self.xyz, position_lr), ("features", self.features, 0.0025),
                    ("opacity", self.opacity_logits, 0.05), ("scaling", self.log_scaling, 0.005),
                    ("rotation", self.rotation_raw, 0.001))]
