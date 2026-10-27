from dataclasses import dataclass, fields
import torch
from torch import nn
from torch.nn import functional as F
from utils.sh_utils import eval_sh
from .actor_model import GaussianPack, transform_actor_gaussians


@dataclass(frozen=True)
class ComposedGaussians:
    pack: GaussianPack
    node_slices: dict
    node_rotations: dict


class GaussianSceneGraph(nn.Module):
    def __init__(self, background, actors, tracklets):
        super().__init__()
        if "background" in actors or set(actors) - set(tracklets.poses):
            raise ValueError("Actor IDs must have tracklets and cannot be named 'background'")
        if any(actor.sh_degree != background.sh_degree for actor in actors.values()):
            raise ValueError("All graph nodes must use the same SH degree")
        self.background = background
        self.actors = nn.ModuleDict(actors)
        self.tracklets = tracklets
        self.sh_degree = background.sh_degree

    def compose(self, timestamp):
        packs = [self.background.get_gaussian_pack()]
        count = len(packs[0].xyz)
        slices = {"background": slice(0, count)}
        rotations = {"background": torch.eye(3, device=packs[0].xyz.device, dtype=packs[0].xyz.dtype)}
        for track_id, actor in self.actors.items():
            if not self.tracklets.is_active(track_id, timestamp):
                continue
            rotation, translation, quaternion = self.tracklets.pose(track_id, timestamp)
            pack = transform_actor_gaussians(actor.get_gaussian_pack(), rotation, translation, quaternion)
            packs.append(pack)
            slices[track_id] = slice(count, count+len(pack.xyz))
            rotations[track_id] = rotation
            count += len(pack.xyz)
        pack = GaussianPack(**{field.name: torch.cat([getattr(p, field.name) for p in packs])
                               for field in fields(GaussianPack)})
        return ComposedGaussians(pack, slices, rotations)

    def colors(self, composed, camera_center):
        """Evaluate SH in each actor's canonical frame, including rotating actors."""
        colors = []
        pack = composed.pack
        for node, node_slice in composed.node_slices.items():
            # Row-vector convention: world viewing direction -> local is d @ R.
            directions = F.normalize((pack.xyz[node_slice]-camera_center)
                                     @ composed.node_rotations[node], dim=-1)
            colors.append((eval_sh(self.sh_degree, pack.features[node_slice].transpose(1, 2), directions)
                           + 0.5).clamp_min(0))
        return torch.cat(colors)

    def optimizer_groups(self, position_lr=1.6e-4):
        groups = self.background.optimizer_groups("background", position_lr)
        for track_id, actor in self.actors.items():
            groups.extend(actor.optimizer_groups(track_id, position_lr))
        return groups
