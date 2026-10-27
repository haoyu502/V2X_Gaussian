import json
from pathlib import Path
from torch import nn
from .actor_pose import ActorPose


class TrackletManager(nn.Module):
    def __init__(self, poses, time_origin_ns):
        super().__init__()
        if not isinstance(time_origin_ns, int):
            raise ValueError("time_origin_ns must be an integer, not floating-point epoch seconds")
        self.time_origin_ns = time_origin_ns
        self.poses = nn.ModuleDict(poses)

    def is_active(self, track_id, timestamp):
        return self.poses[track_id].is_active(timestamp)

    def pose(self, track_id, timestamp):
        return self.poses[track_id](timestamp)

    def get_extra_state(self):
        return {"time_origin_ns": self.time_origin_ns, "world_frame": getattr(self, "world_frame", None)}

    def set_extra_state(self, state):
        self.time_origin_ns = state["time_origin_ns"]
        self.world_frame = state["world_frame"]

    @classmethod
    def from_json(cls, path, max_gap_sec=0.5):
        payload = json.loads(Path(path).read_text())
        if payload.get("quaternion_order") != "wxyz" or not payload.get("world_frame"):
            raise ValueError("Track file must explicitly declare quaternion_order=wxyz and world_frame")
        origin = payload["time_origin_ns"]
        if not isinstance(origin, int):
            raise ValueError("time_origin_ns must be an integer")
        poses = {}
        for obj in payload["objects"]:
            track_id = str(obj["track_id"])
            if track_id in poses:
                raise ValueError(f"Duplicate track_id: {track_id}")
            obs = obj["observations"]
            if any(not isinstance(o["timestamp_ns"], int) for o in obs):
                raise ValueError("Observation timestamps must be integer nanoseconds")
            poses[track_id] = ActorPose(
                [(o["timestamp_ns"]-origin)*1e-9 for o in obs],
                [o["position"] for o in obs], [o["rotation_wxyz"] for o in obs], max_gap_sec)
        result = cls(poses, origin)
        result.world_frame = payload["world_frame"]
        return result
