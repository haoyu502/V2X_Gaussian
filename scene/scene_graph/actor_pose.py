"""Fixed tracklets with linear translation and shortest-path quaternion SLERP.

All quaternions in this package are wxyz, rotations are local-to-world and
timestamps are float64 seconds relative to ONE shared integer-nanosecond origin.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F


def quaternion_multiply(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz,
                        aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx,
                        aw*bz+ax*by-ay*bx+az*bw), dim=-1)


def quaternion_to_matrix(q):
    w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), dim=-1).reshape(*q.shape[:-1], 3, 3)


def quaternion_slerp(q0, q1, fraction):
    dot = (q0 * q1).sum()
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(0, 1)
    # The branch avoids singular acos gradients for identical rotations.
    if float(dot.detach()) > 0.9995:
        return F.normalize(q0 + fraction * (q1-q0), dim=-1)
    angle = torch.acos(dot)
    return F.normalize((torch.sin((1-fraction)*angle)*q0
                        + torch.sin(fraction*angle)*q1) / torch.sin(angle), dim=-1)


class ActorPose(nn.Module):
    def __init__(self, timestamps, positions, rotations_wxyz, max_gap_sec=0.5):
        super().__init__()
        t = torch.as_tensor(timestamps, dtype=torch.float64)
        p = torch.as_tensor(positions, dtype=torch.float32)
        q = torch.as_tensor(rotations_wxyz, dtype=torch.float32)
        if t.ndim != 1 or t.numel() == 0 or p.shape != (len(t), 3) or q.shape != (len(t), 4):
            raise ValueError("Expected timestamps [K], positions [K,3], quaternions [K,4], K>0")
        if not all(torch.isfinite(x).all() for x in (t, p, q)):
            raise ValueError("Tracklet values must be finite")
        if (t[1:] <= t[:-1]).any() or (q.norm(dim=-1) < 1e-8).any():
            raise ValueError("Track timestamps must be strictly increasing; quaternions must be nonzero")
        if not math.isfinite(max_gap_sec) or max_gap_sec <= 0:
            raise ValueError("max_gap_sec must be finite and positive")
        self.max_gap_sec = float(max_gap_sec)
        self.register_buffer("timestamps", t)
        self.register_buffer("positions", p)
        self.register_buffer("rotations_wxyz", F.normalize(q, dim=-1))

    def _interval(self, timestamp):
        t = float(timestamp)
        if not math.isfinite(t):
            raise ValueError("Query timestamp must be finite")
        times = self.timestamps
        if t < float(times[0]) or t > float(times[-1]):
            return None
        hi = min(int(torch.searchsorted(times, times.new_tensor(t))), len(times)-1)
        if abs(float(times[hi])-t) <= 1e-9:
            return hi, hi, 0.0
        lo = hi-1
        if lo < 0 or float(times[hi]-times[lo]) > self.max_gap_sec:
            return None
        return lo, hi, (t-float(times[lo])) / float(times[hi]-times[lo])

    def is_active(self, timestamp):
        return self._interval(timestamp) is not None

    def get_extra_state(self):
        return {"max_gap_sec": self.max_gap_sec}

    def set_extra_state(self, state):
        self.max_gap_sec = state["max_gap_sec"]

    def forward(self, timestamp):
        interval = self._interval(timestamp)
        if interval is None:
            raise ValueError(f"No supported pose at t={timestamp}; extrapolation and long gaps are disabled")
        lo, hi, alpha = interval
        p = torch.lerp(self.positions[lo], self.positions[hi], alpha)
        q = quaternion_slerp(self.rotations_wxyz[lo], self.rotations_wxyz[hi], alpha)
        return quaternion_to_matrix(q), p, q
