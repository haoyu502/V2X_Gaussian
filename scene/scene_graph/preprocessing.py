"""Per-frame point splitting. Never feed a temporally merged PLY into this."""

import io
import numpy as np
import torch
from .actor_pose import quaternion_to_matrix


RIGID_CLASSES = {"CAR", "VAN", "BUS", "TRUCK", "TRAILER", "EMERGENCY_VEHICLE"}


def read_pcd_xyz(data):
    """Read ASCII or uncompressed binary PCD, without requiring Open3D."""
    stream = io.BytesIO(data)
    header = {}
    while True:
        line = stream.readline()
        if not line:
            raise ValueError("PCD is missing its DATA header")
        line = line.decode("ascii").strip()
        if not line or line.startswith("#"):
            continue
        key, *values = line.split()
        header[key.upper()] = values
        if key.upper() == "DATA":
            break
    fields = header["FIELDS"]
    counts = [int(c) for c in header.get("COUNT", ["1"]*len(fields))]
    if any(c != 1 for c in counts):
        raise ValueError("Only scalar-field PCD records are supported")
    mode = header["DATA"][0]
    if mode == "ascii":
        points = np.loadtxt(stream, usecols=[fields.index(x) for x in ("x", "y", "z")], dtype=np.float32, ndmin=2)
    elif mode == "binary":
        types = {"F": "f", "U": "u", "I": "i"}
        dtype = np.dtype([(f, "<"+types[t]+s) for f, t, s in zip(fields, header["TYPE"], header["SIZE"])])
        records = np.frombuffer(stream.read(), dtype=dtype)
        points = np.column_stack([records[f] for f in ("x", "y", "z")]).astype(np.float32)
    else:
        raise ValueError(f"PCD DATA {mode} is unsupported; convert to ASCII or uncompressed binary first")
    expected = int(header.get("POINTS", [len(points)])[0])
    if len(points) != expected:
        raise ValueError(f"PCD contains {len(points)} records, expected {expected}")
    valid = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) > 1e-5)
    return points[valid]


def parse_objects(frame):
    objects = {}
    for track_id, item in frame.get("objects", {}).items():
        data = item["object_data"]
        cuboid = data["cuboid"]
        values = np.asarray(cuboid["val"], dtype=np.float64)
        if values.shape != (10,) or not np.isfinite(values).all() or (values[7:] <= 0).any():
            raise ValueError(f"Invalid cuboid for {track_id}")
        attributes = {a["name"]: a["val"] for a in cuboid.get("attributes", {}).get("text", [])}
        if attributes.get("sensor_id") != "s110_lidar_ouster_south":
            raise ValueError(f"Unverified cuboid coordinate frame for {track_id}: {attributes.get('sensor_id')}")
        q = values[[6, 3, 4, 5]]  # OpenLABEL xyzw -> internal wxyz.
        if np.linalg.norm(q) < 1e-8:
            raise ValueError(f"Zero quaternion for {track_id}")
        q = q / np.linalg.norm(q)
        objects[track_id] = {"class": data["type"], "position": values[:3].tolist(),
                             "rotation_wxyz": q.tolist(), "size": values[7:].tolist()}
    return objects


def split_points(points, objects, margin=0.1):
    """All labeled boxes are removed from background, including excluded classes.

    Overlapping boxes assign a point once, to the nearest normalized box center.
    Actor output is in local canonical coordinates. Input points are world XYZ.
    """
    points = np.asarray(points, dtype=np.float32)
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("Box margin must be finite and nonnegative")
    owner = np.full(len(points), -1, dtype=np.int32)
    score = np.full(len(points), np.inf)
    local_points = {}
    for index, (track_id, obj) in enumerate(objects.items()):
        rotation = quaternion_to_matrix(torch.tensor(obj["rotation_wxyz"], dtype=torch.float32)).numpy()
        local = (points-np.asarray(obj["position"])) @ rotation
        extent = np.asarray(obj["size"])*0.5+margin
        inside = (np.abs(local) <= extent).all(axis=1)
        distance = np.square(local/extent).sum(axis=1)
        selected = inside & (distance < score)
        owner[selected], score[selected] = index, distance[selected]
        local_points[track_id] = local.astype(np.float32)
    actors = {track_id: local_points[track_id][owner == index]
              for index, track_id in enumerate(objects)}
    return points[owner < 0], actors


def voxel_sample(points, voxel_size, max_points, seed=0):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if voxel_size <= 0 or max_points < 1:
        raise ValueError("Positive voxel size and point cap required")
    if len(points):
        _, indices = np.unique(np.floor(points/voxel_size).astype(np.int64), axis=0, return_index=True)
        points = points[np.sort(indices)]
    if len(points) > max_points:
        points = points[np.sort(np.random.default_rng(seed).choice(len(points), max_points, replace=False))]
    return points
