"""Prepare a small scene graph directly from the local TUMTraf V2X ZIP.

Raw infrastructure LiDAR only: fused vehicle/RSU points have different capture
times and must not be treated as a single synchronized cloud. Box poses remain
oracle annotations, with all-camera holdout groups and train-only point seeds.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image, ImageDraw
from scene.scene_graph.actor_pose import ActorPose
from scene.scene_graph.preprocessing import RIGID_CLASSES, parse_objects, read_pcd_xyz, split_points, voxel_sample
from scene.scene_graph.timestamps import SENSORS, timestamp_ns_from_filename


def decompose_projection(projection):
    k, rotation, center, *_ = cv2.decomposeProjectionMatrix(projection)
    k = k/k[2, 2]
    if abs(k[0, 1]) > 0.01:
        raise ValueError("Rasterizer camera does not support significant intrinsic skew")
    k[0, 1] = 0
    w2c = np.eye(4)
    w2c[:3, :3] = rotation
    w2c[:3, 3] = (-rotation @ (center[:3]/center[3])).ravel()
    return k, w2c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-ns", type=int, required=True, help="First infrastructure frame, integer epoch nanoseconds")
    parser.add_argument("--frames", type=int, default=12, help="Includes one support frame at each end")
    parser.add_argument("--holdout-every", type=int, default=5)
    parser.add_argument("--max-gap-sec", type=float, default=0.5)
    parser.add_argument("--voxel-size", type=float, default=0.1)
    parser.add_argument("--max-bg-points", type=int, default=200000)
    parser.add_argument("--max-actor-points", type=int, default=10000)
    parser.add_argument("--box-margin", type=float, default=0.1)
    parser.add_argument("--track-id", help="Optional single actor; other labeled boxes are still removed from background")
    parser.add_argument("--calibration", default=str(Path(__file__).resolve().parents[1] / "assets/scene_graph_tumtraf_calibration.json"))
    args = parser.parse_args()
    if args.frames < 4 or args.holdout_every < 2:
        parser.error("At least 4 frames and holdout-every >=2 required")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        parser.error("Output already contains a manifest; choose a fresh directory")
    calibration = json.loads(Path(args.calibration).read_text())
    if calibration["world_frame"] != "s110_lidar_ouster_south":
        parser.error("Only infrastructure-LiDAR-frame labels are supported")
    for folder in ("images", "actors", "overlays"):
        (root / folder).mkdir(exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        # Resolve every input by its declared filename, never by sorted zip().
        by_name = {}
        label_paths = []
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            basename = Path(name).name
            if basename in by_name:
                raise ValueError(f"Ambiguous duplicate input basename: {basename}")
            by_name[basename] = name
            if "/labels_point_clouds/" in name and name.endswith(".json"):
                label_paths.append(name)
        labels = sorted((p for p in label_paths if timestamp_ns_from_filename(p) >= args.start_ns),
                        key=timestamp_ns_from_filename)[:args.frames]
        if len(labels) != args.frames:
            raise ValueError("Not enough metadata frames at the requested start")
        times = [timestamp_ns_from_filename(p) for p in labels]
        if np.any(np.diff(times)*1e-9 > args.max_gap_sec):
            raise ValueError("Requested slice crosses a scene/time gap")
        origin = times[0]
        tracks, bg_chunks, actor_chunks, cameras, counts = {}, [], defaultdict(list), [], []
        frames = []
        for path, timestamp in zip(labels, times):
            payload = json.loads(archive.read(path))
            frame_values = list(payload["openlabel"]["frames"].values())
            if len(frame_values) != 1:
                raise ValueError("Expected one frame per metadata file")
            frame = frame_values[0]
            objects = parse_objects(frame)
            frames.append((path, timestamp, frame, objects))
            for track_id, obj in objects.items():
                if obj["class"] not in RIGID_CLASSES or (args.track_id and track_id != args.track_id):
                    continue
                track = tracks.setdefault(track_id, {"track_id": track_id, "class": obj["class"],
                                                      "size": obj["size"], "observations": []})
                track["observations"].append({"timestamp_ns": timestamp, "position": obj["position"],
                                               "rotation_wxyz": obj["rotation_wxyz"]})
        if not tracks:
            raise ValueError("No matching rigid actors in the requested slice")
        # Ego camera extrinsics must follow its own image time as well. The
        # metadata does not separately timestamp its transform: the declared
        # convention here anchors transforms to the metadata reference time.
        ego_positions, ego_rotations = [], []
        ego_intrinsics = None
        for _, _, frame, _ in frames:
            transform = frame["frame_properties"]["transforms"]["vehicle_lidar_robosense_to_s110_lidar_ouster_south"]
            vehicle_to_world = np.asarray(transform["transform_src_to_dst"]["matrix4x4"])
            ego_projection = np.asarray(calibration["projections"]["vehicle_camera_basler_16mm"]) @ np.linalg.inv(vehicle_to_world)
            ego_intrinsics, ego_w2c = decompose_projection(ego_projection)
            c2w = np.linalg.inv(ego_w2c)
            axis_angle = cv2.Rodrigues(c2w[:3, :3])[0].ravel()
            angle = np.linalg.norm(axis_angle)
            quaternion = np.r_[np.cos(angle/2), axis_angle*np.sin(angle/2)/angle] if angle > 1e-10 else np.array([1., 0., 0., 0.])
            ego_positions.append(c2w[:3, 3])
            ego_rotations.append(quaternion)
        ego_pose = ActorPose([(t-origin)*1e-9 for t in times], np.asarray(ego_positions), np.asarray(ego_rotations), args.max_gap_sec)
        for index, (path, timestamp, frame, objects) in enumerate(frames):
            # End frames support interpolation only; they do not seed geometry or images.
            if index in (0, len(frames)-1):
                continue
            split = "test" if index % args.holdout_every == 0 else "train"
            properties = frame["frame_properties"]
            images = [f for f in properties["image_file_names"] if Path(f).stem.split("_", 2)[2] in SENSORS]
            if len(images) != 3:
                raise ValueError("Expected exactly the ego, south1 and south2 image names")
            for name in images:
                sensor = Path(name).stem.split("_", 2)[2]
                projection = np.asarray(calibration["projections"][sensor], dtype=np.float64)
                if sensor == "vehicle_camera_basler_16mm":
                    rotation, position, _ = ego_pose((timestamp_ns_from_filename(name)-origin)*1e-9)
                    k, w2c = ego_intrinsics, np.eye(4)
                    w2c[:3, :3] = rotation.numpy().T
                    w2c[:3, 3] = -rotation.numpy().T @ position.numpy()
                    projection = k @ w2c[:3]
                else:
                    k, w2c = decompose_projection(projection)
                image_path = root / "images" / name
                image_path.write_bytes(archive.read(by_name[name]))
                cameras.append({"image": str(image_path.relative_to(root)), "agent": SENSORS[sensor],
                                "timestamp_ns": timestamp_ns_from_filename(name), "sync_group_id": str(timestamp),
                                "split": split, "intrinsics": k.tolist(), "world_to_camera": w2c.tolist(),
                                "source_metadata": path})
                # Independent calibration inspection: annotate GT centers at the
                # LiDAR reference time. Pixel offsets due to image time remain.
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    draw = ImageDraw.Draw(image)
                    for track_id, obj in objects.items():
                        if track_id not in tracks:
                            continue
                        homogeneous = np.r_[obj["position"], 1.]
                        pixel = projection @ homogeneous
                        if pixel[2] <= 0:
                            continue
                        u, v = pixel[:2]/pixel[2]
                        if 0 <= u < image.width and 0 <= v < image.height:
                            draw.ellipse((u-5, v-5, u+5, v+5), fill="red")
                            draw.text((u+6, v), track_id[:8], fill="yellow")
                    if index == 1:
                        image.save(root / "overlays" / f"centers_{SENSORS[sensor]}.jpg")
            if split == "train":
                pcd_names = [name for name in properties["point_cloud_file_names"]
                             if name.endswith("_s110_lidar_ouster_south.pcd")]
                if len(pcd_names) != 1 or timestamp_ns_from_filename(pcd_names[0]) != timestamp:
                    raise ValueError("Infrastructure point cloud must match the box timestamp exactly")
                points = read_pcd_xyz(archive.read(by_name[pcd_names[0]]))
                # Limit distant outliers but retain the observed intersection.
                points = points[np.linalg.norm(points, axis=1) < 150]
                background, actors = split_points(points, objects, args.box_margin)
                bg_chunks.append(voxel_sample(background, args.voxel_size, args.max_bg_points))
                for track_id, local in actors.items():
                    if track_id in tracks and len(local):
                        actor_chunks[track_id].append(local)
                counts.append({"timestamp_ns": timestamp, "raw": len(points), "background": len(background),
                               "actor_points": {k: len(v) for k, v in actors.items() if k in tracks}})
        if not bg_chunks:
            raise ValueError("No training point clouds selected")
        bg = voxel_sample(np.concatenate(bg_chunks), args.voxel_size, args.max_bg_points)
        np.savez_compressed(root / "background.npz", xyz=bg, rgb=np.full_like(bg, 0.5))
        actor_paths = {}
        for track_id, chunks in actor_chunks.items():
            xyz = voxel_sample(np.concatenate(chunks), args.voxel_size, args.max_actor_points)
            actor_path = Path("actors") / f"{track_id}.npz"
            np.savez_compressed(root / actor_path, xyz=xyz, rgb=np.full_like(xyz, 0.5))
            actor_paths[track_id] = str(actor_path)
        if not actor_paths:
            raise ValueError("No selected actor has training LiDAR points")
        tracks_payload = {"world_frame": calibration["world_frame"], "time_origin_ns": origin,
                          "quaternion_order": "wxyz", "objects": [tracks[k] for k in actor_paths]}
        (root / "tracks.json").write_text(json.dumps(tracks_payload, indent=2))
        manifest = {"schema_version": 1, "world_frame": calibration["world_frame"], "time_origin_ns": origin,
                    "tracks": "tracks.json", "background": "background.npz", "actors": actor_paths,
                    "cameras": cameras, "oracle_tracks": True,
                    "initialization": "train-only infrastructure LiDAR, neutral gray RGB, fixed topology",
                    "excluded_classes": "All non-rigid labeled classes are removed, not reconstructed",
                    "source_archive": str(Path(args.archive).resolve()), "calibration_source": calibration["source"],
                    "ego_pose_time_assumption": "metadata reference timestamp; interpolate camera pose to each image timestamp",
                    "support_metadata": [frames[0][0], frames[-1][0]], "prepare_args": vars(args)}
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (root / "point_counts.json").write_text(json.dumps(counts, indent=2))
        print(f"Prepared {len(bg)} background points, {len(actor_paths)} actors and {len(cameras)} cameras")
        print(f"Manifest: {root / 'manifest.json'}")


if __name__ == "__main__":
    main()
