"""Lossless filename timestamp parsing and explicit sensor identity."""

from pathlib import Path
import re

SENSORS = {
    "vehicle_camera_basler_16mm": "ego",
    "s110_camera_basler_south1_8mm": "rsu_south1",
    "s110_camera_basler_south2_8mm": "rsu_south2",
}


def timestamp_ns_from_filename(path):
    match = re.match(r"^(\d+)_(\d{9})_", Path(path).name)
    if match is None:
        raise ValueError(f"Expected seconds_nanoseconds_sensor filename: {path}")
    return int(match[1])*1_000_000_000 + int(match[2])


def camera_metadata(path, time_origin_ns):
    if not isinstance(time_origin_ns, int):
        raise ValueError("time_origin_ns must be an integer")
    timestamp_ns = timestamp_ns_from_filename(path)
    sensor = Path(path).stem.split("_", 2)[2]
    if sensor not in SENSORS:
        raise ValueError(f"Unsupported camera sensor: {sensor}")
    return {"timestamp_ns": timestamp_ns,
            "timestamp_sec": (timestamp_ns-time_origin_ns)*1e-9,
            "time_origin_ns": time_origin_ns, "agent": SENSORS[sensor], "sensor_id": sensor}
