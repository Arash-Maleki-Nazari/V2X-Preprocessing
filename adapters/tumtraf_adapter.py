"""Adapter for the extracted TUMTraf-V2X_mini OpenLABEL dataset.

Reference coordinates: infrastructure Ouster LiDAR.
Vehicle coordinates: native vehicle RoboSense LiDAR, with T_ref_from_agent
provided by OpenLABEL. The registered/fused cloud is NOT a third agent.

Cameras are represented as source paths in extras only: this mini archive
has no verified camera intrinsics/extrinsics to populate CameraData.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from adapters.base_adapter import BaseAdapter
from schema import AgentData, AnnotationData, CanonicalV2XSample, LidarData, SourceInfo


INFRA_LIDAR = "s110_lidar_ouster_south"
VEHICLE_LIDAR = "vehicle_lidar_robosense"
FUSED_LIDAR = "s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered"
TRANSFORM_KEY = f"{VEHICLE_LIDAR}_to_{INFRA_LIDAR}"


def read_ascii_pcd(path: Path) -> LidarData:
    """Load xyz/intensity from an ASCII PCD, preserving extra fields."""
    fields = None
    declared_points = None
    with path.open("rb") as stream:
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError(f"Missing DATA line in {path}")
            line = raw.decode("ascii").strip()
            parts = line.split()
            if not parts:
                continue
            key = parts[0].upper()
            if key == "FIELDS":
                fields = parts[1:]
            elif key == "POINTS":
                declared_points = int(parts[1])
            elif key == "DATA":
                if parts[1].lower() != "ascii":
                    raise NotImplementedError(f"Only ASCII PCD is supported: {path}")
                break

        if fields is None or not all(name in fields for name in ("x", "y", "z")):
            raise ValueError(f"Missing XYZ fields in {path}")

        # Read remaining rows from the current stream position.
        # Float64 parsing also preserves the original integer-valued PCD columns.
        values = np.loadtxt(stream, dtype=np.float64, ndmin=2)

    if values.shape[1] != len(fields):
        raise ValueError(f"PCD field count mismatch: {path}")
    if declared_points is not None and values.shape[0] != declared_points:
        raise ValueError(f"PCD point count mismatch: {path}")

    xyz = values[:, [fields.index(k) for k in ("x", "y", "z")]].astype(np.float32)
    intensity = (values[:, fields.index("intensity")].astype(np.float32)
                 if "intensity" in fields else None)
    # Raw Ouster scans can contain nonfinite coordinates for invalid returns.
    # Filter all per-point fields with the same mask to preserve alignment.
    valid = np.isfinite(xyz).all(axis=1)
    removed = int((~valid).sum())
    xyz = xyz[valid]
    if intensity is not None:
        # Invalid intensity is also unusable for canonical point attributes.
        valid_intensity = np.isfinite(intensity)
        if not valid_intensity.all():
            # Apply an additional shared mask to every point-aligned field.
            xyz = xyz[valid_intensity[valid]]
            removed += int((~valid_intensity[valid]).sum())
            valid_indices = np.flatnonzero(valid)[valid_intensity[valid]]
        else:
            valid_indices = np.flatnonzero(valid)
        intensity = intensity[valid_indices]
    else:
        valid_indices = np.flatnonzero(valid)
    extra_fields = {name: values[valid_indices, i] for i, name in enumerate(fields)
                    if name not in {"x", "y", "z", "intensity"}}
    return LidarData(xyz=xyz, intensity=intensity,
                     extras={"path": str(path), "pcd_fields": extra_fields,
                             "num_invalid_points_removed": removed,
                             "num_original_points": int(values.shape[0])})


def quaternion_to_yaw(quaternion: list[float]) -> float:
    """OpenLABEL quaternion order: [qx, qy, qz, qw]."""
    qx, qy, qz, qw = map(float, quaternion)
    norm = np.linalg.norm([qx, qy, qz, qw])
    if norm == 0:
        raise ValueError("Zero-length cuboid quaternion")
    qx, qy, qz, qw = np.array([qx, qy, qz, qw]) / norm
    sin_yaw = 2 * (qw * qz + qx * qy)
    cos_yaw = 1 - 2 * (qy * qy + qz * qz)
    return float(np.arctan2(sin_yaw, cos_yaw))


def parse_annotations(objects: dict) -> AnnotationData:
    boxes = []
    native_labels = []
    object_ids = []
    quaternions = []
    for object_id, obj in objects.items():
        data = obj.get("object_data", {})
        cuboid = data.get("cuboid")
        if not cuboid:
            continue
        if isinstance(cuboid, list):
            # OpenLABEL permits a list of cuboids; this mini sample uses one.
            cuboid = cuboid[0] if cuboid else None
        if not cuboid:
            continue
        val = cuboid.get("val", [])
        if len(val) != 10:
            raise ValueError(f"Expected 10 cuboid values for {object_id}")
        x, y, z = val[:3]
        quaternion = val[3:7]
        length, width, height = val[7:10]
        boxes.append([x, y, z, length, width, height,
                      quaternion_to_yaw(quaternion)])
        native_labels.append(data.get("type", "UNKNOWN"))
        object_ids.append(object_id)
        quaternions.append(quaternion)

    return AnnotationData(
        boxes_3d=np.asarray(boxes, dtype=np.float32).reshape(-1, 7),
        native_labels=native_labels,
        # OpenLABEL UUIDs are not guaranteed to be integer tracking IDs.
        track_ids=np.full(len(boxes), -1, dtype=np.int64),
        labels=None,
        extras={"openlabel_object_ids": object_ids,
                "openlabel_quaternions_xyzw": quaternions},
    )


class TUMTrafAdapter(BaseAdapter):
    """Load TUMTraf-V2X_mini from root containing train/val/test folders."""

    def __init__(self, dataset_root: str | Path, split: str = "train") -> None:
        super().__init__(dataset_root, split)
        if split not in ("train", "val", "test"):
            raise ValueError("split must be one of 'train', 'val', 'test'")
        self.split_root = self.dataset_root / split
        label_dir = self.split_root / "labels_point_clouds" / FUSED_LIDAR
        if not label_dir.is_dir():
            raise FileNotFoundError(f"Missing annotation folder: {label_dir}")
        self.label_files = sorted(label_dir.glob("*.json"))
        if not self.label_files:
            raise FileNotFoundError(f"No annotation JSONs found in {label_dir}")

    def __len__(self) -> int:
        return len(self.label_files)

    def get_sample(self, index: int) -> CanonicalV2XSample:
        label_path = self.label_files[index]
        with label_path.open("r", encoding="utf-8") as stream:
            openlabel = json.load(stream)["openlabel"]
        frames = openlabel["frames"]
        if len(frames) != 1:
            raise ValueError(f"Expected one frame per JSON: {label_path}")
        frame_key, frame = next(iter(frames.items()))
        properties = frame["frame_properties"]
        timestamp = float(properties["timestamp"])

        cloud_names = properties["point_cloud_file_names"]
        # JSON contains the exact synchronized filenames, which need not
        # share equal timestamps with each other or with the JSON filename.
        names_by_sensor = {}
        for name in cloud_names:
            for sensor in (VEHICLE_LIDAR, INFRA_LIDAR, FUSED_LIDAR):
                if name.endswith(f"_{sensor}.pcd"):
                    names_by_sensor[sensor] = name
                    break
        for sensor in (INFRA_LIDAR, VEHICLE_LIDAR):
            if sensor not in names_by_sensor:
                raise KeyError(f"Missing {sensor} cloud in {label_path}")

        def cloud_path(sensor: str) -> Path:
            path = self.split_root / "point_clouds" / sensor / names_by_sensor[sensor]
            if not path.is_file():
                raise FileNotFoundError(path)
            return path

        transform_info = properties["transforms"][TRANSFORM_KEY]
        if (transform_info.get("src") != VEHICLE_LIDAR
                or transform_info.get("dst") != INFRA_LIDAR):
            raise ValueError(f"Unexpected transform direction in {label_path}")
        T_ref_from_vehicle = np.asarray(
            transform_info["transform_src_to_dst"]["matrix4x4"], dtype=np.float64
        )
        if T_ref_from_vehicle.shape != (4, 4):
            raise ValueError(f"Invalid vehicle transform shape: {label_path}")

        infrastructure = AgentData(
            agent_id=INFRA_LIDAR,
            agent_type="infrastructure",
            is_reference=True,
            T_ref_from_agent=np.eye(4, dtype=np.float64),
            lidar=read_ascii_pcd(cloud_path(INFRA_LIDAR)),
        )
        vehicle = AgentData(
            agent_id=VEHICLE_LIDAR,
            agent_type="vehicle",
            is_reference=False,
            T_ref_from_agent=T_ref_from_vehicle,
            lidar=read_ascii_pcd(cloud_path(VEHICLE_LIDAR)),
        )

        image_paths = {}
        for image_name in properties.get("image_file_names", []):
            # Each filename ends in a sensor name known from the dataset tree.
            matches = list((self.split_root / "images").glob(f"*/{image_name}"))
            image_paths[image_name] = str(matches[0]) if matches else None

        fused_path = (str(cloud_path(FUSED_LIDAR))
                      if FUSED_LIDAR in names_by_sensor else None)

        return CanonicalV2XSample(
            source=SourceInfo(dataset="TUMTraf-V2X", sequence_id=self.split,
                              frame_id=label_path.stem),
            timestamp_s=timestamp,
            reference_agent_id=INFRA_LIDAR,
            agents=[infrastructure, vehicle],
            annotations=parse_annotations(frame.get("objects", {})),
            extras={
                "annotation_path": str(label_path),
                "openlabel_frame_key": frame_key,
                "registered_point_cloud_path": fused_path,
                "camera_image_paths_uncalibrated": image_paths,
                "note": "CameraData omitted: camera calibration not supplied/verified.",
            },
        )
