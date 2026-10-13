
"""
V2XPnP -> canonical V2X adapter.

Expected dataset layout:
    datasets/V2XPnP/
        scenario_name/
            1/
                000001.yaml
                000001.bin
                000001_camera0.png
                ...
            -1/
                000001.yaml
                000001.bin
            2/
            -2/

Also supports:
    datasets/V2XPnP/train/scenario_name/agent_id/...
    datasets/V2XPnP/validate/scenario_name/agent_id/...

Verified against the inspected sample format:
    - LiDAR: Nx4 float32 [x, y, z, intensity]
    - Pose: [x, y, z, roll, yaw, pitch] in degrees
    - Agent type: YAML 'infra' boolean
    - Object annotations: YAML 'vehicles' mapping
    - Object extent: half [length, width, height]
    - Camera calibration: cameraN/intrinsic/extrinsic

Important:
    Native-to-canonical LiDAR conversion uses F=diag(1,-1,1).
    All pose transforms use the exact OpenCOOD x_to_world formula.

    Camera calibration is preserved in its native convention
    pending separate camera-projection validation.
"""

from __future__ import annotations

import re
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import yaml

from adapters.base_adapter import BaseAdapter
from project_paths import resolve_path

from schema import (
    CanonicalV2XSample,
    SourceInfo,
    AgentData,
    LidarData,
    CameraData,
    AnnotationData,
)


# ============================================================
# Coordinate conventions
# ============================================================

# Native CARLA/OpenCOOD-style frame -> canonical frame.
#
# Native:    x forward, y right, z up
# Canonical: x forward, y left,  z up
#
# This is a reflection (determinant -1), NOT a rigid rotation.
# Relative rigid transforms must be converted by conjugation.

F3 = np.diag([1.0, -1.0, 1.0])

F4 = np.eye(4, dtype=np.float64)
F4[:3, :3] = F3


def x_to_world(pose: np.ndarray) -> np.ndarray:
    """
    OpenCOOD x_to_world, using its exact roll/yaw/pitch ordering.

    pose = [x, y, z, roll, yaw, pitch]
    Angles are in DEGREES.
    """

    pose = np.asarray(pose, dtype=np.float64)

    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("Invalid OpenCOOD pose; expected six finite values")

    x, y, z, roll, yaw, pitch = pose
    roll, yaw, pitch = np.radians([roll, yaw, pitch])

    cr, sr = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)

    T = np.eye(4, dtype=np.float64)

    T[:3, :3] = [
        [cp * cy, cy * sp * sr - sy * cr,
         -cy * sp * cr - sy * sr],

        [sy * cp, sy * sp * sr + cy * cr,
         -sy * sp * cr + cy * sr],

        [sp, -cp * sr, cp * cr],
    ]

    T[:3, 3] = [x, y, z]

    return T


def native_relative_transform(
    source_pose: np.ndarray,
    destination_pose: np.ndarray,
) -> np.ndarray:
    """
    Transform source coordinates into destination coordinates,
    both expressed in native OpenCOOD frames.
    """
    return (
        np.linalg.inv(x_to_world(destination_pose))
        @ x_to_world(source_pose)
    )


def canonical_relative_transform(
    source_pose: np.ndarray,
    destination_pose: np.ndarray,
) -> np.ndarray:
    """
    Same relative transformation expressed using the
    canonical coordinate axes at both endpoints.
    """
    T_native = native_relative_transform(
        source_pose, destination_pose
    )

    return F4 @ T_native @ F4


def transform_points(
    xyz: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """Transform Nx3 row-vector points using a 4x4 matrix."""
    return xyz @ T[:3, :3].T + T[:3, 3]


def wrap_yaw(yaw: float) -> float:
    return float(np.arctan2(np.sin(yaw), np.cos(yaw)))


def check_rigid_transform(T: np.ndarray) -> None:
    """Reject malformed homogeneous rigid transformations."""

    T = np.asarray(T, dtype=np.float64)

    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError("Transform is not a finite 4x4 matrix")

    np.testing.assert_allclose(
        T[3],
        [0.0, 0.0, 0.0, 1.0],
        atol=1e-7,
    )

    R = T[:3, :3]

    np.testing.assert_allclose(
        R.T @ R,
        np.eye(3),
        atol=1e-6,
    )

    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-6):
        raise ValueError("Rotation determinant differs from +1")


# ============================================================
# Schema construction
# ============================================================

def make_schema_object(cls, values: dict[str, Any]):
    """
    Construct an existing schema dataclass using supported fields.

    This avoids passing optional arguments that do not exist in a
    particular version of schema.py.

    Missing required fields are reported explicitly, rather than
    silently inserting invented values.
    """

    if not is_dataclass(cls):
        raise TypeError(
            f"{cls.__name__} must be a dataclass for this adapter"
        )

    definitions = {
        field.name: field
        for field in fields(cls)
        if field.init
    }

    kwargs = {
        name: value
        for name, value in values.items()
        if name in definitions
    }

    missing = [
        name
        for name, definition in definitions.items()
        if name not in kwargs
        and definition.default is MISSING
        and definition.default_factory is MISSING
    ]

    if missing:
        raise TypeError(
            f"Cannot construct {cls.__name__}. "
            f"Required schema fields not mapped: {missing}. "
            "Check the field names in schema.py."
        )

    return cls(**kwargs)


# ============================================================
# YAML and LiDAR loaders
# ============================================================

def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML document: {path}")

    return data


def read_pose(metadata: dict, path: Path) -> np.ndarray:
    if "lidar_pose" not in metadata:
        raise ValueError(f"Missing lidar_pose in {path}")

    pose = np.asarray(metadata["lidar_pose"], dtype=np.float64)

    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError(f"Invalid lidar_pose in {path}")

    return pose


def load_lidar_bin(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read the observed V2XPnP binary layout:
        float32 [x, y, z, intensity].
    """

    if not path.is_file():
        raise FileNotFoundError(path)

    raw = np.fromfile(path, dtype="<f4")

    if raw.size == 0 or raw.size % 4 != 0:
        raise ValueError(f"Invalid Nx4 LiDAR binary: {path}")

    points = raw.reshape(-1, 4)

    if not np.isfinite(points).all():
        raise ValueError(f"Non-finite LiDAR values: {path}")

    xyz_native = points[:, :3].astype(np.float32, copy=True)
    intensity = points[:, 3].astype(np.float32, copy=True)

    return xyz_native, intensity


# ============================================================
# Camera handling
# ============================================================

CAMERA_KEY_PATTERN = re.compile(r"^camera\d+$")


def find_camera_image(
    agent_dir: Path,
    frame_id: str,
    camera_id: str,
) -> Path | None:

    for extension in (".png", ".jpg", ".jpeg"):
        candidate = agent_dir / (
            f"{frame_id}_{camera_id}{extension}"
        )

        if candidate.is_file():
            return candidate

    return None


def load_rgb_image(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Camera decoding requires Pillow: pip install Pillow"
        ) from exc

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy()


def build_cameras(
    agent_dir: Path,
    frame_id: str,
    metadata: dict,
    load_images: bool,
    strict_camera_files: bool,
    *,
    native_cameras: list[dict] | None = None,
) -> list[CameraData]:
    """
    Discover real camera files and preserve the native calibration.

    Native camera-optical axes have not been verified. Omit canonical
    CameraData rather than invent its required T_agent_from_camera.
    Append native information to the optional agent-extras collector;
    load_images controls decoding, and strict_camera_files rejects missing files.
    """

    cameras = []

    for camera_id in sorted(metadata):
        if not CAMERA_KEY_PATTERN.fullmatch(str(camera_id)):
            continue

        calibration = metadata[camera_id]

        if not isinstance(calibration, dict):
            raise ValueError(
                f"Invalid calibration for {agent_dir.name}/{camera_id}"
            )

        image_path = find_camera_image(
            agent_dir, frame_id, camera_id
        )

        if image_path is None:
            message = (
                f"Camera metadata exists but image is missing: "
                f"{agent_dir}/{frame_id}_{camera_id}"
            )

            if strict_camera_files:
                raise FileNotFoundError(message)

            warnings.warn(message)

        K = None
        if "intrinsic" in calibration:
            K = np.asarray(
                calibration["intrinsic"], dtype=np.float64
            )

            if K.shape != (3, 3) or not np.isfinite(K).all():
                raise ValueError(
                    f"Invalid camera intrinsic: {image_path}"
                )

        E_native = None
        if "extrinsic" in calibration:
            E_native = np.asarray(
                calibration["extrinsic"], dtype=np.float64
            )

            if (
                E_native.shape != (4, 4)
                or not np.isfinite(E_native).all()
            ):
                raise ValueError(
                    f"Invalid native camera extrinsic: {image_path}"
                )

        rgb = (
            load_rgb_image(image_path)
            if load_images and image_path is not None else None
        )

        # OpenCOOD describes 'extrinsic' as camera-to-LiDAR.
        # Preserve the native matrix here; camera coordinate
        # basis conversion needs an independent projection test.

        extra = {
            "camera_id": str(camera_id),
            "dataset": "V2XPnP",
            "image_path": str(image_path) if image_path is not None else None,
            "image": rgb,
            "native_extrinsic_camera_to_lidar": E_native,
            "native_intrinsic": K,
            "native_camera_pose": calibration.get("cords"),
            "native_metadata": dict(calibration),
            "calibration_status": "native_not_canonical",
            "omission_reason": "camera_optical_to_canonical_agent_axes_unverified",
        }

        if native_cameras is not None:
            native_cameras.append(extra)

    return cameras


# ============================================================
# OpenCOOD object annotations
# ============================================================

def object_to_canonical_box(
    obj: dict,
    T_ref_from_world_native: np.ndarray,
) -> np.ndarray:
    """
    Convert a world-annotated OpenCOOD object to:

        [x, y, z, length, width, height, yaw]

    in the canonical reference LiDAR coordinates.

    The native object 'center' is an offset in the object's
    local frame, not a second world-coordinate position.
    """

    for key in ("location", "angle", "extent"):
        if key not in obj:
            raise ValueError(
                f"Object annotation missing {key!r}: {obj}"
            )

    location = np.asarray(
        obj["location"], dtype=np.float64
    )
    angles = np.asarray(
        obj["angle"], dtype=np.float64
    )
    extent = np.asarray(
        obj["extent"], dtype=np.float64
    )
    center = np.asarray(
        obj.get("center", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    )

    for name, array in (
        ("location", location),
        ("angle", angles),
        ("extent", extent),
        ("center", center),
    ):
        if array.shape != (3,) or not np.isfinite(array).all():
            raise ValueError(
                f"Invalid object {name}: {array}"
            )

    if np.any(extent <= 0.0):
        raise ValueError(
            f"Nonpositive object extent: {extent}"
        )

    # Object transform in native world coordinates.
    object_pose = np.concatenate([location, angles])
    T_world_from_object = x_to_world(object_pose)

    # OpenCOOD: box center offset is in object-local axes.
    center_world_native = transform_points(
        center.reshape(1, 3),
        T_world_from_object,
    )[0]

    # World -> reference LiDAR, still native axes.
    center_ref_native = transform_points(
        center_world_native.reshape(1, 3),
        T_ref_from_world_native,
    )[0]

    # Convert the reference frame into canonical axes.
    center_ref = F3 @ center_ref_native

    # Native heading is the object's positive local X axis.
    heading_world_native = T_world_from_object[:3, 0]

    heading_ref_native = (
        T_ref_from_world_native[:3, :3]
        @ heading_world_native
    )

    heading_ref = F3 @ heading_ref_native

    horizontal_norm = np.linalg.norm(heading_ref[:2])

    if horizontal_norm < 1e-8:
        raise ValueError("Object heading has no horizontal component")

    yaw = wrap_yaw(
        np.arctan2(heading_ref[1], heading_ref[0])
    )

    # OpenCOOD extent is half [length, width, height].
    dimensions = 2.0 * extent

    return np.array(
        [
            center_ref[0],
            center_ref[1],
            center_ref[2],
            dimensions[0],
            dimensions[1],
            dimensions[2],
            yaw,
        ],
        dtype=np.float32,
    )


def collect_annotations(
    agent_records: list[dict],
    reference_pose: np.ndarray,
) -> AnnotationData:
    """
    Merge per-agent observations of objects using object IDs.

    The reference agent has priority. Other agents contribute
    previously unseen IDs, avoiding duplicate supervision.
    """

    T_ref_from_world_native = np.linalg.inv(
        x_to_world(reference_pose)
    )

    objects_by_id = {}

    for record in agent_records:
        annotations = record["metadata"].get("vehicles", {})

        if annotations is None:
            continue

        if not isinstance(annotations, dict):
            raise ValueError(
                "'vehicles' must be a mapping of object IDs"
            )

        for object_id, obj in annotations.items():
            key = str(object_id)

            if key in objects_by_id:
                continue

            if not isinstance(obj, dict):
                raise ValueError(
                    f"Invalid object annotation for ID {key}"
                )

            box = object_to_canonical_box(
                obj, T_ref_from_world_native
            )

            objects_by_id[key] = {
                "box": box,
                "label": str(
                    obj.get("obj_type", "unknown")
                ),
                "track_id": key,
                "native_object_id": object_id,
                "attribute": obj.get("attribute"),
                "source_agent": record["agent_id"],
            }

    boxes = [
        item["box"]
        for item in objects_by_id.values()
    ]

    boxes_3d = (
        np.stack(boxes).astype(np.float32)
        if boxes
        else np.empty((0, 7), dtype=np.float32)
    )

    native_labels = [
        item["label"]
        for item in objects_by_id.values()
    ]

    # Keep usable native integer IDs. UUIDs, malformed IDs and values outside
    # int64 cannot be canonical integer tracking IDs; the schema uses -1.
    # No per-frame remapping: that would break tracking across frames.
    integer_ids = []
    for item in objects_by_id.values():
        try:
            value = int(item["track_id"])
        except ValueError:
            value = -1
        integer_ids.append(value if 0 <= value <= np.iinfo(np.int64).max else -1)
    track_ids = np.asarray(integer_ids, dtype=np.int64)

    annotation_metadata = {
        "native_object_ids": [item["native_object_id"] for item in objects_by_id.values()],
        "annotation_coordinate_frame": "canonical_reference_lidar",
        "box_convention": "center_xyz_lwh_yaw",
        "box_center_assumption": "location_plus_rotated_center",
        "box_extent_assumption": "half_length_width_height",
        "object_attributes": [
            {
                "track_id": item["track_id"],
                "attribute": item["attribute"],
                "source_agent": item["source_agent"],
            }
            for item in objects_by_id.values()
        ],
    }

    return make_schema_object(
        AnnotationData,
        {
            "boxes_3d": boxes_3d,
            "native_labels": native_labels,
            "labels": None,
            "track_ids": track_ids,
            "extras": annotation_metadata,
        },
    )


# ============================================================
# Scenario discovery
# ============================================================

def is_agent_directory(path: Path) -> bool:
    return (
        path.is_dir()
        and bool(re.fullmatch(r"-?\d+", path.name))
        and any(path.glob("*.yaml"))
    )


def find_scenarios(dataset_root: Path) -> list[Path]:
    """
    Find scenarios beneath a dataset root.

    Works with both direct scenario folders and split folders.
    """

    scenarios = ([dataset_root] if any(is_agent_directory(child)
                 for child in dataset_root.iterdir()) else [])

    for directory in dataset_root.rglob("*"):
        if not directory.is_dir():
            continue

        if any(
            is_agent_directory(child)
            for child in directory.iterdir()
            if child.is_dir()
        ):
            scenarios.append(directory)

    return sorted(set(scenarios))


def sort_agent_ids(agent_id: str):
    return int(agent_id)


# ============================================================
# Adapter
# ============================================================

class V2XPnPAdapter(BaseAdapter):
    """
    Dataset-specific adapter returning CanonicalV2XSample.

    The selected reference agent is a vehicle by default.
    All available agents are represented at each indexed frame.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: str | None = None,
        reference_agent_id: str | None = "1",
        load_images: bool = False,
        strict_camera_files: bool = False,
        frame_rate_hz: float = 10.0,
    ):

        dataset_root = resolve_path(dataset_root)

        if not dataset_root.is_dir():
            raise FileNotFoundError(dataset_root)

        if frame_rate_hz <= 0:
            raise ValueError("frame_rate_hz must be positive")

        super().__init__(
            dataset_root=dataset_root,
            split=split,
        )

        self.reference_agent_id = (
            str(reference_agent_id)
            if reference_agent_id is not None
            else None
        )

        self.load_images = load_images
        self.strict_camera_files = strict_camera_files
        self.frame_rate_hz = float(frame_rate_hz)

        search_root = dataset_root

        if split is not None:
            candidates = [dataset_root / split]

            if split == "val":
                candidates.append(dataset_root / "validate")

            if split == "validate":
                candidates.append(dataset_root / "val")

            matching = [
                path for path in candidates
                if path.is_dir()
            ]

            if not matching:
                raise FileNotFoundError(
                    f"Split {split!r} not found under {dataset_root}"
                )

            search_root = matching[0]

        self.scenarios = find_scenarios(search_root)

        if not self.scenarios:
            raise ValueError(
                f"No V2XPnP scenarios found under {search_root}"
            )

        self._index = []

        for scenario_dir in self.scenarios:
            agent_dirs = {
                child.name: child
                for child in scenario_dir.iterdir()
                if is_agent_directory(child)
            }

            if not agent_dirs:
                continue

            if self.reference_agent_id is not None:
                if self.reference_agent_id not in agent_dirs:
                    warnings.warn(
                        f"Skipping {scenario_dir.name}: "
                        f"reference agent {self.reference_agent_id} absent"
                    )
                    continue

                ref_id = self.reference_agent_id

            else:
                # Prefer the lowest positive agent ID.
                positive_agents = sorted(
                    (
                        agent_id
                        for agent_id in agent_dirs
                        if int(agent_id) > 0
                    ),
                    key=sort_agent_ids,
                )

                all_agents = sorted(
                    agent_dirs,
                    key=sort_agent_ids,
                )

                ref_id = (
                    positive_agents[0]
                    if positive_agents
                    else all_agents[0]
                )

            # Index only complete reference-agent frames.
            ref_dir = agent_dirs[ref_id]

            for yaml_path in sorted(ref_dir.glob("*.yaml")):
                frame_id = yaml_path.stem

                if not frame_id.isdigit():
                    continue

                bin_path = ref_dir / f"{frame_id}.bin"

                if not bin_path.is_file():
                    warnings.warn(
                        f"Skipping incomplete frame: {yaml_path}"
                    )
                    continue

                self._index.append(
                    (scenario_dir, frame_id, ref_id)
                )

        if not self._index:
            raise ValueError(
                "No complete V2XPnP reference-agent frames found"
            )

    def __len__(self) -> int:
        return len(self._index)

    def get_sample(self, index: int) -> CanonicalV2XSample:
        """
        Load one canonical multi-agent sample.
        """

        scenario_dir, frame_id, ref_id = self._index[index]

        available_agents = sorted(
            (
                child
                for child in scenario_dir.iterdir()
                if is_agent_directory(child)
            ),
            key=lambda path: sort_agent_ids(path.name),
        )

        records = []

        for agent_dir in available_agents:
            yaml_path = agent_dir / f"{frame_id}.yaml"
            bin_path = agent_dir / f"{frame_id}.bin"

            # Additional agents may be missing particular frames.
            if not yaml_path.is_file() or not bin_path.is_file():
                continue

            metadata = read_yaml(yaml_path)
            pose = read_pose(metadata, yaml_path)

            if not isinstance(metadata.get("infra"), bool):
                raise ValueError(
                    f"Missing/invalid 'infra' flag: {yaml_path}"
                )

            records.append(
                {
                    "agent_id": agent_dir.name,
                    "agent_dir": agent_dir,
                    "metadata": metadata,
                    "pose": pose,
                    "lidar_path": bin_path,
                    "yaml_path": yaml_path,
                }
            )

        ref_record = next(
            (
                record
                for record in records
                if record["agent_id"] == ref_id
            ),
            None,
        )

        if ref_record is None:
            raise RuntimeError(
                f"Reference frame missing: {scenario_dir}/{ref_id}/{frame_id}"
            )

        # Process reference first so its object annotations win
        # when the same track ID occurs in multiple YAML files.
        records.sort(
            key=lambda record: (
                record["agent_id"] != ref_id,
                sort_agent_ids(record["agent_id"]),
            )
        )

        reference_pose = ref_record["pose"]

        agents = []

        for record in records:
            agent_id = record["agent_id"]
            metadata = record["metadata"]

            xyz_native, intensity = load_lidar_bin(
                record["lidar_path"]
            )

            # Native sensor xyz -> canonical sensor xyz.
            xyz_canonical = (
                xyz_native.astype(np.float64) @ F3.T
            ).astype(np.float32)

            lidar = make_schema_object(
                LidarData,
                {
                    "xyz": xyz_canonical,
                    "intensity": intensity,
                    "metadata": {
                        "source_path": str(record["lidar_path"]),
                        "native_format": "float32_Nx4",
                        "intensity_normalized": False,
                        "coordinate_conversion": "mirror_y",
                    },
                },
            )

            T_ref_from_agent = canonical_relative_transform(
                record["pose"],
                reference_pose,
            )

            check_rigid_transform(T_ref_from_agent)

            if agent_id == ref_id:
                np.testing.assert_allclose(
                    T_ref_from_agent,
                    np.eye(4),
                    atol=1e-7,
                )

            native_cameras = []
            cameras = build_cameras(
                agent_dir=record["agent_dir"],
                frame_id=frame_id,
                metadata=metadata,
                load_images=self.load_images,
                strict_camera_files=self.strict_camera_files,
                native_cameras=native_cameras,
            )

            agent_type = (
                "infrastructure"
                if metadata["infra"]
                else "vehicle"
            )

            agent = make_schema_object(
                AgentData,
                {
                    "agent_id": str(agent_id),
                    "agent_type": agent_type,
                    "is_reference": agent_id == ref_id,
                    "T_ref_from_agent": T_ref_from_agent,
                    "lidar": lidar,
                    "cameras": cameras,
                    "extras": {"native_cameras": native_cameras},
                    "metadata": {
                        "native_pose": record["pose"].tolist(),
                        "yaml_path": str(record["yaml_path"]),
                        "native_agent_id": agent_id,
                        "infra": metadata["infra"],
                        "ego_speed": metadata.get("ego_speed"),
                        "plan_trajectory_native": metadata.get(
                            "plan_trajectory"
                        ),
                        "predicted_ego_pos_native": metadata.get(
                            "predicted_ego_pos"
                        ),
                        "true_ego_pos_native": metadata.get(
                            "true_ego_pos"
                        ),
                    },
                },
            )

            agents.append(agent)

        annotations = collect_annotations(
            records,
            reference_pose,
        )

        # Nominal frame-index time, NOT an absolute measurement
        # timestamp. The original frame ID is retained below.
        timestamp_s = int(frame_id) / self.frame_rate_hz

        source = make_schema_object(
            SourceInfo,
            {
                "dataset": "V2XPnP",
                "dataset_name": "V2XPnP",
                "sequence_id": scenario_dir.name,
                "scenario_id": scenario_dir.name,
                "frame_id": frame_id,
                "split": self.split,
                "original_path": str(scenario_dir),
                "source_path": str(scenario_dir),
                "metadata": {
                    "timestamp_kind": "nominal_frame_offset",
                    "nominal_frame_rate_hz": self.frame_rate_hz,
                    "native_convention": "OpenCOOD/CARLA",
                    "canonical_axes": "x_forward_y_left_z_up",
                    "camera_calibration_status": (
                        "native_preserved; projection_not_validated"
                    ),
                },
            },
        )

        sample = make_schema_object(
            CanonicalV2XSample,
            {
                "source": source,
                "timestamp_s": timestamp_s,
                "timestamp": timestamp_s,
                "reference_agent_id": str(ref_id),
                "agents": agents,
                "annotations": annotations,
            },
        )

        # The base adapter's __getitem__ also validates.
        # Calling get_sample directly should be safe as well.
        sample.validate()

        return sample
