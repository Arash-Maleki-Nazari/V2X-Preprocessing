"""DeepAccident converted-info -> canonical V2X adapter (provisional).

This adapter consumes the official ``*_infos_<split>.pkl`` dictionaries
produced by ``tools/data_converter/carla_converter.py``. It does NOT require
MMCV, MMDetection3D, or a copy of the full dataset to index the info file.

One canonical sample represents a (scene_name, native frame index) pair,
with up to five synchronized agents. ``ego_vehicle`` is the reference.

Important validation limits
---------------------------
* The official converter writes frame indices as ``timestamp``; these are NOT
  known to be seconds. Without an explicitly supplied ``frame_rate_hz``,
  ``timestamp_s`` is a placeholder 0.0 and the native index is kept in extras.
* The original calibration/world handedness has not been independently
  projection-validated. Relative agent poses are provisionally converted
  assuming CARLA-style x-forward/y-right/z-up native axes. This assumption
  is marked in the returned metadata. Override ``native_axes`` if verified.
* Camera image paths, decoded RGB arrays (optional), intrinsics and native
  extrinsics are kept under ``AgentData.extras['native_cameras']``. No
  ``CameraData`` is emitted until camera optical-axis conversion is verified.
* The ``gt_boxes`` in official infos are LiDAR-frame annotations, not
  reference-frame boxes. They are preserved in
  ``AnnotationData.extras['native_annotations_by_agent']``. Canonical boxes
  remain empty rather than mislabeling native geometry as reference geometry.
* Native LiDAR .npz decoding is unverified. Paths are preserved, but
  ``AgentData.lidar`` is None. 

Pickle loading is unsafe for untrusted inputs. Only load info files obtained
from a trusted source or generated locally.
"""

from __future__ import annotations

import pickle
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np

from adapters.base_adapter import BaseAdapter
from project_paths import resolve_path
from schema import AgentData, AnnotationData, CanonicalV2XSample, SourceInfo


AGENT_ORDER = (
    "ego_vehicle",
    "ego_vehicle_behind",
    "other_vehicle",
    "other_vehicle_behind",
    "infrastructure",
)

# The reflection changes coordinate handedness; it is not a rotation.
_Y_REFLECTION = np.diag([1.0, -1.0, 1.0, 1.0])


def _matrix(value: Any, name: str, shape: tuple[int, int]) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite {shape} matrix, got {matrix.shape}")
    return matrix.copy()


def _rigid_matrix(value: Any, name: str) -> np.ndarray:
    matrix = _matrix(value, name, (4, 4))
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotation = matrix[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)):
        raise ValueError(f"{name} must contain a proper rigid rotation")
    return matrix


def _resolve_native_path(dataset_root: Path, value: str | Path | None,
                         scenario_type: str | None = None) -> Path | None:
    """Rebase converter paths when infos were made on another machine.

    The official converter stores absolute paths. If one no longer exists,
    anchor the suffix starting at the known scenario-type directory under
    the current dataset root. This does not search for or fabricate files.
    """
    if not value:
        return None
    path = Path(value)
    if path.is_absolute() and path.exists():
        return path
    if not path.is_absolute():
        candidate = dataset_root / path
        if candidate.exists():
            return candidate
    if scenario_type:
        parts = path.parts
        if scenario_type in parts:
            offset = parts.index(scenario_type)
            return dataset_root.joinpath(*parts[offset:])
    return path if path.is_absolute() else dataset_root / path


def _load_rgb(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Camera decoding needs Pillow: pip install Pillow") from exc
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB")).copy()


def _native_annotations(info: dict[str, Any]) -> dict[str, Any]:
    """Preserve native per-agent labels; never claim they are canonical."""
    has_boxes = "gt_boxes" in info
    boxes = np.asarray(info.get("gt_boxes", np.empty((0, 7))), dtype=np.float64)
    if boxes.size == 0:
        boxes = np.empty((0, 7), dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 7 or not np.isfinite(boxes).all():
        raise ValueError("Native gt_boxes must be finite [M, 7]")
    size = len(boxes)
    labels = [str(label) for label in info.get("gt_names", [])]
    if has_boxes and len(labels) != size:
        raise ValueError("Native gt_names count does not match gt_boxes")
    if not has_boxes and labels:
        raise ValueError("Native gt_names present without gt_boxes")

    native: dict[str, Any] = {"gt_boxes": boxes.copy(), "gt_names": labels}
    for name, width in (("gt_velocity", 2),):
        if name in info:
            values = np.asarray(info[name], dtype=np.float64)
            if values.shape != (size, width):
                raise ValueError(f"Native {name} must have shape ({size}, {width})")
            native[name] = values.copy()  # NaNs may exist in native velocities.
    for name in ("vehicle_id", "num_lidar_pts", "camera_visibility"):
        if name in info:
            values = np.asarray(info[name])
            if values.shape != (size,):
                raise ValueError(f"Native {name} must have shape ({size},)")
            native[name] = values.copy()
    return native


class DeepAccidentAdapter(BaseAdapter):
    """Index official converted DeepAccident infos as multi-agent frames.

    Parameters
    ----------
    dataset_root:
        Root of native DeepAccident files. They may be absent when only
        inspecting metadata from a local/synthetic converted-info pickle.
    split:
        One of train, val, test, train_mini, val_mini. If unspecified,
        ``info_path`` must identify a particular file (unless only one
        matching info pickle exists).
    info_path:
        Path to an official converted-info pickle. Auto-discovered beneath
        ``dataset_root`` when omitted.
    reference_agent_id:
        Defaults to ``ego_vehicle``. Frames without that agent are skipped.
    frame_rate_hz:
        Optional independently verified native frame rate. If omitted,
        seconds cannot be derived from the raw frame index.
    native_axes:
        ``carla`` provisionally assumes x-forward/y-right/z-up in native
        ego/world frames and conjugates relative transforms with a Y
        reflection. ``canonical`` assumes they already use x-forward/y-left.
        Neither setting establishes camera optical-axis conventions.
    load_images:
        Decode RGB images into native_cameras metadata; False by default.
        CameraData remains empty regardless, pending calibration validation.
    strict_camera_files:
        Raise for missing images when inspecting cameras; otherwise warn
        only if load_images=True.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: str | None = None,
        *,
        info_path: str | Path | None = None,
        reference_agent_id: str = "ego_vehicle",
        frame_rate_hz: float | None = None,
        native_axes: Literal["carla", "canonical"] = "carla",
        load_images: bool = False,
        strict_camera_files: bool = False,
    ) -> None:
        super().__init__(dataset_root, split)
        if not reference_agent_id:
            raise ValueError("reference_agent_id must be nonempty")
        if frame_rate_hz is not None and (not np.isfinite(frame_rate_hz) or frame_rate_hz <= 0):
            raise ValueError("frame_rate_hz must be a positive finite number")
        if native_axes not in ("carla", "canonical"):
            raise ValueError("native_axes must be 'carla' or 'canonical'")

        self.reference_agent_id = reference_agent_id
        self.frame_rate_hz = float(frame_rate_hz) if frame_rate_hz is not None else None
        self.native_axes = native_axes
        self.load_images = bool(load_images)
        self.strict_camera_files = bool(strict_camera_files)
        self.info_path = self._find_info_file(info_path)

        # Only load trusted .pkl files. Official MMCV .pkl uses plain pickle.
        with self.info_path.open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict) or not isinstance(payload.get("infos"), list):
            raise ValueError("Expected official pickle mapping containing an 'infos' list")
        self.info_metadata = payload.get("metadata", {})

        grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
        for info in payload["infos"]:
            if not isinstance(info, dict):
                raise ValueError("Each DeepAccident info record must be a mapping")
            try:
                scene = str(info["scene_name"])
                frame = int(info["timestamp"])
                agent_name = str(info["vehicle_name"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Info missing scene_name, timestamp, or vehicle_name") from exc
            if not scene or not agent_name or frame < 0:
                raise ValueError("Invalid scene_name, vehicle_name, or timestamp")
            if agent_name in grouped[(scene, frame)]:
                raise ValueError(f"Duplicate DeepAccident agent {agent_name} at {scene}/{frame}")
            grouped[(scene, frame)][agent_name] = info

        self._index = sorted(
            (key, records) for key, records in grouped.items()
            if self.reference_agent_id in records
        )
        skipped = len(grouped) - len(self._index)
        if skipped:
            warnings.warn(f"Skipped {skipped} frames without reference agent "
                          f"{self.reference_agent_id!r}", stacklevel=2)
        if not self._index:
            raise ValueError(f"No synchronized frames with reference agent "
                             f"{self.reference_agent_id!r} in {self.info_path}")

    def _find_info_file(self, info_path: str | Path | None) -> Path:
        if info_path is not None:
            path = resolve_path(info_path)
            if not path.is_file():
                raise FileNotFoundError(path)
            return path
        if self.split is not None:
            allowed = {"train", "val", "test", "train_mini", "val_mini"}
            if self.split not in allowed:
                raise ValueError(f"Unsupported split {self.split!r}; choose from {sorted(allowed)}")
            matches = sorted(self.dataset_root.rglob(f"*_infos_{self.split}.pkl"))
        else:
            matches = sorted(self.dataset_root.rglob("*_infos_*.pkl"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one DeepAccident info pickle for split "
                f"{self.split!r} under {self.dataset_root}, found {len(matches)}. "
                "Pass info_path=... explicitly."
            )
        return matches[0]

    def __len__(self) -> int:
        return len(self._index)

    def _native_cameras(self, info: dict[str, Any]) -> list[dict[str, Any]]:
        native_cameras: list[dict[str, Any]] = []
        cams = info.get("cams", {})
        if not isinstance(cams, dict):
            raise ValueError("Native cams must be a dictionary")
        scenario_type = info.get("scenario_type")
        for name, camera in sorted(cams.items()):
            if not isinstance(camera, dict):
                raise ValueError(f"Invalid camera info {name!r}")
            image_path = _resolve_native_path(
                self.dataset_root, camera.get("image_path"), scenario_type
            )
            K = camera.get("camera_intrinsic_matrix")
            E = camera.get("lidar_to_camera_matrix")
            K = _matrix(K, f"{name}.intrinsics", (3, 3)) if K is not None else None
            E = _matrix(E, f"{name}.lidar_to_camera", (4, 4)) if E is not None else None

            if self.strict_camera_files and (image_path is None or not image_path.is_file()):
                raise FileNotFoundError(f"Missing DeepAccident image: {image_path}")
            image = None
            if self.load_images:
                if image_path is not None and image_path.is_file():
                    image = _load_rgb(image_path)
                else:
                    warnings.warn(f"Missing DeepAccident camera image: {image_path}",
                                  stacklevel=2)
            native_cameras.append({
                "camera_id": str(name),
                "image_path": str(image_path) if image_path is not None else None,
                "image": image,
                "native_intrinsic": K,
                "native_lidar_to_camera_matrix": E,
                "native_timestamp_frame": camera.get("timestamp"),
                "calibration_status": "native_not_canonical",
                "omission_reason": "camera_optical_to_canonical_agent_unverified",
            })
        return native_cameras

    def get_sample(self, index: int) -> CanonicalV2XSample:
        if not isinstance(index, int):
            raise TypeError("Sample index must be an integer")
        # Also makes direct get_sample() behave like inherited __getitem__().
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)

        (scene, frame), records = self._index[index]
        ref_info = records[self.reference_agent_id]
        T_world_from_ref = _rigid_matrix(
            ref_info["ego_to_world_matrix"], "reference ego_to_world_matrix"
        )
        T_ref_from_world = np.linalg.inv(T_world_from_ref)
        reflection = _Y_REFLECTION if self.native_axes == "carla" else np.eye(4)

        agents = []
        native_annotations: dict[str, dict[str, Any]] = {}
        ordered_agents = sorted(records, key=lambda agent: (
            AGENT_ORDER.index(agent) if agent in AGENT_ORDER else len(AGENT_ORDER),
            agent,
        ))
        for agent_id in ordered_agents:
            info = records[agent_id]
            T_world_from_agent = _rigid_matrix(
                info["ego_to_world_matrix"], f"{agent_id}.ego_to_world_matrix"
            )
            T_native_ref_from_agent = T_ref_from_world @ T_world_from_agent
            T_ref_from_agent = reflection @ T_native_ref_from_agent @ reflection
            if agent_id == self.reference_agent_id:
                T_ref_from_agent = np.eye(4, dtype=np.float64)
            else:
                T_ref_from_agent = _rigid_matrix(
                    T_ref_from_agent, f"{agent_id}.T_ref_from_agent"
                )

            lidar_to_ego = info.get("lidar_to_ego_matrix")
            if lidar_to_ego is not None:
                lidar_to_ego = _matrix(
                    lidar_to_ego, f"{agent_id}.lidar_to_ego_matrix", (4, 4)
                )
            scenario_type = info.get("scenario_type")
            native_annotations[agent_id] = _native_annotations(info)
            agents.append(AgentData(
                agent_id=agent_id,
                agent_type="infrastructure" if agent_id == "infrastructure" else "vehicle",
                is_reference=(agent_id == self.reference_agent_id),
                T_ref_from_agent=T_ref_from_agent,
                lidar=None,
                cameras=[],
                extras={
                    "native_lidar_path": str(_resolve_native_path(
                        self.dataset_root, info.get("lidar_path"), scenario_type
                    )) if info.get("lidar_path") else None,
                    "native_bev_path": str(_resolve_native_path(
                        self.dataset_root, info.get("bev_path"), scenario_type
                    )) if info.get("bev_path") else None,
                    "native_lidar_to_ego_matrix": lidar_to_ego,
                    "native_ego_to_world_matrix": T_world_from_agent,
                    "native_cameras": self._native_cameras(info),
                    "native_vehicle_speed_xy": (
                        info.get("vehicle_speed_x"), info.get("vehicle_speed_y")
                    ),
                    "native_lidar_prefix": info.get("lidar_prefix"),
                    "sensor_conversion_status": "unverified",
                    "relative_pose_conversion": f"provisional_{self.native_axes}",
                },
            ))

        # No misleading native-LiDAR-frame boxes in canonical reference frame.
        annotations = AnnotationData(
            boxes_3d=np.empty((0, 7), dtype=np.float32),
            native_labels=[],
            track_ids=np.empty((0,), dtype=np.int64),
            labels=None,
            velocity_xy=None,
            extras={
                "native_annotations_by_agent": native_annotations,
                "annotation_status": "native_preserved; reference_frame_conversion_unverified",
                "omitted_from_canonical": True,
            },
        )

        # A native frame index is not a time expressed in seconds.
        timestamp_s = (frame / self.frame_rate_hz) if self.frame_rate_hz else 0.0
        sample = CanonicalV2XSample(
            source=SourceInfo(dataset="DeepAccident", sequence_id=scene, frame_id=str(frame)),
            timestamp_s=float(timestamp_s),
            reference_agent_id=self.reference_agent_id,
            agents=agents,
            annotations=annotations,
            extras={
                "split": self.split,
                "info_path": str(self.info_path),
                "info_metadata": self.info_metadata,
                "native_frame_index": frame,
                "timestamp_status": ("nominal_frame_rate" if self.frame_rate_hz
                                     else "seconds_unknown_placeholder_zero"),
                "frame_rate_hz": self.frame_rate_hz,
                "scenario_type": ref_info.get("scenario_type"),
                "scenario_length_native": ref_info.get("scenario_length"),
                "native_axes_assumption": self.native_axes,
                "relative_pose_status": "provisional; not physically validated",
            },
        )
        sample.validate()
        return sample
