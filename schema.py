"""
Canonical V2X sample schema.

Every dataset adapter must translate its native dataset format into
these common data structures.

Canonical geometry conventions
------------------------------
Coordinate system:
    +x = forward
    +y = left
    +z = up

3D bounding box:
    [x, y, z, length, width, height, yaw]

    - (x, y, z) is the geometric center of the box.
    - yaw is in radians.
    - positive yaw is counter-clockwise around +z.

Transform naming:
    T_destination_from_source

For example:

    p_ref = T_ref_from_agent @ p_agent

All final 3D annotations are expressed in the reference-agent frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np


# ---------------------------------------------------------------------
# Source information
# ---------------------------------------------------------------------------


@dataclass
class SourceInfo:
    """
    Identifies where a canonical sample originally came from.
    """

    dataset: str
    sequence_id: str
    frame_id: str


# ---------------------------------------------------------------------
# LiDAR
# ---------------------------------------------------------------------------


@dataclass
class LidarData:
    """
    Canonical LiDAR representation.

    xyz:
        Shape [N, 3]
        Columns: [x, y, z]

    intensity:
        Optional shape [N].
        Intensity is preserved from the original dataset.
        Dataset-specific normalization should NOT happen here.

    time_offset_s:
        Optional per-point time offset relative to the sample timestamp.
        Shape [N].

        Useful, for example, for UrbanIng-V2X.

    sensor_index:
        Optional LiDAR sensor identifier for each point.
        Shape [N].

        Useful when points from multiple LiDAR sensors are combined.
    """

    xyz: np.ndarray

    intensity: Optional[np.ndarray] = None
    time_offset_s: Optional[np.ndarray] = None
    sensor_index: Optional[np.ndarray] = None

    extras: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Check that the LiDAR data follows the canonical contract."""

        if not isinstance(self.xyz, np.ndarray):
            raise TypeError("lidar.xyz must be a numpy array.")

        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3:
            raise ValueError(
                f"lidar.xyz must have shape [N, 3], got {self.xyz.shape}."
            )

        num_points = self.xyz.shape[0]

        if self.intensity is not None:
            if self.intensity.shape != (num_points,):
                raise ValueError(
                    "lidar.intensity must have shape [N], "
                    f"got {self.intensity.shape}."
                )

        if self.time_offset_s is not None:
            if self.time_offset_s.shape != (num_points,):
                raise ValueError(
                    "lidar.time_offset_s must have shape [N], "
                    f"got {self.time_offset_s.shape}."
                )

        if self.sensor_index is not None:
            if self.sensor_index.shape != (num_points,):
                raise ValueError(
                    "lidar.sensor_index must have shape [N], "
                    f"got {self.sensor_index.shape}."
                )


# ---------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------


@dataclass
class CameraData:
    """
    Canonical camera representation.

    image:
        Decoded image as a numpy array.

        RGB cameras normally use:
            [H, W, 3]

        Other modalities such as thermal cameras may have a different
        channel layout.

    K:
        Camera intrinsic matrix.
        Shape [3, 3].

    T_agent_from_camera:
        4x4 homogeneous transformation from the camera coordinate frame
        into the canonical local agent frame.

        p_agent = T_agent_from_camera @ p_camera

    time_offset_s:
        Camera timestamp offset relative to the canonical sample timestamp.
    """

    camera_id: str

    modality: Literal["rgb", "thermal"]

    image: np.ndarray

    K: np.ndarray

    T_agent_from_camera: np.ndarray

    time_offset_s: float = 0.0

    distortion: Optional[np.ndarray] = None

    extras: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Check that the camera data follows the canonical contract."""

        if not isinstance(self.image, np.ndarray):
            raise TypeError(
                f"Camera '{self.camera_id}' image must be a numpy array."
            )

        if self.image.ndim not in (2, 3):
            raise ValueError(
                f"Camera '{self.camera_id}' image must have "
                f"2 or 3 dimensions, got {self.image.shape}."
            )

        if self.K.shape != (3, 3):
            raise ValueError(
                f"Camera '{self.camera_id}' K must have shape [3, 3], "
                f"got {self.K.shape}."
            )

        if self.T_agent_from_camera.shape != (4, 4):
            raise ValueError(
                f"Camera '{self.camera_id}' T_agent_from_camera "
                f"must have shape [4, 4], "
                f"got {self.T_agent_from_camera.shape}."
            )


# ---------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------


@dataclass
class AgentData:
    """
    Canonical representation of one cooperative agent.

    An agent can be either:
        - a vehicle
        - an infrastructure unit

    T_ref_from_agent:
        Transformation from this agent's canonical local frame
        to the reference-agent frame.

        p_ref = T_ref_from_agent @ p_agent

    For the reference agent itself:

        T_ref_from_agent = identity matrix
    """

    agent_id: str

    agent_type: Literal["vehicle", "infrastructure"]

    is_reference: bool

    T_ref_from_agent: np.ndarray

    lidar: Optional[LidarData] = None

    cameras: list[CameraData] = field(default_factory=list)

    extras: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Check that the agent follows the canonical contract."""

        if self.T_ref_from_agent.shape != (4, 4):
            raise ValueError(
                f"Agent '{self.agent_id}' T_ref_from_agent "
                f"must have shape [4, 4], "
                f"got {self.T_ref_from_agent.shape}."
            )

        if self.lidar is not None:
            self.lidar.validate()

        for camera in self.cameras:
            camera.validate()


# ---------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------


@dataclass
class AnnotationData:
    """
    Canonical 3D object annotations.

    boxes_3d:
        Shape [M, 7].

        Each row:

            [x, y, z, length, width, height, yaw]

        All boxes are expressed in the reference-agent coordinate frame.

    native_labels:
        Original semantic class names from the source dataset.

        Examples:
            "Car"
            "Pedestrian"
            "Truck"

        We preserve these because the final cross-dataset class taxonomy
        may be configured separately.

    labels:
        Optional canonical integer class IDs.
        Shape [M].

        This can remain None until a cross-dataset class mapping
        is selected.

    track_ids:
        Shape [M].

        Canonical integer tracking IDs.
        Use -1 if an object has no usable track ID.

    velocity_xy:
        Optional shape [M, 2].

        Object velocity:

            [vx, vy]

        expressed in the reference-agent frame.
    """

    boxes_3d: np.ndarray

    native_labels: list[str]

    track_ids: np.ndarray

    labels: Optional[np.ndarray] = None

    velocity_xy: Optional[np.ndarray] = None

    extras: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Check that annotations follow the canonical contract."""

        if not isinstance(self.boxes_3d, np.ndarray):
            raise TypeError("annotations.boxes_3d must be a numpy array.")

        if self.boxes_3d.ndim != 2 or self.boxes_3d.shape[1] != 7:
            raise ValueError(
                "annotations.boxes_3d must have shape [M, 7], "
                f"got {self.boxes_3d.shape}."
            )

        num_objects = self.boxes_3d.shape[0]

        if len(self.native_labels) != num_objects:
            raise ValueError(
                "Number of native_labels must match number of 3D boxes."
            )

        if self.track_ids.shape != (num_objects,):
            raise ValueError(
                "annotations.track_ids must have shape [M], "
                f"got {self.track_ids.shape}."
            )

        if self.labels is not None:
            if self.labels.shape != (num_objects,):
                raise ValueError(
                    "annotations.labels must have shape [M], "
                    f"got {self.labels.shape}."
                )

        if self.velocity_xy is not None:
            if self.velocity_xy.shape != (num_objects, 2):
                raise ValueError(
                    "annotations.velocity_xy must have shape [M, 2], "
                    f"got {self.velocity_xy.shape}."
                )


# ---------------------------------------------------------------------
# Complete canonical V2X sample
# ---------------------------------------------------------------------


@dataclass
class CanonicalV2XSample:
    """
    One canonical cooperative V2X sample.

    Conceptually, one sample represents one synchronized frame/timestamp
    from one sequence.

    Every dataset adapter must return this object.
    """

    source: SourceInfo

    timestamp_s: float

    reference_agent_id: str

    agents: list[AgentData]

    annotations: AnnotationData

    schema_version: str = "0.1"

    extras: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """
        Validate the complete canonical sample.

        Dataset adapters should call this before returning a sample.
        """

        if not self.agents:
            raise ValueError(
                "A CanonicalV2XSample must contain at least one agent."
            )

        # -------------------------------------------------------------
        # Validate unique agent IDs
        # -------------------------------------------------------------

        agent_ids = [agent.agent_id for agent in self.agents]

        if len(agent_ids) != len(set(agent_ids)):
            raise ValueError(
                "Every agent_id must be unique inside one sample."
            )

        # -------------------------------------------------------------
        # Check reference agent
        # -------------------------------------------------------------

        if self.reference_agent_id not in agent_ids:
            raise ValueError(
                f"reference_agent_id '{self.reference_agent_id}' "
                "does not exist in agents."
            )

        reference_agents = [
            agent
            for agent in self.agents
            if agent.is_reference
        ]

        if len(reference_agents) != 1:
            raise ValueError(
                "Exactly one agent must have is_reference=True."
            )

        reference_agent = reference_agents[0]

        if reference_agent.agent_id != self.reference_agent_id:
            raise ValueError(
                "reference_agent_id does not match the agent "
                "with is_reference=True."
            )

        # The reference transform should be identity.
        if not np.allclose(
            reference_agent.T_ref_from_agent,
            np.eye(4),
            atol=1e-5,
        ):
            raise ValueError(
                "The reference agent must have "
                "T_ref_from_agent = identity matrix."
            )

        # -------------------------------------------------------------
        # Validate every agent
        # -------------------------------------------------------------

        for agent in self.agents:
            agent.validate()

        # -------------------------------------------------------------
        # Validate annotations
        # -------------------------------------------------------------

        self.annotations.validate()