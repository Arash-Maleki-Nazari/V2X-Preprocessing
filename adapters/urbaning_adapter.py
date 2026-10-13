"""


Adapter for UrbanIng-V2X.

This adapter uses the official UrbanIng devkit to read the native
dataset and converts each synchronized UrbanIng frame into our
CanonicalV2XSample defined in schema.py.

Canonical choices used here
---------------------------
Reference agent:
    vehicle1 by default.

Vehicle local frame:
    vehicle body coordinate frame.

Infrastructure local frame:
    UrbanIng global coordinate frame.
    The official devkit represents the infrastructure grouping in
    the global frame.

LiDAR:
    All LiDARs belonging to one agent are combined into one
    canonical LidarData object.

Camera images:
    Images are undistorted using the official UrbanIng devkit and
    converted from OpenCV BGR ordering to RGB.

Annotations:
    [x, y, z, length, width, height, yaw]

    All annotations are transformed into the reference-agent frame.

Timestamp:
    Seconds relative to the start of the sequence.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Sequence as TypingSequence

import numpy as np
from scipy.spatial.transform import Rotation

from urbaning.data.sequence import Sequence

from adapters.base_adapter import BaseAdapter
from schema import (
    AgentData,
    AnnotationData,
    CameraData,
    CanonicalV2XSample,
    LidarData,
    SourceInfo,
)


class UrbanIngAdapter(BaseAdapter):
    """
    Convert UrbanIng-V2X into the canonical V2X representation.

    Parameters
    ----------
    dataset_root:
        Root directory of the extracted UrbanIng-V2X dataset.

        Expected structure:

            dataset_root/
            ├── dataset/
            ├── labels/
            ├── av_vehicle_data.json
            ├── labels_av_track_ids.json
            └── crossings_lanelet2map.osm

    sequence_names:
        Optional list of UrbanIng sequence names to use.

        If None, every sequence inside dataset_root/dataset is used.

    reference_agent_id:
        Agent whose coordinate frame becomes the canonical reference
        frame.

        Default:
            "vehicle1"
    """

    def __init__(
        self,
        dataset_root: str | Path,
        sequence_names: Optional[TypingSequence[str]] = None,
        reference_agent_id: str = "vehicle1",
    ):
        super().__init__(
            dataset_root=dataset_root,
            split=None,
        )

        self.reference_agent_id = reference_agent_id

        # -------------------------------------------------------------
        # Check basic dataset structure
        # -------------------------------------------------------------

        self.data_dir = self.dataset_root / "dataset"

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"UrbanIng dataset directory not found: {self.data_dir}"
            )

        # -------------------------------------------------------------
        # Select sequences
        # -------------------------------------------------------------

        if sequence_names is None:
            self.sequence_names = sorted(
                path.name
                for path in self.data_dir.iterdir()
                if path.is_dir()
            )
        else:
            self.sequence_names = list(sequence_names)

        if not self.sequence_names:
            raise ValueError("No UrbanIng sequences were found.")

        # -------------------------------------------------------------
        # Build global adapter index
        #
        # Example:
        #
        #   adapter[0]
        #       -> sequence A, frame 0
        #
        #   adapter[1]
        #       -> sequence A, frame 1
        #
        #   ...
        # -------------------------------------------------------------

        self._samples: list[tuple[str, int]] = []

        # Sequence start times are needed to convert UrbanIng's UTC
        # timestamps into relative sequence timestamps.
        self._sequence_start_time_s: dict[str, float] = {}

        for sequence_name in self.sequence_names:

            sequence_path = self.data_dir / sequence_name

            if not sequence_path.exists():
                raise FileNotFoundError(
                    f"UrbanIng sequence not found: {sequence_path}"
                )

            num_frames, start_time_s = self._read_sequence_timing(
                sequence_name
            )

            self._sequence_start_time_s[sequence_name] = start_time_s

            for frame_index in range(num_frames):
                self._samples.append(
                    (sequence_name, frame_index)
                )

        # -------------------------------------------------------------
        # Keep only one loaded official Sequence object in memory.
        #
        # Sequential access is common, so this avoids repeatedly
        # loading the same sequence while also avoiding keeping the
        # entire dataset in memory.
        # -------------------------------------------------------------

        self._cached_sequence_name: Optional[str] = None
        self._cached_sequence: Optional[Sequence] = None

    # ------------------------------------------------------------------
    # BaseAdapter interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """
        Return total number of synchronized UrbanIng frames.
        """

        return len(self._samples)

    def get_sample(
        self,
        index: int,
    ) -> CanonicalV2XSample:
        """
        Convert one UrbanIng frame into CanonicalV2XSample.
        """

        sequence_name, frame_index = self._samples[index]

        sequence = self._get_sequence(sequence_name)

        # Official UrbanIng synchronized frame.
        frame = sequence[frame_index]

        # -------------------------------------------------------------
        # Find reference agent
        # -------------------------------------------------------------

        reference_kind, reference_object = (
            self._get_reference_agent(frame)
        )

        # Transformation:
        #
        #     global -> reference
        #
        # This will later be used to construct:
        #
        #     T_ref_from_agent
        # -------------------------------------------------------------

        T_ref_from_global = self._reference_from_global(
            reference_kind,
            reference_object,
        )

        # -------------------------------------------------------------
        # Vehicles
        # -------------------------------------------------------------

        agents: list[AgentData] = []

        for vehicle_id, vehicle in frame.vehicles.items():

            agent = self._convert_vehicle(
                frame=frame,
                vehicle_id=vehicle_id,
                vehicle=vehicle,
                T_ref_from_global=T_ref_from_global,
            )

            agents.append(agent)

        # -------------------------------------------------------------
        # Infrastructure
        # -------------------------------------------------------------

        for infrastructure_id, infrastructure in (
            frame.infrastructures.items()
        ):

            agent = self._convert_infrastructure(
                frame=frame,
                infrastructure_id=infrastructure_id,
                infrastructure=infrastructure,
                T_ref_from_global=T_ref_from_global,
            )

            agents.append(agent)

        # -------------------------------------------------------------
        # Annotations
        # -------------------------------------------------------------

        annotations = self._convert_annotations(
            frame=frame,
            reference_kind=reference_kind,
            reference_object=reference_object,
        )

        # -------------------------------------------------------------
        # Canonical relative timestamp
        # -------------------------------------------------------------

        sequence_start_s = self._sequence_start_time_s[
            sequence_name
        ]

        timestamp_s = float(
            frame.timestamp - sequence_start_s
        )

        # Use the original column identifier from timesync_info.csv
        # as frame_id.
        frame_id = str(
            sequence.time_sync_df.columns[frame_index]
        )

        # -------------------------------------------------------------
        # Create canonical sample
        # -------------------------------------------------------------

        sample = CanonicalV2XSample(

            source=SourceInfo(
                dataset="UrbanIng-V2X",
                sequence_id=sequence_name,
                frame_id=frame_id,
            ),

            timestamp_s=timestamp_s,

            reference_agent_id=self.reference_agent_id,

            agents=agents,

            annotations=annotations,

            extras={
                # Keep the original absolute timestamp for traceability.
                "urbaning_timestamp_s": float(frame.timestamp),
            },
        )

        return sample

    # ------------------------------------------------------------------
    # Sequence handling
    # ------------------------------------------------------------------

    def _get_sequence(
        self,
        sequence_name: str,
    ) -> Sequence:
        """
        Load an UrbanIng Sequence using the official devkit.

        Only the most recently used sequence is cached.
        """

        if self._cached_sequence_name != sequence_name:

            self._cached_sequence = Sequence(
                str(self.dataset_root),
                sequence_name,
            )

            self._cached_sequence_name = sequence_name

        return self._cached_sequence

    def _read_sequence_timing(
        self,
        sequence_name: str,
    ) -> tuple[int, float]:
        """
        Read the number of frames and the first synchronized timestamp
        directly from timesync_info.csv.

        UrbanIng stores synchronized time steps as columns.

        Returns
        -------
        num_frames:
            Number of synchronized frames.

        start_time_s:
            First frame timestamp in seconds.
        """

        sync_path = (
            self.data_dir
            / sequence_name
            / "timesync_info.csv"
        )

        if not sync_path.exists():
            raise FileNotFoundError(
                f"Missing synchronization file: {sync_path}"
            )

        with open(
            sync_path,
            "r",
            newline="",
            encoding="utf-8",
        ) as file:

            reader = csv.reader(file)

            header = next(reader)

            # First column contains the row/index name.
            num_frames = len(header) - 1

            start_timestamp_ms = None

            for row in reader:

                if row and row[0] == "timestamp_ms":

                    if len(row) < 2:
                        raise ValueError(
                            f"No timestamps found in {sync_path}"
                        )

                    start_timestamp_ms = float(row[1])
                    break

        if start_timestamp_ms is None:
            raise ValueError(
                f"'timestamp_ms' row not found in {sync_path}"
            )

        return (
            num_frames,
            start_timestamp_ms / 1000.0,
        )

    # ------------------------------------------------------------------
    # Reference frame
    # ------------------------------------------------------------------

    def _get_reference_agent(self, frame):
        """
        Find the requested reference agent in this frame.

        Returns
        -------
        ("vehicle", vehicle)
            or

        ("infrastructure", infrastructure)
        """

        if self.reference_agent_id in frame.vehicles:

            vehicle = frame.vehicles[
                self.reference_agent_id
            ]

            if vehicle.state is None:
                raise ValueError(
                    f"Reference vehicle "
                    f"'{self.reference_agent_id}' "
                    "has no vehicle state."
                )

            return "vehicle", vehicle

        if self.reference_agent_id in frame.infrastructures:

            return (
                "infrastructure",
                frame.infrastructures[
                    self.reference_agent_id
                ],
            )

        available_agents = (
            list(frame.vehicles.keys())
            + list(frame.infrastructures.keys())
        )

        raise ValueError(
            f"Reference agent '{self.reference_agent_id}' "
            f"not present in frame. "
            f"Available agents: {available_agents}"
        )

    @staticmethod
    def _reference_from_global(
        reference_kind: str,
        reference_object,
    ) -> np.ndarray:
        """
        Return T_ref_from_global.
        """

        if reference_kind == "vehicle":

            # UrbanIng:
            #
            # gTv_body = vehicle -> global
            # vTg_body = global -> vehicle
            #
            return np.asarray(
                reference_object.state.vTg_body,
                dtype=np.float32,
            )

        if reference_kind == "infrastructure":

            # The official UrbanIng Infrastructure object uses the
            # global frame as its infrastructure grouping frame.
            return np.eye(4, dtype=np.float32)

        raise ValueError(
            f"Unknown reference kind: {reference_kind}"
        )

    # ------------------------------------------------------------------
    # Vehicle conversion
    # ------------------------------------------------------------------

    def _convert_vehicle(
        self,
        frame,
        vehicle_id: str,
        vehicle,
        T_ref_from_global: np.ndarray,
    ) -> AgentData:
        """
        Convert one UrbanIng vehicle into AgentData.
        """

        if vehicle.state is None:
            raise ValueError(
                f"Vehicle '{vehicle_id}' has no state information."
            )

        # UrbanIng:
        #
        # gTv_body = vehicle body -> global
        #
        T_global_from_agent = np.asarray(
            vehicle.state.gTv_body,
            dtype=np.float32,
        )

        T_ref_from_agent = (
            T_ref_from_global
            @ T_global_from_agent
        ).astype(np.float32)

        if vehicle_id == self.reference_agent_id:
            T_ref_from_agent = np.eye(
                4,
                dtype=np.float32,
            )

        lidar = self._convert_vehicle_lidar(
            frame,
            vehicle,
        )

        cameras = self._convert_cameras(
            frame=frame,
            cameras=vehicle.cameras,
            T_agent_from_global=np.asarray(
                vehicle.state.vTg_body,
                dtype=np.float32,
            ),
        )

        return AgentData(

            agent_id=vehicle_id,

            agent_type="vehicle",

            is_reference=(
                vehicle_id == self.reference_agent_id
            ),

            T_ref_from_agent=T_ref_from_agent,

            lidar=lidar,

            cameras=cameras,

            extras={
                "velocity_global": np.asarray(
                    vehicle.state.velocity,
                    dtype=np.float32,
                ),
                "acceleration_global": np.asarray(
                    vehicle.state.acceleration,
                    dtype=np.float32,
                ),
                "rotation_rates_global": np.asarray(
                    vehicle.state.rotation_rates,
                    dtype=np.float32,
                ),
            },
        )

    # ------------------------------------------------------------------
    # Infrastructure conversion
    # ------------------------------------------------------------------

    def _convert_infrastructure(
        self,
        frame,
        infrastructure_id: str,
        infrastructure,
        T_ref_from_global: np.ndarray,
    ) -> AgentData:
        """
        Convert one UrbanIng infrastructure grouping into AgentData.

        UrbanIng's official devkit uses the global coordinate system
        as the infrastructure grouping coordinate frame.
        """

        T_global_from_agent = np.eye(
            4,
            dtype=np.float32,
        )

        T_ref_from_agent = (
            T_ref_from_global
            @ T_global_from_agent
        ).astype(np.float32)

        if infrastructure_id == self.reference_agent_id:
            T_ref_from_agent = np.eye(
                4,
                dtype=np.float32,
            )

        lidar = self._convert_infrastructure_lidar(
            frame,
            infrastructure,
        )

        cameras = self._convert_cameras(
            frame=frame,
            cameras=infrastructure.cameras,
            T_agent_from_global=np.eye(
                4,
                dtype=np.float32,
            ),
        )

        return AgentData(

            agent_id=infrastructure_id,

            agent_type="infrastructure",

            is_reference=(
                infrastructure_id
                == self.reference_agent_id
            ),

            T_ref_from_agent=T_ref_from_agent,

            lidar=lidar,

            cameras=cameras,

            extras={
                "lidar_sensor_names": list(
                    infrastructure.lidars.keys()
                ),
                "camera_sensor_names": list(
                    infrastructure.cameras.keys()
                ),
            },
        )

    # ------------------------------------------------------------------
    # LiDAR conversion
    # ------------------------------------------------------------------

    def _convert_vehicle_lidar(
        self,
        frame,
        vehicle,
    ) -> Optional[LidarData]:
        """
        Convert all LiDARs belonging to a vehicle.

        Points are expressed in the vehicle body frame.
        """

        if not vehicle.lidars:
            return None

        xyz_parts = []
        intensity_parts = []
        time_parts = []
        index_parts = []

        for lidar in vehicle.lidars.values():

            # Official UrbanIng devkit returns:
            #
            # [x, y, z, intensity, time_offset_ms, lidar_index]
            #
            # and transforms only xyz, preserving remaining columns.
            points = (
                lidar.point_cloud_in_vehicle_body_coordinates(
                    vehicle
                )
            )

            xyz_parts.append(
                points[:, :3].astype(
                    np.float32,
                    copy=False,
                )
            )

            intensity_parts.append(
                points[:, 3].astype(
                    np.float32,
                    copy=False,
                )
            )

            time_parts.append(
                self._canonical_point_times(
                    frame=frame,
                    sensor_path=lidar.lidar_file_path,
                    raw_time_offset_ms=points[:, 4],
                )
            )

            index_parts.append(
                points[:, 5].astype(
                    np.int64,
                    copy=False,
                )
            )

        return LidarData(

            xyz=np.concatenate(
                xyz_parts,
                axis=0,
            ),

            intensity=np.concatenate(
                intensity_parts,
                axis=0,
            ),

            time_offset_s=np.concatenate(
                time_parts,
                axis=0,
            ),

            sensor_index=np.concatenate(
                index_parts,
                axis=0,
            ),

            extras={
                "coordinate_frame": "vehicle_body",
            },
        )

    def _convert_infrastructure_lidar(
        self,
        frame,
        infrastructure,
    ) -> Optional[LidarData]:
        """
        Convert infrastructure LiDAR data.

        Infrastructure points are expressed in the UrbanIng global
        frame, which is used here as the infrastructure agent frame.
        """

        if not infrastructure.lidars:
            return None

        xyz_parts = []
        intensity_parts = []
        time_parts = []
        index_parts = []

        for lidar in infrastructure.lidars.values():

            points = (
                lidar.point_cloud_in_global_coordinates()
            )

            xyz_parts.append(
                points[:, :3].astype(
                    np.float32,
                    copy=False,
                )
            )

            intensity_parts.append(
                points[:, 3].astype(
                    np.float32,
                    copy=False,
                )
            )

            time_parts.append(
                self._canonical_point_times(
                    frame=frame,
                    sensor_path=lidar.lidar_file_path,
                    raw_time_offset_ms=points[:, 4],
                )
            )

            index_parts.append(
                points[:, 5].astype(
                    np.int64,
                    copy=False,
                )
            )

        return LidarData(

            xyz=np.concatenate(
                xyz_parts,
                axis=0,
            ),

            intensity=np.concatenate(
                intensity_parts,
                axis=0,
            ),

            time_offset_s=np.concatenate(
                time_parts,
                axis=0,
            ),

            sensor_index=np.concatenate(
                index_parts,
                axis=0,
            ),

            extras={
                "coordinate_frame": "urbaning_global",
            },
        )

    def _canonical_point_times(
        self,
        frame,
        sensor_path: str,
        raw_time_offset_ms: np.ndarray,
    ) -> np.ndarray:
        """
        Convert UrbanIng per-point timing into seconds relative to
        the synchronized canonical frame timestamp.

        Canonical relation:

            point offset
                =
            sensor timestamp - frame timestamp
                +
            UrbanIng per-point scan offset
        """

        sensor_timestamp_s = self._timestamp_from_path(
            sensor_path
        )

        if sensor_timestamp_s is None:

            # We still preserve the UrbanIng per-point timing if the
            # filename cannot be interpreted.
            sensor_to_frame_offset_s = 0.0

        else:

            sensor_to_frame_offset_s = (
                sensor_timestamp_s
                - float(frame.timestamp)
            )

        return (
            sensor_to_frame_offset_s
            + np.asarray(
                raw_time_offset_ms,
                dtype=np.float32,
            )
            / 1000.0
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Camera conversion
    # ------------------------------------------------------------------

    def _convert_cameras(
        self,
        frame,
        cameras,
        T_agent_from_global: np.ndarray,
    ) -> list[CameraData]:
        """
        Convert UrbanIng camera objects into canonical CameraData.
        """

        canonical_cameras: list[CameraData] = []

        for camera_id, camera in cameras.items():

            # ---------------------------------------------------------
            # Image
            #
            # UrbanIng's official CameraData uses OpenCV internally,
            # therefore the returned image is BGR.
            #
            # We use the devkit's undistorted image because its iTc
            # matrix corresponds to the processed camera geometry.
            # ---------------------------------------------------------

            image_bgr = camera.undistorted_image

            if image_bgr is None:
                raise ValueError(
                    f"Could not load camera image: "
                    f"{camera.image_path}"
                )

            if (
                image_bgr.ndim == 3
                and image_bgr.shape[2] == 3
            ):
                image = image_bgr[
                    :, :, ::-1
                ].copy()
            else:
                image = image_bgr.copy()

            # ---------------------------------------------------------
            # Camera transform
            #
            # UrbanIng:
            #
            # cTg = global -> camera
            #
            # therefore:
            #
            # global_from_camera = inverse(cTg)
            #
            # and:
            #
            # agent_from_camera
            #     =
            # agent_from_global @ global_from_camera
            # ---------------------------------------------------------

            T_global_from_camera = np.linalg.inv(
                np.asarray(
                    camera.cTg,
                    dtype=np.float64,
                )
            )

            T_agent_from_camera = (
                T_agent_from_global
                @ T_global_from_camera
            ).astype(np.float32)

            # ---------------------------------------------------------
            # Sensor timing
            # ---------------------------------------------------------

            camera_timestamp_s = (
                self._timestamp_from_path(
                    camera.image_path
                )
            )

            if camera_timestamp_s is None:

                time_offset_s = 0.0

            else:

                time_offset_s = float(
                    camera_timestamp_s
                    - frame.timestamp
                )

            # Infrastructure camera names explicitly contain
            # "thermal_camera".
            modality = (
                "thermal"
                if "thermal" in camera_id.lower()
                else "rgb"
            )

            canonical_cameras.append(

                CameraData(

                    camera_id=camera_id,

                    modality=modality,

                    image=image,

                    K=np.asarray(
                        camera.iTc,
                        dtype=np.float32,
                    ),

                    T_agent_from_camera=(
                        T_agent_from_camera
                    ),

                    time_offset_s=time_offset_s,

                    # We use the devkit's already-undistorted image.
                    distortion=None,

                    extras={
                        "source_path": str(
                            camera.image_path
                        ),
                        "undistorted": True,
                    },
                )
            )

        return canonical_cameras

    # ------------------------------------------------------------------
    # Annotation conversion
    # ------------------------------------------------------------------

    def _convert_annotations(
        self,
        frame,
        reference_kind: str,
        reference_object,
    ) -> AnnotationData:
        """
        Convert UrbanIng annotations into:

            [x, y, z, length, width, height, yaw]

        in the reference-agent coordinate frame.
        """

        # Frames without labels are valid.
        if frame.labels is None:

            return AnnotationData(

                boxes_3d=np.empty(
                    (0, 7),
                    dtype=np.float32,
                ),

                native_labels=[],

                track_ids=np.empty(
                    (0,),
                    dtype=np.int64,
                ),

                labels=None,

                velocity_xy=None,

                extras={
                    "attributes": [],
                },
            )

        # -------------------------------------------------------------
        # Transform official UrbanIng ObjectLabels into reference frame.
        # -------------------------------------------------------------

        if reference_kind == "vehicle":

            objects = (
                frame.labels
                .objects_in_vehicle_body_coordinates(
                    reference_object
                )
            )

        elif reference_kind == "infrastructure":

            # Our infrastructure agent frame is UrbanIng global.
            objects = (
                frame.labels
                .objects_in_global_coordinates()
            )

        else:

            raise ValueError(
                f"Unknown reference kind: "
                f"{reference_kind}"
            )

        boxes = []
        native_labels = []
        track_ids = []
        attributes = []

        for obj in objects.values():

            position = np.asarray(
                obj.position,
                dtype=np.float32,
            )

            dimensions = np.asarray(
                obj.dimension,
                dtype=np.float32,
            )

            # UrbanIng quaternion:
            #
            # [x, y, z, w]
            #
            # After transforming the ObjectLabel into the reference
            # coordinate system, extract canonical yaw about +z.
            rotation = Rotation.from_quat(
                np.asarray(
                    obj.quaternion,
                    dtype=np.float64,
                )
            )

            yaw = float(
                rotation.as_euler(
                    "ZYX",
                    degrees=False,
                )[0]
            )

            length = float(dimensions[0])
            width = float(dimensions[1])
            height = float(dimensions[2])

            boxes.append(
                [
                    float(position[0]),
                    float(position[1]),
                    float(position[2]),
                    length,
                    width,
                    height,
                    yaw,
                ]
            )

            native_labels.append(
                str(obj.object_type)
            )

            track_ids.append(
                int(obj.track_id)
            )

            attributes.append(
                dict(obj.attributes)
            )

        # -------------------------------------------------------------
        # Convert to fixed canonical numpy shapes.
        # -------------------------------------------------------------

        if boxes:

            boxes_3d = np.asarray(
                boxes,
                dtype=np.float32,
            )

            track_ids_array = np.asarray(
                track_ids,
                dtype=np.int64,
            )

        else:

            boxes_3d = np.empty(
                (0, 7),
                dtype=np.float32,
            )

            track_ids_array = np.empty(
                (0,),
                dtype=np.int64,
            )

        return AnnotationData(

            boxes_3d=boxes_3d,

            native_labels=native_labels,

            # Canonical cross-dataset class mapping has intentionally
            # not been selected yet.
            labels=None,

            track_ids=track_ids_array,

            velocity_xy=None,

            extras={
                "attributes": attributes,
            },
        )

    # ------------------------------------------------------------------
    # Timestamp helper
    # ------------------------------------------------------------------

    @staticmethod
    def _timestamp_from_path(
        path: str | Path,
    ) -> Optional[float]:
        """
        Extract UrbanIng sensor timestamp from a filename.

        Example:

            1732632673950.npz

        becomes:

            1732632673.950 seconds

        Returns None if the filename is not numeric.
        """

        stem = Path(path).stem

        try:
            timestamp_ms = int(stem)
        except ValueError:
            return None

        return timestamp_ms / 1000.0