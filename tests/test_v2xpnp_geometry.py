"""Validate OpenCOOD-style LiDAR pose transformations for V2XPnP.

Run from the Preprocessing project root:
    python -m tests.test_v2xpnp_geometry

Requires: numpy, PyYAML
"""

from pathlib import Path
from tests.cli import parser_for, select_v2xpnp

import numpy as np
import yaml


ATOL = 1e-8


def load_lidar_pose(agent_id: str, scenario_dir: Path, frame: str, *, expected_infra: bool) -> np.ndarray:
    """Read the six-value [x, y, z, roll, yaw, pitch] LiDAR pose."""
    path = scenario_dir / agent_id / f"{frame}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find YAML file: {path}")

    with path.open("r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f)

    if not isinstance(metadata, dict) or "lidar_pose" not in metadata:
        raise ValueError(f"Missing 'lidar_pose' in {path}")

    pose = np.asarray(metadata["lidar_pose"], dtype=np.float64)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError(f"Invalid six-value lidar_pose in {path}: {pose}")

    if metadata.get("infra") is not expected_infra:
        raise ValueError(
            f"Unexpected 'infra' flag in {path}: "
            f"expected {expected_infra}, got {metadata.get('infra')!r}"
        )

    return pose


def x_to_world(pose: np.ndarray) -> np.ndarray:
    """OpenCOOD convention: [x, y, z, roll, yaw, pitch], angles in degrees."""
    x, y, z, roll, yaw, pitch = pose

    c_y, s_y = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    c_r, s_r = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    c_p, s_p = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = [x, y, z]

    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix


def x1_to_x2(pose1: np.ndarray, pose2: np.ndarray) -> np.ndarray:
    """Transform homogeneous coordinates from agent 1 to agent 2."""
    return np.linalg.inv(x_to_world(pose2)) @ x_to_world(pose1)


def check_rotation(name: str, transform: np.ndarray) -> None:
    """Ensure R is orthogonal with determinant +1 and T is homogeneous."""
    rotation = transform[:3, :3]
    np.testing.assert_allclose(
        rotation.T @ rotation, np.eye(3), atol=ATOL, rtol=0,
        err_msg=f"{name}: rotation is not orthogonal",
    )
    np.testing.assert_allclose(
        np.linalg.det(rotation), 1.0, atol=ATOL, rtol=0,
        err_msg=f"{name}: rotation determinant is not +1",
    )
    np.testing.assert_allclose(
        transform[3, :], [0, 0, 0, 1], atol=ATOL, rtol=0,
        err_msg=f"{name}: invalid homogeneous last row",
    )
    print(f"PASS  {name}: orthogonal rotation, determinant +1")


def main(argv=None) -> None:
    args = parser_for("v2xpnp").parse_args(argv)
    SCENARIO_DIR, FRAME, VEHICLE_ID, INFRA_ID = select_v2xpnp(args)
    print(f"Scenario: {SCENARIO_DIR.name} | Frame: {FRAME}")
    vehicle_pose = load_lidar_pose(VEHICLE_ID, SCENARIO_DIR, FRAME, expected_infra=False)
    infra_pose = load_lidar_pose(INFRA_ID, SCENARIO_DIR, FRAME, expected_infra=True)
    print(f"Vehicle ({VEHICLE_ID}) pose: {vehicle_pose}")
    print(f"Infrastructure ({INFRA_ID}) pose: {infra_pose}")

    t_world_from_vehicle = x_to_world(vehicle_pose)
    t_world_from_infra = x_to_world(infra_pose)

    # Test 1: Both agents map to themselves by an identity transformation.
    for name, pose in (("vehicle", vehicle_pose), ("infrastructure", infra_pose)):
        np.testing.assert_allclose(
            x1_to_x2(pose, pose), np.eye(4), atol=ATOL, rtol=0,
            err_msg=f"{name}: self-to-self transform must be identity",
        )
        print(f"PASS  {name}: self-transform is identity")

    # Test 2: Transform from infrastructure to vehicle and the reverse.
    t_vehicle_from_infra = x1_to_x2(infra_pose, vehicle_pose)
    t_infra_from_vehicle = x1_to_x2(vehicle_pose, infra_pose)
    np.testing.assert_allclose(
        t_vehicle_from_infra @ t_infra_from_vehicle,
        np.eye(4), atol=ATOL, rtol=0,
        err_msg="Forward and reverse agent transforms are not inverses",
    )
    np.testing.assert_allclose(
        t_infra_from_vehicle @ t_vehicle_from_infra,
        np.eye(4), atol=ATOL, rtol=0,
        err_msg="Reverse and forward agent transforms are not inverses",
    )
    print("PASS  vehicle/infrastructure transforms are mutual inverses")

    # Test 3: Relative transforms agree with world-frame composition.
    np.testing.assert_allclose(
        t_world_from_vehicle @ t_vehicle_from_infra,
        t_world_from_infra, atol=ATOL, rtol=0,
        err_msg="Infrastructure-to-vehicle transform disagrees with world poses",
    )
    print("PASS  relative transform agrees with world-frame composition")

    # Test 4: All world and relative rotations are proper rotations.
    for name, transform in (
        ("world_from_vehicle", t_world_from_vehicle),
        ("world_from_infra", t_world_from_infra),
        ("vehicle_from_infra", t_vehicle_from_infra),
        ("infra_from_vehicle", t_infra_from_vehicle),
    ):
        check_rotation(name, transform)

    print("\nT_vehicle_from_infra (OpenCOOD native coordinates):")
    print(np.array2string(t_vehicle_from_infra, precision=6, suppress_small=True))
    print("\nALL V2XPnP GEOMETRY TESTS PASSED")
    print("Note: This checks matrix consistency, not point-cloud alignment or canonical handedness.")


def test_validation(cli_args):
    main(cli_args("v2xpnp"))


if __name__ == "__main__":
    main()
