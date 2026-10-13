"""
V2XPnP multi-agent LiDAR alignment validation.

Validation:
1. Load both agents' LiDAR point clouds and YAML metadata.
2. Verify agent types and LiDAR data integrity.
3. Compute relative transforms using OpenCOOD conventions.
4. Transform infrastructure LiDAR into the vehicle frame.
5. Verify identity, inverse and round-trip transformations.
6. Generate BEV visualizations of both agents in one frame.
7. Save the figures for visual alignment inspection.

Coordinates remain in OpenCOOD's native convention.
Canonical right-handed conversion is NOT performed here.

Requirements:
    pip install numpy PyYAML matplotlib

Run:
    python -m tests.test_v2xpnp_lidar_alignment
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from tests.test_v2xpnp_geometry import x1_to_x2
from tests.cli import parser_for, select_v2xpnp
from project_paths import output_path


# ============================================================
# Configuration
# ============================================================


MAX_PLOT_POINTS = 80000
RANDOM_SEED = 42

OVERVIEW_RANGE_M = 200
ZOOM_RANGE_M = 80

ATOL = 1e-7


# ============================================================
# 1. Load YAML metadata
# ============================================================

def load_metadata(agent_id, scenario_dir, frame):
    """
    Load YAML metadata for one agent at the selected frame.
    """

    yaml_path = scenario_dir / agent_id / f"{frame}.yaml"

    if not yaml_path.is_file():
        raise FileNotFoundError(
            f"YAML file not found: {yaml_path}"
        )

    with yaml_path.open("r", encoding="utf-8") as file:
        metadata = yaml.safe_load(file)

    if not isinstance(metadata, dict):
        raise ValueError(
            f"Invalid YAML metadata: {yaml_path}"
        )

    if "lidar_pose" not in metadata:
        raise ValueError(
            f"Missing lidar_pose: {yaml_path}"
        )

    if "infra" not in metadata:
        raise ValueError(
            f"Missing infra flag: {yaml_path}"
        )

    pose = np.asarray(
        metadata["lidar_pose"],
        dtype=np.float64
    )

    if pose.shape != (6,):
        raise ValueError(
            f"Expected 6 pose values, got {pose.shape}"
        )

    if not np.isfinite(pose).all():
        raise ValueError(
            f"Non-finite pose values: {yaml_path}"
        )

    return metadata, pose


# ============================================================
# 2. Load LiDAR binary
# ============================================================

def load_lidar(agent_id, scenario_dir, frame):
    """
    Load V2XPnP LiDAR data.

    Observed sample binary format:
        float32 [x, y, z, intensity]

    Returns:
        xyz       (N, 3)
        intensity (N,)
    """

    bin_path = scenario_dir / agent_id / f"{frame}.bin"

    if not bin_path.is_file():
        raise FileNotFoundError(
            f"LiDAR file not found: {bin_path}"
        )

    raw = np.fromfile(
        bin_path,
        dtype="<f4"
    )

    if raw.size == 0:
        raise ValueError(
            f"Empty LiDAR file: {bin_path}"
        )

    if raw.size % 4 != 0:
        raise ValueError(
            f"Invalid LiDAR layout in {bin_path}: "
            f"{raw.size} float32 values cannot form Nx4 points"
        )

    points = raw.reshape(-1, 4)

    if not np.isfinite(points).all():
        raise ValueError(
            f"NaN or Inf values in {bin_path}"
        )

    xyz = points[:, :3].copy()
    intensity = points[:, 3].copy()

    return xyz, intensity


# ============================================================
# 3. Apply homogeneous transformation
# ============================================================

def transform_points(xyz, transformation):
    """
    Transform Nx3 points using a 4x4 matrix.

    p_target = R @ p_source + t

    Point coordinates are stored as rows, so:
        xyz_target = xyz @ R.T + t
    """

    xyz = np.asarray(xyz, dtype=np.float64)
    transformation = np.asarray(
        transformation,
        dtype=np.float64
    )

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(
            "Point cloud must have shape (N, 3)"
        )

    if transformation.shape != (4, 4):
        raise ValueError(
            "Transformation must have shape (4, 4)"
        )

    rotation = transformation[:3, :3]
    translation = transformation[:3, 3]

    transformed_xyz = xyz @ rotation.T + translation

    return transformed_xyz


# ============================================================
# 4. Validate the relative transformation
# ============================================================

def validate_transformations(
    vehicle_pose,
    infra_pose,
    infra_xyz,
    T_vehicle_from_infra
):
    """
    Check identity, inverse and point round-trip consistency.
    """

    identity = np.eye(4)

    # Reference agent must map to itself.
    T_vehicle_from_vehicle = x1_to_x2(
        vehicle_pose,
        vehicle_pose
    )

    np.testing.assert_allclose(
        T_vehicle_from_vehicle,
        identity,
        atol=ATOL,
        rtol=0
    )

    print("PASS: Vehicle self-transform is identity")

    # Reverse transformation.
    T_infra_from_vehicle = x1_to_x2(
        vehicle_pose,
        infra_pose
    )

    np.testing.assert_allclose(
        T_vehicle_from_infra @ T_infra_from_vehicle,
        identity,
        atol=ATOL,
        rtol=0
    )

    np.testing.assert_allclose(
        T_infra_from_vehicle @ T_vehicle_from_infra,
        identity,
        atol=ATOL,
        rtol=0
    )

    print("PASS: Forward/reverse transforms are inverses")

    # Round-trip a subset of actual infrastructure LiDAR points.
    sample = infra_xyz[:min(1000, len(infra_xyz))]

    points_in_vehicle = transform_points(
        sample,
        T_vehicle_from_infra
    )

    points_back_in_infra = transform_points(
        points_in_vehicle,
        T_infra_from_vehicle
    )

    np.testing.assert_allclose(
        points_back_in_infra,
        sample.astype(np.float64),
        atol=1e-5,
        rtol=0
    )

    print("PASS: LiDAR point round-trip reconstruction")

    # Ensure relative rotation remains a proper rotation.
    rotation = T_vehicle_from_infra[:3, :3]

    np.testing.assert_allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=ATOL,
        rtol=0
    )

    np.testing.assert_allclose(
        np.linalg.det(rotation),
        1.0,
        atol=ATOL,
        rtol=0
    )

    print("PASS: Relative rotation is orthogonal, det +1")


# ============================================================
# 5. Inspect LiDAR cloud statistics
# ============================================================

def print_cloud_statistics(name, xyz):
    """
    Print spatial bounds and point count.
    """

    minimum = xyz.min(axis=0)
    maximum = xyz.max(axis=0)

    print(f"\n{name}")
    print(f"  Points: {len(xyz):,}")
    print(f"  X range: {minimum[0]:.2f} to {maximum[0]:.2f} m")
    print(f"  Y range: {minimum[1]:.2f} to {maximum[1]:.2f} m")
    print(f"  Z range: {minimum[2]:.2f} to {maximum[2]:.2f} m")


# ============================================================
# 6. Downsample for visualization only
# ============================================================

def downsample_for_plot(xyz, max_points, seed):
    """
    Randomly select points for faster visualization.

    Original point clouds remain unchanged.
    """

    if len(xyz) <= max_points:
        return xyz

    rng = np.random.default_rng(seed)

    indices = rng.choice(
        len(xyz),
        size=max_points,
        replace=False
    )

    return xyz[indices]


# ============================================================
# 7. Plot a shared-frame BEV
# ============================================================

def plot_bev(
    vehicle_xyz,
    infra_xyz_in_vehicle,
    infra_origin_in_vehicle,
    view_range,
    output_path,
    title
):
    """
    Plot vehicle and transformed infrastructure LiDAR
    in the vehicle reference frame.
    """

    vehicle_plot = downsample_for_plot(
        vehicle_xyz,
        MAX_PLOT_POINTS,
        RANDOM_SEED
    )

    infra_plot = downsample_for_plot(
        infra_xyz_in_vehicle,
        MAX_PLOT_POINTS,
        RANDOM_SEED + 1
    )

    fig, ax = plt.subplots(figsize=(11, 11))

    # Vehicle point cloud: blue.
    ax.scatter(
        vehicle_plot[:, 0],
        vehicle_plot[:, 1],
        s=0.18,
        c="royalblue",
        alpha=0.35,
        linewidths=0,
        rasterized=True,
        label="Vehicle LiDAR"
    )

    # Infrastructure point cloud: orange.
    ax.scatter(
        infra_plot[:, 0],
        infra_plot[:, 1],
        s=0.18,
        c="darkorange",
        alpha=0.35,
        linewidths=0,
        rasterized=True,
        label="Infrastructure LiDAR (transformed)"
    )

    # Vehicle reference origin.
    ax.scatter(
        0,
        0,
        c="black",
        marker="x",
        s=100,
        linewidths=2,
        label="Vehicle reference origin",
        zorder=5
    )

    # Infrastructure sensor origin expressed in vehicle frame.
    ax.scatter(
        infra_origin_in_vehicle[0],
        infra_origin_in_vehicle[1],
        c="red",
        marker="^",
        s=110,
        edgecolors="black",
        linewidths=0.7,
        label="Infrastructure sensor origin",
        zorder=5
    )

    ax.set_xlim(-view_range, view_range)
    ax.set_ylim(-view_range, view_range)

    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(
        "X in vehicle reference frame (m)"
    )

    ax.set_ylabel(
        "Y in vehicle reference frame (m)"
    )

    ax.set_title(title)

    ax.grid(
        True,
        linestyle="--",
        alpha=0.25
    )

    ax.legend(
        loc="upper right",
        markerscale=3,
        fontsize=9
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved visualization: {output_path}")


# ============================================================
# 8. Main validation workflow
# ============================================================

def main(argv=None):
    args = parser_for("v2xpnp").parse_args(argv)
    SCENARIO_DIR, FRAME, VEHICLE_ID, INFRA_ID = select_v2xpnp(args)
    OUTPUT_DIR = output_path(args.output, args.config)

    print("=" * 65)
    print("V2XPnP MULTI-AGENT LIDAR ALIGNMENT VALIDATION")
    print("=" * 65)

    print(f"\nScenario: {SCENARIO_DIR.name}")
    print(f"Frame: {FRAME}")

    if not SCENARIO_DIR.is_dir():
        raise FileNotFoundError(
            f"Scenario directory not found: {SCENARIO_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Step 1: Load metadata
    # --------------------------------------------------------

    vehicle_metadata, vehicle_pose = load_metadata(
        VEHICLE_ID, SCENARIO_DIR, FRAME
    )

    infra_metadata, infra_pose = load_metadata(
        INFRA_ID, SCENARIO_DIR, FRAME
    )

    if vehicle_metadata["infra"] is not False:
        raise ValueError(
            f"Agent {VEHICLE_ID} is not marked as a vehicle"
        )

    if infra_metadata["infra"] is not True:
        raise ValueError(
            f"Agent {INFRA_ID} is not marked as infrastructure"
        )

    print("\nAgent metadata loaded successfully")
    print(f"Vehicle pose: {vehicle_pose}")
    print(f"Infrastructure pose: {infra_pose}")

    # --------------------------------------------------------
    # Step 2: Load both LiDAR point clouds
    # --------------------------------------------------------

    vehicle_xyz, vehicle_intensity = load_lidar(
        VEHICLE_ID, SCENARIO_DIR, FRAME
    )

    infra_xyz, infra_intensity = load_lidar(
        INFRA_ID, SCENARIO_DIR, FRAME
    )

    print("\nLiDAR point clouds loaded successfully")

    print_cloud_statistics(
        "Vehicle LiDAR (local frame)",
        vehicle_xyz
    )

    print_cloud_statistics(
        "Infrastructure LiDAR (local frame)",
        infra_xyz
    )

    print(
        f"\nVehicle intensity range: "
        f"{vehicle_intensity.min():.1f} - "
        f"{vehicle_intensity.max():.1f}"
    )

    print(
        f"Infrastructure intensity range: "
        f"{infra_intensity.min():.1f} - "
        f"{infra_intensity.max():.1f}"
    )

    # --------------------------------------------------------
    # Step 3: Compute OpenCOOD relative transformation
    # --------------------------------------------------------

    T_vehicle_from_infra = x1_to_x2(
        infra_pose,
        vehicle_pose
    )

    print("\nT_vehicle_from_infra:")
    print(
        np.array2string(
            T_vehicle_from_infra,
            precision=6,
            suppress_small=True
        )
    )

    # --------------------------------------------------------
    # Step 4: Mathematical transformation checks
    # --------------------------------------------------------

    print("\nRunning transformation checks...")

    validate_transformations(
        vehicle_pose,
        infra_pose,
        infra_xyz,
        T_vehicle_from_infra
    )

    # --------------------------------------------------------
    # Step 5: Transform infrastructure points
    # --------------------------------------------------------

    infra_xyz_in_vehicle = transform_points(
        infra_xyz,
        T_vehicle_from_infra
    )

    # Vehicle points are already in the reference frame.
    vehicle_xyz_in_vehicle = vehicle_xyz

    print(
        "\nInfrastructure LiDAR transformed "
        "into vehicle reference frame"
    )

    print_cloud_statistics(
        "Infrastructure LiDAR (vehicle reference frame)",
        infra_xyz_in_vehicle
    )

    # --------------------------------------------------------
    # Step 6: Compute sensor origin in reference frame
    # --------------------------------------------------------

    infra_origin_in_vehicle = transform_points(
        np.array([[0.0, 0.0, 0.0]]),
        T_vehicle_from_infra
    )[0]

    print(
        "\nInfrastructure origin in vehicle frame:"
    )

    print(
        np.array2string(
            infra_origin_in_vehicle,
            precision=3
        )
    )

    # --------------------------------------------------------
    # Step 7: Generate overview visualization
    # --------------------------------------------------------

    overview_path = (
        OUTPUT_DIR
        / "v2xpnp_all_agents_reference_frame.png"
    )

    plot_bev(
        vehicle_xyz=vehicle_xyz_in_vehicle,
        infra_xyz_in_vehicle=infra_xyz_in_vehicle,
        infra_origin_in_vehicle=infra_origin_in_vehicle,
        view_range=OVERVIEW_RANGE_M,
        output_path=overview_path,
        title=(
            "V2XPnP Multi-Agent LiDAR — Vehicle Reference Frame\n"
            f"Scenario: {SCENARIO_DIR.name} | Frame: {FRAME}"
        )
    )

    # --------------------------------------------------------
    # Step 8: Generate close-up visualization
    # --------------------------------------------------------

    zoom_path = (
        OUTPUT_DIR
        / "v2xpnp_reference_lidar_zoom.png"
    )

    plot_bev(
        vehicle_xyz=vehicle_xyz_in_vehicle,
        infra_xyz_in_vehicle=infra_xyz_in_vehicle,
        infra_origin_in_vehicle=infra_origin_in_vehicle,
        view_range=ZOOM_RANGE_M,
        output_path=zoom_path,
        title=(
            "V2XPnP Multi-Agent LiDAR — Reference Vehicle Detail\n"
            f"Frame: {FRAME} | Range: +/-{ZOOM_RANGE_M} m"
        )
    )

    # --------------------------------------------------------
    # Step 9: Summary
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("V2XPnP LIDAR ALIGNMENT TEST COMPLETED")
    print("=" * 65)

    print(
        f"Vehicle points: {len(vehicle_xyz_in_vehicle):,}"
    )

    print(
        f"Infrastructure points: {len(infra_xyz_in_vehicle):,}"
    )

    print(
        f"Total points: "
        f"{len(vehicle_xyz_in_vehicle) + len(infra_xyz_in_vehicle):,}"
    )

    print("\nMathematical transformation checks: PASSED")

    print("\nGenerated files:")
    print(f"  {overview_path}")
    print(f"  {zoom_path}")

    print(
        "\nIMPORTANT: Visual inspection is still required "
        "to assess physical point-cloud alignment."
    )

    print(
        "Coordinates are OpenCOOD-native; "
        "canonical handedness has not yet been validated."
    )


def test_validation(cli_args):
    main(cli_args("v2xpnp"))


if __name__ == "__main__":
    main()