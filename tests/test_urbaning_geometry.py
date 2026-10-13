from tests.cli import parser_for, urbaning_adapter
from project_paths import output_path

import matplotlib.pyplot as plt
import numpy as np

from adapters.urbaning_adapter import UrbanIngAdapter


# ============================================================
# Configuration
# ============================================================

# Limit the visualization around the reference vehicle.
REFERENCE_VIEW_RADIUS_M = 60.0
COOPERATIVE_VIEW_RADIUS_M = 200.0

# Plotting every LiDAR point makes matplotlib slow.
MAX_POINTS_PER_AGENT = 30000


# ============================================================
# Helper: transform LiDAR points into reference-agent frame
# ============================================================

def transform_points(
    xyz: np.ndarray,
    T_ref_from_agent: np.ndarray,
) -> np.ndarray:
    """
    Transform points from the agent coordinate system into
    the reference-agent coordinate system.

    p_ref = T_ref_from_agent @ p_agent
    """

    rotation = T_ref_from_agent[:3, :3]
    translation = T_ref_from_agent[:3, 3]

    xyz_ref = (rotation @ xyz.T).T + translation

    return xyz_ref


# ============================================================
# Helper: deterministic point subsampling
# ============================================================

def subsample_points(
    xyz: np.ndarray,
    max_points: int,
) -> np.ndarray:

    if len(xyz) <= max_points:
        return xyz

    indices = np.linspace(
        0,
        len(xyz) - 1,
        max_points,
        dtype=int,
    )

    return xyz[indices]


# ============================================================
# Helper: draw one canonical 3D box from above
# ============================================================

def draw_box_bev(
    ax,
    box: np.ndarray,
):
    """
    Draw a canonical box in bird's-eye view.

    Canonical format:

        [x, y, z, length, width, height, yaw]

    Only x, y, length, width and yaw are required for
    the top-down visualization.
    """

    x, y, _, length, width, _, yaw = box

    # Box corners before rotation.
    corners = np.array(
        [
            [length / 2.0, width / 2.0],
            [length / 2.0, -width / 2.0],
            [-length / 2.0, -width / 2.0],
            [-length / 2.0, width / 2.0],
        ],
        dtype=np.float64,
    )

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    rotation = np.array(
        [
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw],
        ]
    )

    corners = (rotation @ corners.T).T

    corners[:, 0] += x
    corners[:, 1] += y

    # Close the rectangle.
    corners = np.vstack([corners, corners[0]])

    ax.plot(
        corners[:, 0],
        corners[:, 1],
        linewidth=1.0,
    )

    # Draw direction of the box's positive x-axis.
    heading_x = x + np.cos(yaw) * length / 2.0
    heading_y = y + np.sin(yaw) * length / 2.0

    ax.plot(
        [x, heading_x],
        [y, heading_y],
        linewidth=1.0,
    )


# ============================================================
# Helper: check whether transform looks like a rigid transform
# ============================================================

def check_transform(
    agent_id: str,
    T: np.ndarray,
):

    R = T[:3, :3]

    orthogonality_error = np.linalg.norm(
        R.T @ R - np.eye(3)
    )

    determinant = np.linalg.det(R)

    translation = T[:3, 3]

    print(f"\nAgent: {agent_id}")
    print(f"  Rotation determinant     : {determinant:.6f}")
    print(f"  Orthogonality error      : {orthogonality_error:.6e}")
    print(f"  Translation [m]          : {translation}")

    if not np.allclose(
        T[3],
        np.array([0.0, 0.0, 0.0, 1.0]),
        atol=1e-6,
    ):
        print("  WARNING: invalid final transform row")

    if not np.isclose(
        determinant,
        1.0,
        atol=1e-3,
    ):
        print("  WARNING: rotation determinant is not close to 1")

    if orthogonality_error > 1e-3:
        print("  WARNING: rotation matrix is not orthogonal")


# ============================================================
# Load canonical UrbanIng sample
# ============================================================

def main(argv=None):
    args = parser_for("urbaning").parse_args(argv)
    adapter = urbaning_adapter(args)
    OUTPUT_DIR = output_path(args.output, args.config)
    FRAME_INDEX = args.index
    print("=" * 70)
    print("URBANING GEOMETRIC VALIDATION")
    print("=" * 70)

    # Adapter already configured above.

    sample = adapter[FRAME_INDEX]

    sample.validate()

    print(f"Sequence        : {sample.source.sequence_id}")
    print(f"Frame           : {sample.source.frame_id}")
    print(f"Timestamp [s]   : {sample.timestamp_s}")
    print(f"Reference agent : {sample.reference_agent_id}")
    print(f"Agents          : {len(sample.agents)}")
    print(f"3D boxes        : {len(sample.annotations.boxes_3d)}")


    # ============================================================
    # Test 1: numerical transform sanity
    # ============================================================

    print("\n" + "=" * 70)
    print("TEST 1: RIGID TRANSFORM CHECK")
    print("=" * 70)

    for agent in sample.agents:
        check_transform(
            agent.agent_id,
            agent.T_ref_from_agent,
        )


    # ============================================================
    # Test 2: box sanity
    # ============================================================

    print("\n" + "=" * 70)
    print("TEST 2: 3D BOX SANITY")
    print("=" * 70)

    boxes = sample.annotations.boxes_3d

    print("Boxes shape:", boxes.shape)

    if len(boxes) > 0:

        dimensions = boxes[:, 3:6]

        print(
            "Minimum box dimensions [l,w,h]:",
            dimensions.min(axis=0),
        )

        print(
            "Maximum box dimensions [l,w,h]:",
            dimensions.max(axis=0),
        )

        if np.any(dimensions <= 0):
            print("WARNING: one or more boxes have non-positive dimensions")
        else:
            print("All box dimensions are positive.")

        if np.all(np.isfinite(boxes)):
            print("All box values are finite.")
        else:
            print("WARNING: NaN or infinity found in boxes.")


    # ============================================================
    # Test 3: reference LiDAR + boxes
    # ============================================================

    print("\n" + "=" * 70)
    print("TEST 3: REFERENCE LIDAR + BOXES")
    print("=" * 70)

    reference_agent = next(
        agent
        for agent in sample.agents
        if agent.is_reference
    )

    if reference_agent.lidar is None:
        raise RuntimeError(
            "Reference agent does not contain LiDAR data."
        )

    reference_xyz = reference_agent.lidar.xyz

    print(
        "Reference LiDAR points:",
        reference_xyz.shape,
    )

    reference_xyz = subsample_points(
        reference_xyz,
        MAX_POINTS_PER_AGENT,
    )

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    ax.scatter(
        reference_xyz[:, 0],
        reference_xyz[:, 1],
        s=0.5,
        alpha=0.5,
        label=reference_agent.agent_id,
    )

    for box in boxes:
        draw_box_bev(ax, box)

    ax.set_title(
        "UrbanIng: Reference LiDAR + Canonical 3D Boxes"
    )

    ax.set_xlabel("x [m] — forward")
    ax.set_ylabel("y [m] — left")

    ax.set_xlim(
        -REFERENCE_VIEW_RADIUS_M,
        REFERENCE_VIEW_RADIUS_M,
    )

    ax.set_ylim(
        -REFERENCE_VIEW_RADIUS_M,
        REFERENCE_VIEW_RADIUS_M,
    )

    ax.set_aspect("equal")

    ax.legend()

    ax.grid(True)

    reference_output = (
        OUTPUT_DIR
        / "urbaning_reference_lidar_boxes.png"
    )

    plt.savefig(
        reference_output,
        dpi=200,
        bbox_inches="tight",
    )

    print(
        f"Saved reference geometry plot:\n"
        f"{reference_output}"
    )

    plt.show() if args.show else plt.close(fig)


    # ============================================================
    # Test 4: transform ALL agents into reference frame
    # ============================================================

    print("\n" + "=" * 70)
    print("TEST 4: ALL AGENTS IN REFERENCE FRAME")
    print("=" * 70)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    for agent in sample.agents:

        if agent.lidar is None:
            print(
                f"{agent.agent_id}: no LiDAR, skipped"
            )
            continue

        xyz = agent.lidar.xyz

        xyz_ref = transform_points(
            xyz,
            agent.T_ref_from_agent,
        )

        print(f"\n{agent.agent_id}")
        print(
            "  Local point range x:",
            xyz[:, 0].min(),
            "to",
            xyz[:, 0].max(),
        )

        print(
            "  Local point range y:",
            xyz[:, 1].min(),
            "to",
            xyz[:, 1].max(),
        )

        print(
            "  Reference-frame range x:",
            xyz_ref[:, 0].min(),
            "to",
            xyz_ref[:, 0].max(),
        )

        print(
            "  Reference-frame range y:",
            xyz_ref[:, 1].min(),
            "to",
            xyz_ref[:, 1].max(),
        )

        xyz_ref = subsample_points(
            xyz_ref,
            MAX_POINTS_PER_AGENT,
        )

        ax.scatter(
            xyz_ref[:, 0],
            xyz_ref[:, 1],
            s=0.5,
            alpha=0.35,
            label=agent.agent_id,
        )


    # Draw annotations after all LiDAR clouds.
    for box in boxes:
        draw_box_bev(ax, box)


    # Reference-agent origin.
    ax.scatter(
        [0],
        [0],
        s=80,
        marker="x",
        label="reference origin",
    )

    ax.set_title(
        "UrbanIng: All Agent LiDARs Transformed to Reference Frame"
    )

    ax.set_xlabel("x [m] — forward")
    ax.set_ylabel("y [m] — left")

    ax.set_xlim(
        -COOPERATIVE_VIEW_RADIUS_M,
        COOPERATIVE_VIEW_RADIUS_M,
    )

    ax.set_ylim(
        -COOPERATIVE_VIEW_RADIUS_M,
        COOPERATIVE_VIEW_RADIUS_M,
    )

    ax.set_aspect("equal")

    ax.legend()

    ax.grid(True)

    all_agents_output = (
        OUTPUT_DIR
        / "urbaning_all_agents_reference_frame.png"
    )

    plt.savefig(
        all_agents_output,
        dpi=200,
        bbox_inches="tight",
    )

    print(
        f"\nSaved cooperative geometry plot:\n"
        f"{all_agents_output}"
    )

    plt.show() if args.show else plt.close(fig)


    print("\n" + "=" * 70)
    print("GEOMETRIC TEST FINISHED")
    print("=" * 70)

    print(
        "\nInspect the two generated images manually."
    )


def test_validation(cli_args):
    main(cli_args("urbaning"))


if __name__ == "__main__":
    main()
