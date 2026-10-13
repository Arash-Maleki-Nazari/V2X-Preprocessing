"""Overlap-focused BEV validation for V2XPnP vehicle/infrastructure LiDAR.

Run from the Preprocessing project root:
    python -m tests.test_v2xpnp_overlap_alignment

Requires: numpy, matplotlib, PyYAML (for test_v2xpnp_geometry.py).

Reads scenario/frame/agent IDs from CLI or native metadata; reuses the OpenCOOD transform routine
from the previously validated test_v2xpnp_geometry.py module.

Creates three plots in the configured output directory:
    v2xpnp_overlap_midpoint_bev.png
    v2xpnp_overlap_closeup_bev.png
    v2xpnp_overlap_occupancy.png

NOTE: Shared 2D occupied cells are an exploratory diagnostic, NOT a geometric
registration score or a substitute for checking common 3D structures.
The data remain in OpenCOOD's native coordinate convention.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save images even when running without a GUI.
import matplotlib.pyplot as plt
import numpy as np

from tests.test_v2xpnp_geometry import load_lidar_pose, x1_to_x2
from tests.cli import parser_for, select_v2xpnp
from project_paths import output_path

# ------------------------------- Configuration -------------------------------
MIDPOINT_HALF_WIDTH_M = 90.0
DETAIL_HALF_WIDTH_M = 40.0
GRID_SIZE_M = 1.0
MIN_POINTS_PER_CELL = 3
MAX_PLOT_POINTS_PER_AGENT = 130_000
RANDOM_SEED = 42
# No height filtering by default: preserve all measured geometry.
# Optionally set e.g. (-3.0, 3.0) to limit plotted points by reference-frame z.
Z_FILTER_M = None


# ------------------------------- Data and poses -------------------------------
def load_xyz(agent_id: str, scenario_dir: Path, frame: str) -> np.ndarray:
    path = scenario_dir / agent_id / f"{frame}.bin"
    if not path.is_file():
        raise FileNotFoundError(f"Missing LiDAR file: {path}")
    data = np.fromfile(path, dtype="<f4")
    if data.size == 0 or data.size % 4:
        raise ValueError(f"Expected nonempty float32 Nx4 LiDAR binary: {path}")
    points = data.reshape(-1, 4)
    if not np.isfinite(points).all():
        raise ValueError(f"Non-finite LiDAR values: {path}")
    return points[:, :3].astype(np.float64, copy=False)


def apply_transform(xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Row-vector form of p_target = R p_source + t."""
    if transform.shape != (4, 4):
        raise ValueError("Expected a 4x4 homogeneous transform")
    return xyz @ transform[:3, :3].T + transform[:3, 3]


def xy_crop(xyz: np.ndarray, center: np.ndarray, half_width: float) -> np.ndarray:
    xy = xyz[:, :2]
    mask = (
        (np.abs(xy[:, 0] - center[0]) <= half_width)
        & (np.abs(xy[:, 1] - center[1]) <= half_width)
    )
    if Z_FILTER_M is not None:
        mask &= (xyz[:, 2] >= Z_FILTER_M[0]) & (xyz[:, 2] <= Z_FILTER_M[1])
    return xyz[mask]


def plot_sample(xyz: np.ndarray, seed: int) -> np.ndarray:
    if len(xyz) <= MAX_PLOT_POINTS_PER_AGENT:
        return xyz
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(xyz), MAX_PLOT_POINTS_PER_AGENT, replace=False)
    return xyz[indices]


# ------------------------------- Occupancy analysis --------------------------
def occupied_cells(
    points: np.ndarray, origin: np.ndarray, cell_size: float, min_count: int
) -> set[tuple[int, int]]:
    """2D grid cells having >= min_count LiDAR returns (BEV only)."""
    if len(points) == 0:
        return set()
    ij = np.floor((points[:, :2] - origin) / cell_size).astype(np.int64)
    cells, counts = np.unique(ij, axis=0, return_counts=True)
    return {tuple(cell) for cell in cells[counts >= min_count]}


def compute_common_cells(
    vehicle_points: np.ndarray,
    infra_points: np.ndarray,
    center: np.ndarray,
    half_width: float,
) -> tuple[set, set, set, np.ndarray]:
    """Compare grid-cell occupancy over the midpoint-focused region."""
    origin = center - half_width
    v_region = xy_crop(vehicle_points, center, half_width)
    i_region = xy_crop(infra_points, center, half_width)
    v_cells = occupied_cells(v_region, origin, GRID_SIZE_M, MIN_POINTS_PER_CELL)
    i_cells = occupied_cells(i_region, origin, GRID_SIZE_M, MIN_POINTS_PER_CELL)
    common = v_cells & i_cells
    return v_cells, i_cells, common, origin


def common_cells_center(common: set, origin: np.ndarray) -> np.ndarray | None:
    if len(common) < 15:
        return None
    ij = np.asarray(sorted(common), dtype=np.float64)
    xy = origin + (ij + 0.5) * GRID_SIZE_M
    # Median is less affected by a few coincident outlier cells.
    return np.median(xy, axis=0)


# ------------------------------- Visualization -------------------------------
def save_overlay(
    vehicle: np.ndarray,
    infra: np.ndarray,
    vehicle_origin: np.ndarray,
    infra_origin: np.ndarray,
    center: np.ndarray,
    half_width: float,
    title: str,
    output: Path,
) -> None:
    v_roi = xy_crop(vehicle, center, half_width)
    i_roi = xy_crop(infra, center, half_width)
    v_plot = plot_sample(v_roi, RANDOM_SEED)
    i_plot = plot_sample(i_roi, RANDOM_SEED + 1)

    fig, ax = plt.subplots(figsize=(11, 10))
    if len(v_plot):
        ax.scatter(
            v_plot[:, 0], v_plot[:, 1], c="#2358bc", s=0.55,
            alpha=0.45, linewidths=0, rasterized=True,
            label=f"Vehicle ({len(v_roi):,} points in view)",
        )
    if len(i_plot):
        ax.scatter(
            i_plot[:, 0], i_plot[:, 1], c="#ed7800", s=0.7,
            alpha=0.60, linewidths=0, rasterized=True,
            label=f"Infrastructure ({len(i_roi):,} points in view)",
        )

    for origin, label, color, marker in (
        (vehicle_origin, "Vehicle sensor origin", "black", "x"),
        (infra_origin, "Infrastructure sensor origin", "red", "^"),
    ):
        if np.all(np.abs(origin[:2] - center) <= half_width):
            ax.scatter(
                origin[0], origin[1], c=color, marker=marker, s=100,
                linewidths=2 if marker == "x" else 0.5,
                edgecolors="black" if marker == "^" else None,
                zorder=7, label=label,
            )

    ax.set_xlim(center[0] - half_width, center[0] + half_width)
    ax.set_ylim(center[1] - half_width, center[1] + half_width)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X in vehicle LiDAR frame (m; OpenCOOD convention)")
    ax.set_ylabel("Y in vehicle LiDAR frame (m; OpenCOOD convention)")
    ax.set_title(title)
    ax.grid(True, ls="--", alpha=0.23)
    ax.legend(loc="upper right", markerscale=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=210, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def save_occupancy_plot(
    v_cells: set,
    i_cells: set,
    common: set,
    origin: np.ndarray,
    center: np.ndarray,
    half_width: float,
    output: Path,
    frame: str,
) -> None:
    """Visualize exclusively vehicle, exclusively infra and jointly occupied cells."""
    fig, ax = plt.subplots(figsize=(11, 10))
    groups = (
        (v_cells - common, "Vehicle-only occupied cells", "#2358bc"),
        (i_cells - common, "Infrastructure-only occupied cells", "#ed7800"),
        (common, "Cells occupied by both (2D)", "#8a2387"),
    )
    for cells, label, color in groups:
        if not cells:
            continue
        xy = origin + (np.asarray(sorted(cells), dtype=float) + 0.5) * GRID_SIZE_M
        ax.scatter(
            xy[:, 0], xy[:, 1], c=color, marker="s", s=13,
            alpha=0.8, linewidths=0, rasterized=True, label=f"{label}: {len(cells):,}",
        )
    ax.set_xlim(center[0] - half_width, center[0] + half_width)
    ax.set_ylim(center[1] - half_width, center[1] + half_width)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X in vehicle LiDAR frame (m; OpenCOOD convention)")
    ax.set_ylabel("Y in vehicle LiDAR frame (m; OpenCOOD convention)")
    ax.set_title(
        f"BEV occupied-cell overlap — frame {frame}\n"
        f"Grid {GRID_SIZE_M:g} m; minimum {MIN_POINTS_PER_CELL} points per cell"
    )
    ax.grid(True, ls="--", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=210, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


# ------------------------------- Execution -----------------------------------
def main(argv=None) -> None:
    args = parser_for("v2xpnp").parse_args(argv)
    SCENARIO_DIR, FRAME, VEHICLE_ID, INFRA_ID = select_v2xpnp(args)
    OUTPUT_DIR = output_path(args.output, args.config)
    print("=" * 65)
    print("V2XPnP OVERLAP-FOCUSED LIDAR VALIDATION")
    print("=" * 65)
    print(f"Scenario: {SCENARIO_DIR.name} | Frame: {FRAME}")
    if not SCENARIO_DIR.is_dir():
        raise FileNotFoundError(f"Scenario directory not found: {SCENARIO_DIR}")

    # Reuse metadata checking and the exact tested OpenCOOD transformation.
    v_pose = load_lidar_pose(VEHICLE_ID, SCENARIO_DIR, FRAME, expected_infra=False)
    i_pose = load_lidar_pose(INFRA_ID, SCENARIO_DIR, FRAME, expected_infra=True)
    transform = x1_to_x2(i_pose, v_pose)
    np.testing.assert_allclose(transform[:3, :3].T @ transform[:3, :3],
                               np.eye(3), atol=1e-8, rtol=0)
    np.testing.assert_allclose(np.linalg.det(transform[:3, :3]),
                               1.0, atol=1e-8, rtol=0)

    vehicle_xyz = load_xyz(VEHICLE_ID, SCENARIO_DIR, FRAME)
    infra_xyz = load_xyz(INFRA_ID, SCENARIO_DIR, FRAME)
    infra_in_vehicle = apply_transform(infra_xyz, transform)
    vehicle_origin = np.zeros(3, dtype=np.float64)
    infra_origin = transform[:3, 3]
    midpoint = 0.5 * (vehicle_origin[:2] + infra_origin[:2])

    print(f"Vehicle points: {len(vehicle_xyz):,}")
    print(f"Infrastructure points: {len(infra_xyz):,}")
    print(f"Infrastructure origin in vehicle frame: {np.round(infra_origin, 3)} m")
    print(f"Sensor midpoint XY: {np.round(midpoint, 3)} m")
    print("PASS: OpenCOOD relative rotation orthogonal; determinant +1")

    v_cells, i_cells, common, grid_origin = compute_common_cells(
        vehicle_xyz, infra_in_vehicle, midpoint, MIDPOINT_HALF_WIDTH_M
    )
    union = v_cells | i_cells
    occupancy_iou = len(common) / len(union) if union else 0.0
    print(f"Midpoint ROI: +/-{MIDPOINT_HALF_WIDTH_M:g} m around {np.round(midpoint, 2)}")
    print(f"Occupied grid cells: vehicle={len(v_cells):,}, infra={len(i_cells):,}")
    print(f"Both-occupied 2D cells: {len(common):,}")
    print(f"Exploratory 2D occupancy IoU: {occupancy_iou:.4f}")
    print("NOTE: 2D occupancy overlap alone does NOT prove LiDAR registration.")

    detail_center = common_cells_center(common, grid_origin)
    if detail_center is None:
        detail_center = midpoint
        print("Few common occupied cells: close-up uses sensor midpoint.")
    else:
        print(f"Close-up centered on jointly occupied region: {np.round(detail_center, 2)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_overlay(
        vehicle_xyz, infra_in_vehicle, vehicle_origin, infra_origin,
        midpoint, MIDPOINT_HALF_WIDTH_M,
        f"V2XPnP LiDAR overlap — sensor midpoint BEV\nFrame {FRAME}, +/-{MIDPOINT_HALF_WIDTH_M:g} m",
        OUTPUT_DIR / "v2xpnp_overlap_midpoint_bev.png",
    )
    save_overlay(
        vehicle_xyz, infra_in_vehicle, vehicle_origin, infra_origin,
        detail_center, DETAIL_HALF_WIDTH_M,
        f"V2XPnP LiDAR overlap — shared-region close-up\nFrame {FRAME}, +/-{DETAIL_HALF_WIDTH_M:g} m",
        OUTPUT_DIR / "v2xpnp_overlap_closeup_bev.png",
    )
    save_occupancy_plot(
        v_cells, i_cells, common, grid_origin, midpoint,
        MIDPOINT_HALF_WIDTH_M, OUTPUT_DIR / "v2xpnp_overlap_occupancy.png", FRAME,
    )

    print(f"\nFINISHED: Three images saved to {OUTPUT_DIR}.")
    print("Inspect matching stationary edges and surfaces in the overlays.")
    print("Physical alignment and canonical handedness are NOT automatically validated.")


def test_validation(cli_args):
    main(cli_args("v2xpnp"))


if __name__ == "__main__":
    main()
