
from tests.cli import parser_for, urbaning_adapter
import traceback

import numpy as np

from adapters.urbaning_adapter import UrbanIngAdapter


# ============================================================
# Configuration
# ============================================================

# ============================================================
# Helper: Validate one canonical sample
# ============================================================

def validate_sample(sample):
    """
    Check structural and numerical consistency of a
    CanonicalV2XSample.

    Returns a summary dictionary for reporting.
    """

    # Validate against our existing canonical data contract.
    sample.validate()

    assert np.isfinite(sample.timestamp_s), (
        "Timestamp is not finite."
    )

    assert sample.timestamp_s >= 0, (
        "Timestamp is negative."
    )

    assert len(sample.agents) > 0, (
        "Sample contains no agents."
    )

    assert len({
        agent.agent_id for agent in sample.agents
    }) == len(sample.agents), (
        "Duplicate agent IDs."
    )

    reference_agents = [
        agent for agent in sample.agents
        if agent.is_reference
    ]

    assert len(reference_agents) == 1, (
        "Expected exactly one reference agent."
    )

    reference = reference_agents[0]

    assert reference.agent_id == sample.reference_agent_id, (
        "Reference agent ID mismatch."
    )

    assert np.allclose(
        reference.T_ref_from_agent,
        np.eye(4),
        atol=1e-4,
    ), "Reference transform must be identity."

    # --------------------------------------------------------
    # Check all agents
    # --------------------------------------------------------

    total_lidar_points = 0
    total_cameras = 0

    for agent in sample.agents:

        T = agent.T_ref_from_agent

        assert np.all(np.isfinite(T)), (
            f"{agent.agent_id}: non-finite transform."
        )

        assert np.allclose(
            T[3, :],
            [0, 0, 0, 1],
            atol=1e-4,
        ), f"{agent.agent_id}: invalid homogeneous transform."

        R = T[:3, :3]

        assert np.allclose(
            R.T @ R,
            np.eye(3),
            atol=1e-3,
        ), f"{agent.agent_id}: non-orthogonal rotation."

        assert np.isclose(
            np.linalg.det(R),
            1.0,
            atol=1e-3,
        ), f"{agent.agent_id}: invalid rotation determinant."

        # LiDAR is optional in the canonical schema.
        if agent.lidar is not None:

            xyz = agent.lidar.xyz

            assert np.all(np.isfinite(xyz)), (
                f"{agent.agent_id}: non-finite LiDAR coordinates."
            )

            total_lidar_points += len(xyz)

            if agent.lidar.intensity is not None:
                assert np.all(np.isfinite(
                    agent.lidar.intensity
                )), f"{agent.agent_id}: non-finite intensity."

            if agent.lidar.time_offset_s is not None:
                assert np.all(np.isfinite(
                    agent.lidar.time_offset_s
                )), f"{agent.agent_id}: non-finite time offsets."

        # Cameras
        camera_ids = [cam.camera_id for cam in agent.cameras]

        assert len(camera_ids) == len(set(camera_ids)), (
            f"{agent.agent_id}: duplicate camera IDs."
        )

        for camera in agent.cameras:

            assert camera.image.size > 0, (
                f"{agent.agent_id}/{camera.camera_id}: empty image."
            )

            assert np.all(np.isfinite(camera.K)), (
                f"{agent.agent_id}/{camera.camera_id}: invalid intrinsics."
            )

            assert np.all(np.isfinite(
                camera.T_agent_from_camera
            )), (
                f"{agent.agent_id}/{camera.camera_id}: "
                "invalid camera transform."
            )

            assert np.isfinite(camera.time_offset_s), (
                f"{agent.agent_id}/{camera.camera_id}: "
                "invalid camera time offset."
            )

            total_cameras += 1

    # --------------------------------------------------------
    # Check annotations
    # --------------------------------------------------------

    annotations = sample.annotations
    boxes = annotations.boxes_3d

    assert np.all(np.isfinite(boxes)), (
        "Non-finite bounding box coordinates."
    )

    if len(boxes) > 0:
        assert np.all(boxes[:, 3:6] > 0), (
            "Bounding box dimensions must be positive."
        )

    assert len(annotations.native_labels) == len(boxes), (
        "Annotation label count mismatch."
    )

    assert len(annotations.track_ids) == len(boxes), (
        "Annotation track ID count mismatch."
    )

    return {
        "sequence": sample.source.sequence_id,
        "frame_id": sample.source.frame_id,
        "timestamp": float(sample.timestamp_s),
        "agents": len(sample.agents),
        "lidar_points": total_lidar_points,
        "cameras": total_cameras,
        "boxes": len(boxes),
    }


# ============================================================
# Main test
# ============================================================

def main(argv=None):
    args = parser_for("urbaning").parse_args(argv)

    print("=" * 85)
    print("URBANING MULTI-FRAME REGRESSION TEST")
    print("=" * 85)

    adapter = urbaning_adapter(args)

    print(f"Total samples available: {len(adapter)}")

    # Spread default checks across each sequence; never compare relative
    # timestamps across sequence boundaries.
    if args.indices is not None:
        frame_indices = sorted(set(args.indices))
        if any(index < 0 or index >= len(adapter) for index in frame_indices):
            raise IndexError("Requested frame index is outside the adapter")
    else:
        frame_indices = []
        for sequence in adapter.sequence_names:
            indices = [i for i, (name, _) in enumerate(adapter._samples) if name == sequence]
            if indices:
                frame_indices.extend(np.linspace(indices[0], indices[-1],
                                                 min(9, len(indices)), dtype=int).tolist())

    if not frame_indices:
        raise RuntimeError("No valid frame indices selected.")

    passed = 0
    failed = 0
    results = []

    print("\nRunning frame validation...\n")

    for index in frame_indices:

        try:
            sample = adapter[index]

            summary = validate_sample(sample)

            summary["index"] = index
            results.append(summary)

            passed += 1

            print(
                f"[PASS] Frame {index:3d} | "
                f"t={summary['timestamp']:8.3f}s | "
                f"agents={summary['agents']} | "
                f"points={summary['lidar_points']:,} | "
                f"cameras={summary['cameras']} | "
                f"boxes={summary['boxes']}"
            )

        except Exception as error:

            failed += 1

            print(f"[FAIL] Frame {index}: {error}")
            traceback.print_exc()

    # --------------------------------------------------------
    # Check chronological timestamps
    # --------------------------------------------------------

    print("\n" + "-" * 85)
    print("TEMPORAL CONSISTENCY")
    print("-" * 85)

    temporal_passed = False

    if len(results) >= 2:

        results.sort(key=lambda r: r["index"])

        timestamps = np.array([
            row["timestamp"] for row in results
        ])

        differences = np.asarray([
            right["timestamp"] - left["timestamp"]
            for left, right in zip(results, results[1:])
            if left["sequence"] == right["sequence"]
        ])

        if np.all(differences > 0):
            temporal_passed = True
            print("PASS: Timestamps are strictly increasing.")
        else:
            print("FAIL: Timestamps are not strictly increasing.")
            print("Timestamp differences:", differences)

        print(
            f"First tested timestamp: {timestamps[0]:.3f}s"
        )
        print(
            f"Last tested timestamp : {timestamps[-1]:.3f}s"
        )

    else:
        print("SKIPPED: Need at least two successful frames.")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 85)
    print("TEST SUMMARY")
    print("=" * 85)

    print(f"Frames selected : {len(frame_indices)}")
    print(f"Frames passed   : {passed}")
    print(f"Frames failed   : {failed}")
    print(f"Temporal check  : {'PASS' if temporal_passed else 'FAIL/SKIPPED'}")

    all_passed = (
        failed == 0
        and passed == len(frame_indices)
        and (temporal_passed or len(results) == 1)
    )

    if all_passed:
        print("\nRESULT: ALL MULTI-FRAME TESTS PASSED")
    else:
        print("\nRESULT: MULTI-FRAME TEST INCOMPLETE OR FAILED")
        raise SystemExit(1)


def test_validation(cli_args):
    main(cli_args("urbaning"))


if __name__ == "__main__":
    main()
