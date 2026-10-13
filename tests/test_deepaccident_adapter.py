"""Synthetic, dataset-free tests for the provisional DeepAccident adapter.

The tests intentionally assert *metadata preservation*, not unverified
camera, LiDAR, or reference-frame annotation conversions. Only locally
created and trusted pickle files are loaded.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from adapters.deepaccident_adapter import DeepAccidentAdapter
from schema import CanonicalV2XSample


CAMERA_NAME = "Camera_Front"
SCENARIO_TYPE = "accident"


def transform(x=0.0, y=0.0, z=0.0, yaw=0.0):
    """Homogeneous transform with proper yaw rotation and XYZ translation."""
    c, s = np.cos(yaw), np.sin(yaw)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    result[:3, 3] = [x, y, z]
    return result


def make_info(
    *, scene="Town01_case", frame=10, agent="ego_vehicle", pose=None,
    boxes=None, names=None, image_path=None,
):
    """Small representative record matching official converted-info keys."""
    if boxes is None:
        boxes = [[2.0, -3.0, 0.5, 4.0, 1.8, 1.5, 0.2]]
    if names is None:
        names = ["car"] * len(boxes)
    count = len(boxes)
    return {
        "scene_name": scene,
        "vehicle_name": agent,
        "timestamp": frame,  # Native frame index, NOT seconds.
        "scenario_type": SCENARIO_TYPE,
        "scenario_length": 80,
        "ego_to_world_matrix": transform() if pose is None else pose,
        "lidar_to_ego_matrix": transform(z=1.5),
        "lidar_prefix": "lidar01",
        "lidar_path": f"/old/machine/{SCENARIO_TYPE}/{agent}/lidar01/seq/{frame:06d}.npz",
        "bev_path": f"/old/machine/{SCENARIO_TYPE}/{agent}/BEV_instance_camera/seq/{frame:06d}.npz",
        "cams": {
            CAMERA_NAME: {
                "image_path": image_path or (
                    f"/old/machine/{SCENARIO_TYPE}/{agent}/{CAMERA_NAME}/seq/{frame:06d}.jpg"
                ),
                "camera_intrinsic_matrix": np.array(
                    [[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1.0]]
                ),
                "lidar_to_camera_matrix": transform(x=0.2),
                "timestamp": frame,
            }
        },
        "gt_boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 7),
        "gt_names": names,
        "gt_velocity": np.tile([[1.5, -0.2]], (count, 1)),
        "vehicle_id": np.arange(100, 100 + count),
        "num_lidar_pts": np.full(count, 12),
        "camera_visibility": np.ones(count),
    }


def write_infos(root: Path, infos: list[dict], split="train_mini") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"deepaccident_infos_{split}.pkl"
    with path.open("wb") as handle:
        pickle.dump({"infos": infos, "metadata": {"version": "synthetic"}}, handle)
    return path


@pytest.fixture
def dataset_root(tmp_path):
    return tmp_path / "DeepAccident"


@pytest.fixture
def two_agent_infos(dataset_root):
    # Intentionally reverse order to test fixed cooperative-agent ordering.
    infos = [
        make_info(agent="infrastructure", pose=transform(x=10, y=4, yaw=0.0)),
        make_info(agent="ego_vehicle", pose=transform(x=10, y=2, yaw=0.0)),
    ]
    return write_infos(dataset_root, infos)


def test_canonical_sample_and_native_data_preservation(dataset_root, two_agent_infos):
    adapter = DeepAccidentAdapter(dataset_root, "train_mini", info_path=two_agent_infos)
    assert len(adapter) == 1
    assert isinstance(adapter[0], CanonicalV2XSample)

    sample = adapter[0]
    sample.validate()
    assert (sample.source.dataset, sample.source.sequence_id, sample.source.frame_id) == (
        "DeepAccident", "Town01_case", "10"
    )
    assert sample.reference_agent_id == "ego_vehicle"
    assert [agent.agent_id for agent in sample.agents] == ["ego_vehicle", "infrastructure"]
    assert [agent.agent_type for agent in sample.agents] == ["vehicle", "infrastructure"]
    assert [agent.is_reference for agent in sample.agents] == [True, False]
    np.testing.assert_allclose(sample.agents[0].T_ref_from_agent, np.eye(4))
    assert sample.extras["split"] == "train_mini"
    assert sample.extras["info_metadata"]["version"] == "synthetic"
    assert sample.extras["scenario_length_native"] == 80
    assert sample.extras["relative_pose_status"].startswith("provisional")

    # Native paths are rebased but are NOT decoded or claimed as canonical.
    agent = sample.agents[0]
    assert agent.lidar is None
    assert agent.cameras == []
    assert agent.extras["native_lidar_path"] == str(
        dataset_root / SCENARIO_TYPE / "ego_vehicle" / "lidar01" / "seq" / "000010.npz"
    )
    np.testing.assert_allclose(agent.extras["native_lidar_to_ego_matrix"], transform(z=1.5))
    native_camera = agent.extras["native_cameras"][0]
    assert native_camera["camera_id"] == CAMERA_NAME
    assert native_camera["image"] is None
    assert native_camera["calibration_status"] == "native_not_canonical"
    np.testing.assert_allclose(native_camera["native_intrinsic"][0], [800, 0, 320])
    np.testing.assert_allclose(native_camera["native_lidar_to_camera_matrix"], transform(x=0.2))

    # No native LiDAR-frame boxes are silently promoted to reference-frame boxes.
    annotations = sample.annotations
    assert annotations.boxes_3d.shape == (0, 7)
    assert annotations.native_labels == []
    assert annotations.track_ids.shape == (0,)
    assert annotations.labels is None
    assert annotations.velocity_xy is None
    assert annotations.extras["omitted_from_canonical"] is True
    native = annotations.extras["native_annotations_by_agent"]["ego_vehicle"]
    np.testing.assert_allclose(native["gt_boxes"][0], [2, -3, .5, 4, 1.8, 1.5, .2])
    assert native["gt_names"] == ["car"]
    np.testing.assert_array_equal(native["vehicle_id"], [100])
    np.testing.assert_allclose(native["gt_velocity"], [[1.5, -0.2]])


def test_relative_pose_reflection_and_canonical_override(dataset_root, two_agent_infos):
    carla_sample = DeepAccidentAdapter(
        dataset_root, info_path=two_agent_infos, native_axes="carla"
    )[0]
    other = carla_sample.agents[1]
    np.testing.assert_allclose(other.T_ref_from_agent[:3, 3], [0, -2, 0])
    assert other.extras["relative_pose_conversion"] == "provisional_carla"

    canonical_sample = DeepAccidentAdapter(
        dataset_root, info_path=two_agent_infos, native_axes="canonical"
    )[0]
    np.testing.assert_allclose(canonical_sample.agents[1].T_ref_from_agent[:3, 3], [0, 2, 0])
    assert canonical_sample.extras["native_axes_assumption"] == "canonical"


def test_reflection_of_yaw_is_still_a_proper_rotation(dataset_root):
    path = write_infos(dataset_root, [
        make_info(),
        make_info(agent="other_vehicle", pose=transform(yaw=np.pi / 2)),
    ])
    T = DeepAccidentAdapter(dataset_root, info_path=path)[0].agents[1].T_ref_from_agent
    np.testing.assert_allclose(T[:3, :3], transform(yaw=-np.pi / 2)[:3, :3], atol=1e-7)
    assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0)


def test_timestamp_requires_explicit_frame_rate(dataset_root, two_agent_infos):
    unknown = DeepAccidentAdapter(dataset_root, info_path=two_agent_infos)[0]
    assert unknown.timestamp_s == 0.0
    assert unknown.extras["native_frame_index"] == 10
    assert unknown.extras["timestamp_status"] == "seconds_unknown_placeholder_zero"

    nominal = DeepAccidentAdapter(
        dataset_root, info_path=two_agent_infos, frame_rate_hz=20
    )[0]
    assert nominal.timestamp_s == pytest.approx(0.5)
    assert nominal.extras["timestamp_status"] == "nominal_frame_rate"


def test_groups_scenes_frames_and_skips_missing_reference(dataset_root):
    path = write_infos(dataset_root, [
        make_info(scene="B", frame=20),
        make_info(scene="A", frame=20),
        make_info(scene="A", frame=10),
        make_info(scene="A", frame=20, agent="infrastructure"),
        make_info(scene="C", frame=10, agent="other_vehicle"),
    ])
    with pytest.warns(UserWarning, match="Skipped 1 frames"):
        adapter = DeepAccidentAdapter(dataset_root, info_path=path)
    assert len(adapter) == 3
    assert [(adapter[i].source.sequence_id, adapter[i].source.frame_id)
            for i in range(len(adapter))] == [("A", "10"), ("A", "20"), ("B", "20")]
    assert len(adapter[1].agents) == 2


def test_reference_agent_is_configurable(dataset_root):
    path = write_infos(dataset_root, [make_info(agent="infrastructure")])
    adapter = DeepAccidentAdapter(
        dataset_root, info_path=path, reference_agent_id="infrastructure"
    )
    sample = adapter[0]
    assert sample.reference_agent_id == "infrastructure"
    assert sample.agents[0].agent_type == "infrastructure"
    np.testing.assert_allclose(sample.agents[0].T_ref_from_agent, np.eye(4))


def test_indexing_and_representations(dataset_root, two_agent_infos):
    adapter = DeepAccidentAdapter(dataset_root, info_path=two_agent_infos)
    assert adapter[-1].source.frame_id == "10"
    assert adapter.get_sample(-1).source.frame_id == "10"
    with pytest.raises(IndexError):
        _ = adapter[1]
    with pytest.raises(IndexError):
        adapter.get_sample(-2)
    with pytest.raises(TypeError):
        _ = adapter["0"]
    assert "DeepAccidentAdapter" in repr(adapter)


def test_autodiscovery_by_split(dataset_root):
    write_infos(dataset_root, [make_info()], split="train_mini")
    write_infos(dataset_root, [make_info(frame=30)], split="val_mini")
    adapter = DeepAccidentAdapter(dataset_root, split="val_mini")
    assert adapter[0].source.frame_id == "30"
    with pytest.raises(FileNotFoundError, match="exactly one"):
        DeepAccidentAdapter(dataset_root)  # Ambiguous without split.
    with pytest.raises(ValueError, match="Unsupported split"):
        DeepAccidentAdapter(dataset_root, split="invalid")
    with pytest.raises(FileNotFoundError):
        DeepAccidentAdapter(dataset_root, info_path=dataset_root / "missing.pkl")


@pytest.mark.parametrize("bad_fps", [0, -1, float("nan"), float("inf")])
def test_invalid_frame_rate_rejected(dataset_root, two_agent_infos, bad_fps):
    with pytest.raises(ValueError, match="frame_rate_hz"):
        DeepAccidentAdapter(dataset_root, info_path=two_agent_infos, frame_rate_hz=bad_fps)


def test_invalid_coordinate_option_and_reference(dataset_root, two_agent_infos):
    with pytest.raises(ValueError, match="native_axes"):
        DeepAccidentAdapter(dataset_root, info_path=two_agent_infos, native_axes="unknown")
    with pytest.raises(ValueError, match="reference_agent_id"):
        DeepAccidentAdapter(dataset_root, info_path=two_agent_infos, reference_agent_id="")


@pytest.mark.parametrize("mutation, error", [
    (lambda info: info.update(gt_names=[]), "gt_names count"),
    (lambda info: info.update(gt_boxes=np.ones((1, 6))), "gt_boxes must"),
    (lambda info: info.update(gt_velocity=np.ones((1, 3))), "gt_velocity must"),
    (lambda info: info.update(vehicle_id=np.ones((2,))), "vehicle_id must"),
    (lambda info: info.update(cams=[]), "Native cams must"),
    (lambda info: info.update(ego_to_world_matrix=np.zeros((3, 3))), "finite \\(4, 4\\)"),
    (lambda info: info.update(ego_to_world_matrix=np.diag([1, 1, -1, 1])), "proper rigid"),
])
def test_malformed_records_rejected_on_access(dataset_root, mutation, error):
    info = make_info()
    mutation(info)
    path = write_infos(dataset_root, [info])
    adapter = DeepAccidentAdapter(dataset_root, info_path=path)
    with pytest.raises(ValueError, match=error):
        adapter[0]


def test_duplicate_agents_and_invalid_payload_rejected(dataset_root):
    path = write_infos(dataset_root, [make_info(), make_info()])
    with pytest.raises(ValueError, match="Duplicate DeepAccident agent"):
        DeepAccidentAdapter(dataset_root, info_path=path)
    with path.open("wb") as handle:
        pickle.dump({"infos": "not a list"}, handle)
    with pytest.raises(ValueError, match="'infos' list"):
        DeepAccidentAdapter(dataset_root, info_path=path)


def test_missing_reference_for_all_frames_rejected(dataset_root):
    path = write_infos(dataset_root, [make_info(agent="other_vehicle")])
    with pytest.warns(UserWarning, match="Skipped 1 frames"):
        with pytest.raises(ValueError, match="No synchronized frames"):
            DeepAccidentAdapter(dataset_root, info_path=path)


def test_camera_file_policy_and_optional_image_decode(dataset_root):
    image_path = dataset_root / "camera.jpg"
    path = write_infos(dataset_root, [make_info(image_path=str(image_path))])
    with pytest.raises(FileNotFoundError, match="Missing DeepAccident image"):
        DeepAccidentAdapter(dataset_root, info_path=path, strict_camera_files=True)[0]
    with pytest.warns(UserWarning, match="Missing DeepAccident camera image"):
        missing = DeepAccidentAdapter(dataset_root, info_path=path, load_images=True)[0]
    assert missing.agents[0].extras["native_cameras"][0]["image"] is None

    pil = pytest.importorskip("PIL.Image")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    pil.fromarray(np.full((3, 4, 3), 128, dtype=np.uint8), mode="RGB").save(image_path)
    sample = DeepAccidentAdapter(
        dataset_root, info_path=path, load_images=True, strict_camera_files=True
    )[0]
    native = sample.agents[0].extras["native_cameras"][0]
    assert native["image"].shape == (3, 4, 3)
    assert native["image"].dtype == np.uint8
    assert sample.agents[0].cameras == []

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DeepAccident adapter tests."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Optional DeepAccident dataset root.",
    )
    parser.parse_args()
