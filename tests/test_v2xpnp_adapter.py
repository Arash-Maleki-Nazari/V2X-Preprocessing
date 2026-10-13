
from tests.cli import parser_for
from project_paths import dataset_path

import numpy as np
import pytest

from adapters.v2xpnp_adapter import V2XPnPAdapter


def main(argv=None):
    args = parser_for("v2xpnp").parse_args(argv)
    root = dataset_path("v2xpnp", args.root, args.config)
    adapter = V2XPnPAdapter(
        dataset_root=root,
        reference_agent_id=args.reference_agent,
        load_images=False,
        split=args.split,
    )

    print("=" * 65)
    print("V2XPnP CANONICAL ADAPTER TEST")
    print("=" * 65)

    print("Total indexed frames:", len(adapter))

    index = args.index
    if args.scenario or args.frame:
        matching = [i for i, (scenario, frame, _) in enumerate(adapter._index)
                    if (not args.scenario or scenario.name == args.scenario
                        or scenario == root / args.scenario)
                    and (not args.frame or frame == args.frame)]
        if not matching:
            raise ValueError("No adapter sample matches the requested scenario/frame")
        index = matching[args.index]
    sample = adapter[index]
    sample.validate()

    print("Sequence:", sample.source.sequence_id)
    print("Frame:", sample.source.frame_id)
    print("Reference agent:", sample.reference_agent_id)
    print("Agents:", len(sample.agents))
    print("Annotations:", len(sample.annotations.boxes_3d))

    assert sample.annotations.boxes_3d.ndim == 2
    assert sample.annotations.boxes_3d.shape[1] == 7
    assert sample.annotations.track_ids.dtype == np.int64
    assert sample.annotations.track_ids.shape == (len(sample.annotations.boxes_3d),)
    assert sample.annotations.labels is None
    assert len(sample.annotations.extras["native_object_ids"]) == len(sample.annotations.boxes_3d)

    if len(sample.annotations.boxes_3d):
        assert np.all(
            sample.annotations.boxes_3d[:, 3:6] > 0
        )

    for agent in sample.agents:
        print()
        print("Agent:", agent.agent_id)
        print("Type:", agent.agent_type)
        print("Reference:", agent.is_reference)
        print("LiDAR points:", len(agent.lidar.xyz))
        print("Cameras:", len(agent.cameras))
        print("T_ref_from_agent:")
        print(agent.T_ref_from_agent)

        assert agent.lidar.xyz.ndim == 2
        assert agent.lidar.xyz.shape[1] == 3
        assert len(agent.lidar.intensity) == len(
            agent.lidar.xyz
        )

        R = agent.T_ref_from_agent[:3, :3]

        np.testing.assert_allclose(
            R.T @ R,
            np.eye(3),
            atol=1e-6,
        )

        np.testing.assert_allclose(
            np.linalg.det(R),
            1.0,
            atol=1e-6,
        )

        if agent.is_reference:
            np.testing.assert_allclose(
                agent.T_ref_from_agent,
                np.eye(4),
                atol=1e-7,
            )

    print()
    print("PASS: Initial canonical sample checks")
    print(
        "PENDING: Box overlay, camera projection, "
        "multi-frame and canonical-frame validation"
    )


def test_validation(cli_args):
    main(cli_args("v2xpnp"))



def _native_object(label="Car", x=1.0):
    return {"location": [x, 2.0, 3.0], "angle": [0.0, 0.0, 0.0],
            "extent": [2.0, 1.0, 0.75], "obj_type": label}


def test_annotation_ids_classes_and_reference_priority():
    from adapters.v2xpnp_adapter import collect_annotations
    native_ids = [42, "007", "object-uuid", str(2**63), "3.5", "-8"]
    objects = {key: _native_object(f"class-{i}") for i, key in enumerate(native_ids)}
    records = [
        {"agent_id": "7", "metadata": {"vehicles": objects}},
        {"agent_id": "-9", "metadata": {"vehicles": {
            "42": _native_object("duplicate", x=99), "8": _native_object("Truck")}}},
    ]
    annotations = collect_annotations(records, np.zeros(6))
    annotations.validate()
    assert annotations.boxes_3d.shape == (7, 7)
    assert annotations.boxes_3d.dtype == np.float32
    assert annotations.track_ids.dtype == np.int64
    np.testing.assert_array_equal(annotations.track_ids, [42, 7, -1, -1, -1, -1, 8])
    assert annotations.labels is None
    assert annotations.native_labels == [f"class-{i}" for i in range(6)] + ["Truck"]
    assert annotations.extras["native_object_ids"] == native_ids + ["8"]
    assert annotations.extras["object_attributes"][0]["source_agent"] == "7"
    assert annotations.boxes_3d[0, 0] == 1.0


def test_empty_annotations_have_schema_shapes():
    from adapters.v2xpnp_adapter import collect_annotations
    annotations = collect_annotations([], np.zeros(6))
    annotations.validate()
    assert annotations.boxes_3d.shape == (0, 7)
    assert annotations.track_ids.shape == (0,)
    assert annotations.track_ids.dtype == np.int64
    assert annotations.native_labels == []
    assert annotations.labels is None
    assert annotations.extras["native_object_ids"] == []


def _camera_sample(tmp_path, calibration, *, image=True):
    import yaml
    from PIL import Image
    agent = tmp_path / "scenario" / "7"
    agent.mkdir(parents=True)
    metadata = {"infra": False, "lidar_pose": [0, 0, 0, 0, 0, 0],
                "vehicles": {"123": _native_object()}, "camera0": calibration}
    (agent / "000042.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")
    np.asarray([[1, 2, 3, 4]], dtype="<f4").tofile(agent / "000042.bin")
    if image:
        Image.new("RGB", (3, 2), color=(10, 20, 30)).save(agent / "000042_camera0.png")
    return agent


@pytest.mark.parametrize("load_images", [False, True])
@pytest.mark.parametrize("calibrated", [False, True])
def test_samples_with_native_cameras_validate(tmp_path, load_images, calibrated):
    calibration = {"cords": [1, 2, 3, 4, 5, 6], "custom_field": "preserved"}
    if calibrated:
        calibration.update({
            "intrinsic": [[100, 0, 1], [0, 110, 1], [0, 0, 1]],
            "extrinsic": [[0, -1, 0, 3], [1, 0, 0, 4], [0, 0, 1, 5], [0, 0, 0, 1]],
        })
    agent_path = _camera_sample(tmp_path, calibration)
    adapter = V2XPnPAdapter(tmp_path, reference_agent_id="7", load_images=load_images,
                           strict_camera_files=True)
    sample = adapter[0]
    sample.validate()
    assert sample.annotations.track_ids.dtype == np.int64
    assert sample.annotations.labels is None
    assert sample.annotations.extras["native_object_ids"] == ["123"]
    agent = sample.agents[0]
    assert agent.cameras == []
    native = agent.extras["native_cameras"][0]
    assert native["image_path"] == str(agent_path / "000042_camera0.png")
    assert native["camera_id"] == "camera0"
    assert native["native_metadata"] == calibration
    assert native["native_camera_pose"] == calibration["cords"]
    assert native["calibration_status"] == "native_not_canonical"
    assert "unverified" in native["omission_reason"]
    if calibrated:
        np.testing.assert_array_equal(native["native_intrinsic"], calibration["intrinsic"])
        np.testing.assert_array_equal(native["native_extrinsic_camera_to_lidar"], calibration["extrinsic"])
    else:
        assert native["native_intrinsic"] is None
        assert native["native_extrinsic_camera_to_lidar"] is None
    if load_images:
        assert native["image"].shape == (2, 3, 3)
        np.testing.assert_array_equal(native["image"][0, 0], [10, 20, 30])
    else:
        assert native["image"] is None
    # Neither LiDAR coordinates nor reference transforms change with camera options.
    np.testing.assert_array_equal(agent.lidar.xyz, [[1, -2, 3]])
    np.testing.assert_array_equal(agent.T_ref_from_agent, np.eye(4))


@pytest.mark.parametrize("load_images", [False, True])
@pytest.mark.parametrize("strict", [False, True])
def test_missing_camera_files_preserve_strict_behavior(tmp_path, load_images, strict):
    calibration = {"cords": [1, 2, 3, 0, 0, 0]}
    _camera_sample(tmp_path, calibration, image=False)
    adapter = V2XPnPAdapter(tmp_path, reference_agent_id="7", load_images=load_images,
                           strict_camera_files=strict)
    if strict:
        with pytest.raises(FileNotFoundError, match="Camera metadata exists but image is missing"):
            adapter[0]
    else:
        with pytest.warns(UserWarning, match="Camera metadata exists but image is missing"):
            sample = adapter[0]
        sample.validate()
        native = sample.agents[0].extras["native_cameras"][0]
        assert native["image_path"] is None
        assert native["image"] is None
        assert native["native_metadata"] == calibration


@pytest.mark.parametrize("field, value, message", [
    ("intrinsic", [[1, 2]], "Invalid camera intrinsic"),
    ("intrinsic", [[float("nan"), 0, 0], [0, 1, 0], [0, 0, 1]], "Invalid camera intrinsic"),
    ("extrinsic", [[1, 2]], "Invalid native camera extrinsic"),
    ("extrinsic", np.full((4, 4), float("inf")).tolist(), "Invalid native camera extrinsic"),
])
def test_invalid_native_calibration_still_errors(tmp_path, field, value, message):
    _camera_sample(tmp_path, {field: value})
    adapter = V2XPnPAdapter(tmp_path, reference_agent_id="7")
    with pytest.raises(ValueError, match=message):
        adapter[0]


def test_load_images_false_does_not_decode(tmp_path):
    from PIL import UnidentifiedImageError
    agent = _camera_sample(tmp_path, {})
    (agent / "000042_camera0.png").write_bytes(b"not a decodable image")
    V2XPnPAdapter(tmp_path, reference_agent_id="7", load_images=False)[0].validate()
    with pytest.raises(UnidentifiedImageError):
        V2XPnPAdapter(tmp_path, reference_agent_id="7", load_images=True)[0]


if __name__ == "__main__":
    main()
