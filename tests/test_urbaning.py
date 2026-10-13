from tests.cli import parser_for, urbaning_adapter

from urbaning.data import Sequence

from adapters.urbaning_adapter import UrbanIngAdapter


# ============================================================
# Configuration
# ============================================================

# ============================================================
# Test 1: Official UrbanIng loader
# ============================================================

def main(argv=None):
    args = parser_for("urbaning").parse_args(argv)
    adapter = urbaning_adapter(args)
    DATASET_ROOT = adapter.dataset_root
    SEQUENCE_NAME, sequence_index = adapter._samples[args.index]
    print("=" * 70)
    print("TEST 1: OFFICIAL URBANING SEQUENCE LOADER")
    print("=" * 70)

    print(f"Dataset root : {DATASET_ROOT}")
    print(f"Sequence     : {SEQUENCE_NAME}")

    sequence_folder = DATASET_ROOT / "dataset" / SEQUENCE_NAME
    label_file = DATASET_ROOT / "labels" / f"{SEQUENCE_NAME}.json"

    print(f"Sequence path exists : {sequence_folder.exists()}")
    print(f"Label file exists    : {label_file.exists()}")

    if not sequence_folder.exists():
        raise FileNotFoundError(
            f"\nSequence folder was not found:\n{sequence_folder}"
        )

    if not label_file.exists():
        raise FileNotFoundError(
            f"\nLabel file was not found:\n{label_file}"
        )


    print("\nLoading sequence using official UrbanIng devkit...")

    sequence = Sequence(
        str(DATASET_ROOT),
        SEQUENCE_NAME,
    )

    print("Sequence loaded successfully.")
    print(f"Sequence type : {type(sequence)}")
    print(f"Number of frames : {len(sequence)}")


    # ============================================================
    # Test one official frame
    # ============================================================

    print("\nLoading first frame...")

    frame = sequence[sequence_index]

    print("First frame loaded successfully.")
    print(f"Frame type : {type(frame)}")
    print(f"Frame timestamp : {frame.timestamp}")


    # ============================================================
    # Test 2: Our UrbanIng adapter
    # ============================================================

    print("\n" + "=" * 70)
    print("TEST 2: OUR URBANING ADAPTER")
    print("=" * 70)

    print("Creating UrbanIngAdapter...")

    # Adapter already configured above.

    print("Adapter created successfully.")
    print(f"Number of canonical samples : {len(adapter)}")


    # ============================================================
    # Convert first sample
    # ============================================================

    print("\nConverting first UrbanIng frame to CanonicalV2XSample...")

    sample = adapter[args.index]

    print("Conversion completed successfully.")


    # ============================================================
    # Validate canonical sample
    # ============================================================

    print("\nRunning canonical schema validation...")

    sample.validate()

    print("Validation PASSED.")


    # ============================================================
    # Inspect canonical sample
    # ============================================================

    print("\n" + "=" * 70)
    print("CANONICAL SAMPLE SUMMARY")
    print("=" * 70)

    print(f"Schema version      : {sample.schema_version}")
    print(f"Dataset             : {sample.source.dataset}")
    print(f"Sequence ID         : {sample.source.sequence_id}")
    print(f"Frame ID            : {sample.source.frame_id}")
    print(f"Timestamp [s]       : {sample.timestamp_s}")
    print(f"Reference agent     : {sample.reference_agent_id}")
    print(f"Number of agents    : {len(sample.agents)}")


    # ============================================================
    # Inspect agents
    # ============================================================

    print("\nAGENTS")
    print("-" * 70)

    for agent in sample.agents:

        print(f"\nAgent ID        : {agent.agent_id}")
        print(f"Agent type      : {agent.agent_type}")
        print(f"Is reference    : {agent.is_reference}")
        print(f"Transform shape : {agent.T_ref_from_agent.shape}")

        # LiDAR
        if agent.lidar is not None:

            print(f"LiDAR xyz shape : {agent.lidar.xyz.shape}")

            if agent.lidar.intensity is not None:
                print(
                    f"Intensity shape : "
                    f"{agent.lidar.intensity.shape}"
                )
            else:
                print("Intensity       : None")

            if agent.lidar.time_offset_s is not None:
                print(
                    f"Time offset shape : "
                    f"{agent.lidar.time_offset_s.shape}"
                )

            if agent.lidar.sensor_index is not None:
                print(
                    f"Sensor index shape : "
                    f"{agent.lidar.sensor_index.shape}"
                )

        else:
            print("LiDAR           : None")

        # Cameras
        print(f"Number of cameras : {len(agent.cameras)}")

        for camera in agent.cameras:

            print(
                f"    {camera.camera_id}: "
                f"modality={camera.modality}, "
                f"image_shape={camera.image.shape}"
            )


    # ============================================================
    # Inspect annotations
    # ============================================================

    print("\n" + "=" * 70)
    print("ANNOTATIONS")
    print("=" * 70)

    annotations = sample.annotations

    print(f"3D boxes shape    : {annotations.boxes_3d.shape}")
    print(f"Native labels     : {len(annotations.native_labels)}")
    print(f"Track IDs shape   : {annotations.track_ids.shape}")

    if annotations.labels is not None:
        print(f"Canonical labels  : {annotations.labels.shape}")
    else:
        print("Canonical labels  : None")

    if annotations.velocity_xy is not None:
        print(f"Velocity shape    : {annotations.velocity_xy.shape}")
    else:
        print("Velocity          : None")


    # Show only a few objects so terminal output stays readable
    print("\nFirst few annotations:")

    number_to_show = min(5, len(annotations.native_labels))

    for i in range(number_to_show):

        print(
            f"{i}: "
            f"class={annotations.native_labels[i]}, "
            f"track_id={annotations.track_ids[i]}, "
            f"box={annotations.boxes_3d[i]}"
        )


    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


def test_validation(cli_args):
    main(cli_args("urbaning"))


if __name__ == "__main__":
    main()
