# V2X Preprocessing — Tests Explained

## Why these tests exist

The adapters must return well-structured **canonical V2X samples**, even though the original datasets have different files and coordinate conventions. These tests check that the code loads data correctly, handles invalid inputs safely, and performs consistent geometry conversions.

**How to read the results:** **Pass condition** below means what the test considers successful, **not a claim that it was rerun while writing this document**. The previously reported project test run had **58 passing tests** (about 26 seconds). That result does not establish that every real-data check ran: tests requiring unavailable datasets can be skipped, and saved plots still require visual inspection.

## 1. Shared test support (not themselves tests)

### `conftest.py`
- **Purpose:** Supplies dataset paths and command-line options to pytest. If a default dataset is unavailable, its real-data tests are **skipped**; an explicitly requested missing path is an **error**. It also configures noninteractive plotting.
- **Result:** Makes tests portable without pretending that missing datasets were tested.

### `cli.py`
- **Purpose:** Gives validation scripts consistent commands for choosing a dataset, frame, agents, and output location. For V2XPnP, it selects vehicle/infrastructure pairs using actual YAML `infra` flags rather than assuming particular folder IDs.
- **Result:** Reusable, unambiguous test selection; not a separate pass/fail test.

## 2. Dataset-independent and portability tests

### `test_portability.py`

| Test | What it evaluates | Pass condition |
|---|---|---|
| `test_relative_configuration_is_project_relative` | Relative configuration paths | Resolve relative to the project, not the current terminal folder. |
| `test_absolute_external_root_and_cli_override` | Custom external dataset path | A supplied path overrides YAML; an invalid configured path raises an error. |
| `test_invalid_configuration_reports_error` | Malformed/incomplete YAML (3 cases) | Each invalid configuration is rejected. |
| `test_output_configuration_creates_parents` | Output-directory configuration | Nested output folders are created. |
| `test_native_selection_and_external_adapter` | V2XPnP scenario/agent discovery | Real `infra` flags identify agents even with unusual IDs; external-folder loading and Y-axis conversion work. |
| `test_pcd_nonfinite_filter_preserves_all_attributes` | TUMTraf PCD cleaning | Invalid points are removed without misaligning intensity or extra fields. |
| `test_imports_from_other_directory_have_no_execution` | Import safety | Modules import outside the project directory without printing or creating files there. |
| `test_all_module_cli_help_from_other_directory` | Script command-line usability | Test modules respond to `--help` from another directory. |
| `test_geometry_cli_external_root_from_other_directory` | Geometry-test portability | V2XPnP geometry CLI accepts an external dataset root. |
| `test_downloader_destination_is_explicit_and_import_is_safe` | Download safety | Importing does not download anything; calling the download command uses the requested destination. |

**Result established:** Path handling, CLI access, safe imports, and selected format checks behave as the assertions require on test data. These are mainly **dataset-free regression tests**.

## 3. UrbanIng tests

### `test_urbaning.py`
- **Evaluates:** Whether the official UrbanIng devkit loads a sequence and frame, and whether our adapter returns a valid `CanonicalV2XSample` with agents, LiDAR/cameras, and annotations.
- **Pass condition:** Data loads and canonical schema validation succeeds. The script prints sample details; it does not prove every calibration is physically correct.

### `test_urbaning_geometry.py`
- **Evaluates:** Numerical sanity of agent transforms and 3D boxes; produces bird's-eye-view (BEV, top-down) pictures of reference LiDAR + boxes and all agents transformed into one frame.
- **Pass condition:** Sample loads and plots are generated. The script **prints warnings** for some geometry concerns rather than failing on each warning. **Human visual inspection is still needed** to confirm alignment.

### `test_urbaning_multiframe.py`
- **Evaluates:** Several frames per sequence: timestamps, unique agents, reference identity, rigid rotations, finite LiDAR/camera data, and valid annotation shapes. Checks that timestamps increase **within the same sequence**.
- **Pass condition:** All selected frames pass and within-sequence time ordering is consistent; otherwise the script exits with failure.

**Result established:** Real-data UrbanIng tests provide smoke, geometry-inspection, and multi-frame consistency checks **when the dataset is available**. A pytest skip is not a real-data pass.

## 4. V2XPnP tests

### `test_v2xpnp_adapter.py`

| Test | What it evaluates | Pass condition |
|---|---|---|
| `test_validation` | A selected canonical frame | Agents, LiDAR arrays, annotation boxes/IDs, and relative rotations have expected structure. Requires native data. |
| `test_annotation_ids_classes_and_reference_priority` | Multiple agents labeling the same objects | Duplicate object IDs are merged; reference-agent labels win; unusable integer IDs become `-1` while native IDs remain available. |
| `test_empty_annotations_have_schema_shapes` | Frame with no labels | Boxes still have shape `(0, 7)` and track IDs shape `(0,)`. |
| `test_samples_with_native_cameras_validate` | Camera metadata/image options (4 cases) | Native calibration and optional decoded images are preserved; no unverified canonical `CameraData` is invented. |
| `test_missing_camera_files_preserve_strict_behavior` | Missing images (4 cases) | Strict mode raises an error; non-strict mode warns and keeps metadata. |
| `test_invalid_native_calibration_still_errors` | Bad intrinsics/extrinsics (4 cases) | Invalid matrix shape or nonfinite values are rejected. |
| `test_load_images_false_does_not_decode` | Optional image decoding | Malformed image bytes are ignored when loading is off but cause an error when loading is requested. |

### `test_v2xpnp_geometry.py`
- **Evaluates:** Native OpenCOOD pose calculations: identity, inverse transforms, world-frame composition, and valid rotation matrices.
- **Pass condition:** All mathematical relationships agree. **This alone does not verify LiDAR alignment or canonical axis conversion.**

### `test_v2xpnp_lidar_alignment.py`
- **Evaluates:** Reads vehicle/infrastructure point clouds, moves infrastructure points into vehicle coordinates, verifies inverse and point round-trip operations, and saves two BEV overlays.
- **Pass condition:** Mathematical checks succeed and plots are saved. **Physical alignment still requires visual inspection.** Coordinates remain OpenCOOD-native in this script.

### `test_v2xpnp_overlap_alignment.py`
- **Evaluates:** Produces two closer BEV overlays and a shared-occupancy map; reports how many 2D grid cells both sensors occupy.
- **Pass condition:** Valid input, valid relative rotation, and three saved images. **Occupancy overlap is only a diagnostic**, not proof of correct 3D registration or canonical handedness.

**Result established:** Unit tests check annotation and camera handling; real-data scripts inspect pose consistency and LiDAR overlap. The generated figures are evidence for review, **not automatic proof of correct physical geometry**.

## 5. TUMTraf tests

### `test_tumtraf_adapter.py`
- **Evaluates:** Loads one TUMTraf mini OpenLABEL frame; checks the infrastructure/vehicle agents, finite LiDAR points, 7-value boxes, and identity transform for the reference infrastructure.
- **Pass condition:** Canonical sample validates and smoke-test assertions pass. Images are only counted as **uncalibrated paths**, not validated canonical cameras.
- **Data requirement:** Requires the extracted TUMTraf mini dataset; may be skipped by pytest if the default dataset is missing.

## 6. DeepAccident tests

### `test_deepaccident_adapter.py`

These tests use **small synthetic pickle files** shaped like the official converted-info records. They check metadata organization and error handling, **not verified decoding or calibration of real DeepAccident sensors**.

| Test | What it evaluates | Pass condition |
|---|---|---|
| `test_canonical_sample_and_native_data_preservation` | Multi-agent sample and metadata | Canonical structure is valid; native camera, LiDAR paths and annotations are preserved in `extras` rather than falsely declared converted. |
| `test_relative_pose_reflection_and_canonical_override` | Coordinate assumptions | CARLA-style Y reflection and canonical-axis override yield expected relative translations. |
| `test_reflection_of_yaw_is_still_a_proper_rotation` | Rotation after axis reflection | Expected mirrored yaw; rotation determinant stays `+1`. |
| `test_timestamp_requires_explicit_frame_rate` | Frame index versus seconds | Unknown seconds are marked as such; supplied frame rate gives nominal seconds. |
| `test_groups_scenes_frames_and_skips_missing_reference` | Cooperative frame indexing | Agents with the same scene/frame are grouped; missing-reference frames are skipped. |
| `test_reference_agent_is_configurable` | Reference choice | An infrastructure agent can be selected as reference with identity transform. |
| `test_indexing_and_representations` | Collection behavior | Negative indexing works; invalid indices/types raise errors; representation names the adapter. |
| `test_autodiscovery_by_split` | Pickle discovery | Correct split is selected; ambiguous, invalid, or missing files are rejected. |
| `test_invalid_frame_rate_rejected` | Invalid frame rate (4 cases) | Zero, negative, NaN, and infinity are rejected. |
| `test_invalid_coordinate_option_and_reference` | Invalid initialization | Unsupported axis setting or empty reference ID is rejected. |
| `test_malformed_records_rejected_on_access` | Corrupted record fields (7 cases) | Bad boxes, labels, velocities, cameras, or transforms raise errors. |
| `test_duplicate_agents_and_invalid_payload_rejected` | Duplicate/bad pickle structure | Duplicate agent records or wrong `infos` type are rejected. |
| `test_missing_reference_for_all_frames_rejected` | Dataset with no usable reference | Adapter raises a clear error. |
| `test_camera_file_policy_and_optional_image_decode` | Missing versus available RGB files | Strict missing-file error, non-strict warning, and optional RGB decoding behave correctly; canonical cameras remain empty. |

**Result established:** The adapter's **provisional metadata-level behavior** is tested. Camera projection, native `.npz` LiDAR decoding, and reference-frame annotation conversion **remain unverified and are not validated by these tests**.

## 7. What the results mean overall

- **Automated assertions:** A passing test confirms only the behavior it explicitly checks.
- **Synthetic tests:** Useful for correctness and error handling without needing large datasets; they do not replace real-data verification.
- **Real-data smoke tests:** Check selected frames only when the native dataset is accessible; unavailable datasets may be **skipped**.
- **Geometry plots:** Help detect alignment mistakes, but require manual inspection; a script completing is not equivalent to proven calibration correctness.

**Overall:** The previously reported suite result was **58 passed**, but the remaining thesis work includes real-data verification, especially DeepAccident sensor/calibration conversion and cross-dataset use in a single downstream pipeline. For current exact pass/skip counts, run `python -m pytest -ra` in the project environment.
