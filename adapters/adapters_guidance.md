# V2X Dataset Adapters — Core Ideas and Design Choices

## Why do we need adapters?

Our four V2X datasets store frames, sensors, poses, and annotations in different formats. An **adapter** reads one dataset in its original format and translates it into the shared structure defined in `schema.py`: a `CanonicalV2XSample`.

This keeps later code (visualization, analysis, and eventually model training) independent of the original dataset format. The goal is **one way to access samples**, not to pretend that every dataset contains identical information.

## 1. `base_adapter.py` — The common contract

**Core idea:** Every dataset adapter must support `len(adapter)` and `adapter[index]`, and must produce a `CanonicalV2XSample`.
Rather than repeat common checks in four files, `BaseAdapter` handles them once:

- `__init__` stores a resolved dataset path and an optional train/validation/test split.
- `__len__` and `get_sample(index)` are **abstract**: each dataset must implement its own frame counting and data loading, because its files are organized differently.
- `__getitem__` enables `adapter[0]`, checks index bounds and return type, and runs `sample.validate()`.
- `__repr__` gives a readable description for debugging.

**Important distinction:** `get_sample()` does the dataset-specific conversion; `adapter[index]` adds the shared safety checks.

## 2. `urbaning_adapter.py` — Use the official dataset tools

**Core idea:** Use the official UrbanIng devkit to read synchronized frames and convert vehicles, infrastructure, LiDAR, cameras, and annotations.

- **Official devkit:** Reuses UrbanIng's own loading, synchronization, and calibration logic instead of reimplementing it.
- **Reference agent:** Uses `vehicle1` by default. Other agents' poses and object boxes are expressed relative to that reference, so the sample has a consistent spatial viewpoint.
- **One LiDAR object per agent:** Combines the agent's LiDAR scans while retaining per-point timing and sensor IDs. This simplifies later processing without losing which sensor contributed each point.
- **Camera conversion:** Uses undistorted images, changes OpenCV BGR to RGB, and provides intrinsics and camera-to-agent transforms.
- **Practical indexing:** Builds a sequence/frame index and caches only the most recently used sequence to avoid repeatedly loading it or holding every sequence in memory.
- **Time:** Expresses each frame in seconds from the beginning of its sequence, while preserving the original timestamp.

**Result:** This adapter populates canonical LiDAR, cameras, poses, and 3D annotations where available.

## 3. `v2xpnp_adapter.py` — Reconcile CARLA/OpenCOOD geometry

**Core idea:** Find scenario/agent/frame folders, read YAML metadata and binary LiDAR scans, then convert them into canonical coordinates.

- **Scenario-based indexing:** Uses complete frames from a selected reference agent (agent `1` by default); includes other agents when their matching frame files exist.
- **Binary LiDAR:** Reads four values per point: `x, y, z, intensity`.
- **Y-axis reflection:** Native CARLA/OpenCOOD coordinates use **Y right**, while our canonical convention uses **Y left**. The adapter mirrors Y for points and correctly changes relative pose transforms as well.
- **Native pose formula:** Uses OpenCOOD's roll/yaw/pitch convention instead of assuming a standard angle order.
- **Deduplicated annotations:** Combines objects seen by different agents using object IDs, prioritizing the reference agent so the same object is not counted repeatedly. Converts half-size extents to full box dimensions.

**Current limits:** Camera paths and calibration are retained as *native metadata*, not published as canonical `CameraData`, because the camera-coordinate conversion still needs projection verification. Frame time is estimated from a configurable frame rate (default 10 Hz), not read as an independently measured timestamp. Some native metadata is passed under `metadata` rather than the schema's `extras` field, so those particular details are not retained by the current schema-construction helper.

## 4. `tumtraf_adapter.py` — Keep the two real agents separate

**Core idea:** Read the TUMTraf mini dataset's OpenLABEL JSON and ASCII PCD point clouds, producing an infrastructure agent and a vehicle agent.

- **Infrastructure as reference:** Uses the Ouster LiDAR coordinate frame as the shared reference. The OpenLABEL transformation places the vehicle RoboSense LiDAR in that frame.
- **Read exact filenames from annotations:** Avoids guessing synchronization from filenames or assuming the two sensors have identical timestamps.
- **Handle PCD carefully:** Reads XYZ and intensity, filters invalid points consistently, and preserves other original point fields in `extras`.
- **Convert 3D boxes:** Turns OpenLABEL cuboid quaternions into canonical yaw and keeps object categories and original IDs.
- **Do not invent a third agent:** The registered/fused point cloud is a processed combination, not another physical participant; its path is retained separately.

**Current limits:** The mini data does not provide verified camera calibration, so only image paths are stored. OpenLABEL object IDs are preserved in `extras`; canonical integer track IDs use `-1` rather than inventing IDs.

## 5. `deepaccident_adapter.py` — Preserve metadata without claiming unverified conversions

**Core idea:** Read the official converted-info `.pkl` file, group records by **scene and frame**, and represent participating vehicles/infrastructure as one multi-agent sample.

- **Group agents by frame:** The info file has separate agent records; grouping creates the cooperative V2X view needed by our schema.
- **Reference agent:** Uses `ego_vehicle` by default and skips frames where it is absent.
- **Relative poses:** Computes each agent's transform relative to the ego vehicle. CARLA-style Y-axis conversion is currently an explicit, **provisional assumption**, not a fully verified calibration result.
- **Portable file paths:** Attempts to rebase native paths when metadata was generated on a different computer.
- **Keep uncertain data in `extras`:** Preserves native camera information, LiDAR paths, and per-agent annotations without incorrectly labeling them as verified canonical data.

**Current limits:** Canonical LiDAR, cameras, and boxes are intentionally **not populated yet**. Frame indices are not seconds; without a verified frame rate, `timestamp_s` is a placeholder `0.0`. Only open `.pkl` files from trusted sources because Python pickle loading can execute code.

## The main design principle

All adapters share **the same interface and output structure**, but they do **not** currently offer equal levels of converted sensor data. When a native convention is understood, we convert it. When calibration or geometry is uncertain, we preserve the original information in `extras` and mark the limitation instead of silently inventing a conversion.

**In short:** `schema.py` defines **what a sample looks like**; `BaseAdapter` defines **how we access it**; the four dataset adapters define **how each original dataset is translated**.
