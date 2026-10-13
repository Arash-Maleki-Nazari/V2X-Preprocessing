# Cooperative V2X Dataset Preprocessing

This repository is the **data preparation and validation layer for a master's thesis on cooperative vehicle-to-everything (V2X) perception**. Its purpose is to make different vehicle–infrastructure datasets accessible through **one consistent Python interface**, so subsequent experiments can use the same sample structure instead of implementing a separate loader for every dataset.

The adapters read the original datasets **without modifying the raw files** and return a `CanonicalV2XSample` defined in `schema.py`. Each sample contains its source and timestamp information, participating agents, LiDAR data, available camera information, agent-to-reference transforms, and 3D object annotations. The canonical convention is **x forward, y left, z up**, with 3D boxes represented as `[x, y, z, length, width, height, yaw]`.

## Dataset support

| Dataset | Status | Scope |
| --- | --- | --- |
| UrbanIng-V2X | Implemented | Vehicle/infrastructure samples using the official devkit |
| V2XPnP | Implemented | Multi-agent LiDAR, poses and 3D annotations; native camera data retained |
| TUMTraf-V2X | Implemented | Supported mini/OpenLABEL layout and ASCII PCD point clouds |
| DeepAccident | Planned | Adapter not implemented yet |

**Important:** A shared schema does not, by itself, prove correct physical alignment across datasets. Camera projection, 3D box overlays and real-world coordinate alignment still need independent validation. V2XPnP camera calibration is preserved in `agent.extras["native_cameras"]` rather than converted into unverified canonical camera transforms.

## Setup

Use **Python 3.10+**. From the repository root:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Activate the virtual environment first (`.venv\Scripts\Activate.ps1` on Windows PowerShell or `source .venv/bin/activate` on Linux/macOS). Dataset files are **not included** in the repository.

## Configure dataset paths

Edit `configs/dataset_paths.yaml` to point to your local datasets:

```yaml
datasets:
  urbaning: datasets/UrbanIng-V2X
  v2xpnp: datasets/V2XPnP
  tumtraf: datasets/TumTraf
  deepaccident: datasets/DeepAccident
outputs:
  root: outputs
```

These are examples, not required directory names. Relative paths resolve from the **repository root**, not the terminal's working directory; absolute paths are also supported. You can override a path for individual commands using `--root`, without changing the YAML file.

## Use the adapters

```python
from project_paths import dataset_path
from adapters.v2xpnp_adapter import V2XPnPAdapter

adapter = V2XPnPAdapter(dataset_path("v2xpnp"), reference_agent_id=None)
sample = adapter[0]
sample.validate()

print(sample.source.sequence_id, sample.source.frame_id)
print(sample.reference_agent_id, len(sample.agents))
print(sample.annotations.boxes_3d.shape)
```

The equivalent classes are `UrbanIngAdapter` in `adapters/urbaning_adapter.py` and `TUMTrafAdapter` in `adapters/tumtraf_adapter.py` (use `split="train"` for TUMTraf). Adapters load canonical samples **on demand**; they do not automatically export a new dataset.

For quick dataset checks and geometric diagnostics:

```bash
python -m tests.test_urbaning
python -m tests.test_v2xpnp_adapter
python -m tests.test_tumtraf_adapter --split train
python -m tests.test_v2xpnp_lidar_alignment --output outputs/v2xpnp
```

For an external dataset location, append `--root /path/to/dataset`. Other geometry and multiframe checks are in `tests/`.

## Testing and project structure

```bash
python -m pytest tests/ -q
```

**Latest local result: 35 passed, 0 skipped, 0 failed (13 October 2026).** This establishes that the current automated checks pass; it is not a substitute for physical alignment validation. If a dataset is unavailable at its configured path, some integration tests may skip.

- `schema.py` — shared canonical sample definition and validation
- `adapters/` — dataset-specific readers and transformations
- `project_paths.py` and `configs/` — portable path configuration
- `tests/` — schema, integration, geometry and portability checks
- `scripts/` — supporting utilities, including an UrbanIng test-data downloader
- `outputs/` — generated diagnostics; `datasets/` — locally supplied raw data

The next development steps are **cross-dataset geometric validation, consistent class mapping and the DeepAccident adapter**.
