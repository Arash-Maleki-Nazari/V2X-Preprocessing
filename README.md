# Cooperative V2X Dataset Preprocessing

## Purpose

This master's thesis project provides a **common way to access data from different cooperative vehicle-to-everything (V2X) datasets**.

V2X datasets contain information from vehicles and roadside infrastructure, such as LiDAR point clouds, camera images, sensor positions, and object annotations. However, each dataset stores this information in different files, structures, and coordinate systems.

The goal is to **read these different formats and represent their data in one shared structure**, regardless of the source dataset. This makes it easier to build analysis, visualization, and machine-learning code that can work across datasets without needing a separate data-reading procedure for each one.

The four datasets covered are **UrbanIng-V2X, V2XPnP, TUMTraf-V2X, and DeepAccident**.

## How the project works

The project has three main conceptual layers:

| Layer | Responsibility |
| --- | --- |
| **1. Native datasets** | Provide the original sensor data, annotations, and metadata in their own formats. |
| **2. Dataset adapters** | Understand each dataset's format, read its files, and convert the supported information into the common representation. |
| **3. Canonical schema** | Defines the structure that every adapter must return and the validation rules that structure must satisfy. |

The **canonical schema defines the target format**; it is not a separate file-conversion step. At runtime, data flows in this order:

```text
Original V2X dataset
        |
        v
Dataset-specific adapter
        |
        v
CanonicalV2XSample (defined in schema.py)
        |
        v
Structural validation
        |
        v
Downstream use (analysis, visualization, or model training)
```

All dataset adapters inherit a shared interface from `BaseAdapter`, so their samples can be accessed in the same way, for example `adapter[0]`. The adapter loads and converts the native data; the schema describes the expected result; and validation checks that the returned sample follows that structure.

**The original dataset files are left unchanged.** If a sensor or conversion has not yet been verified, the adapter can preserve the original information as metadata rather than incorrectly presenting it as fully standardized data.

## Why each dataset needs its own adapter

Although all four datasets contain cooperative V2X information, they differ in file formats, data organization, coordinate conventions, and metadata. Each dataset-specific adapter handles these differences and converts the supported information into the shared `CanonicalV2XSample` structure.

| Dataset | What is different? | What its adapter does |
| --- | --- | --- |
| **UrbanIng-V2X** | Data is organized into synchronized sequences and accessed through a dedicated devkit. | Uses the devkit to load synchronized sensor observations and their calibrations. |
| **V2XPnP** | Uses a different coordinate convention and organizes data by scenarios, agents, and frames. | Reads the scenario structure and converts coordinates into the canonical convention. |
| **TUMTraf-V2X** | Uses OpenLABEL for annotations and PCD files for LiDAR. | Parses these formats and transforms vehicle observations relative to the infrastructure reference. |
| **DeepAccident** | Provides converted metadata with records for different agents and frames. | Groups records into cooperative samples and preserves information whose conversion is not yet verified. |

All four adapters implement the same `BaseAdapter` interface, allowing downstream code to access their samples consistently despite differences in the original datasets. Some conversions, particularly in DeepAccident, remain provisional.

## Project structure

```text
Preprocessing/
├── adapters/
│   ├── adapters_guidance.md      # Short guide to adapter design
│   ├── base_adapter.py           # Shared adapter interface
│   ├── urbaning_adapter.py       # UrbanIng-V2X reader and converter
│   ├── v2xpnp_adapter.py         # V2XPnP reader and converter
│   ├── tumtraf_adapter.py        # TUMTraf-V2X reader and converter
│   └── deepaccident_adapter.py   # DeepAccident metadata adapter
│
├── configs/
│   └── dataset_paths.yaml        # Dataset and output locations
│
├── tests/
│   ├── tests_guidance.md         # Short guide to the tests
│   ├── cli.py                    # Shared test command-line helpers
│   ├── conftest.py               # Shared pytest setup
│   ├── test_urbaning.py
│   ├── test_urbaning_geometry.py
│   ├── test_urbaning_multiframe.py
│   ├── test_v2xpnp_adapter.py
│   ├── test_v2xpnp_geometry.py
│   ├── test_v2xpnp_lidar_alignment.py
│   ├── test_v2xpnp_overlap_alignment.py
│   ├── test_tumtraf_adapter.py
│   ├── test_deepaccident_adapter.py
│   └── test_portability.py
│
├── scripts/                       # Supporting utilities
├── schema.py                      # Common sample structure and validation
├── project_paths.py               # Portable path handling
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Python project configuration
└── README.md                      # Project overview
```

## Main folders at a glance

- **`adapters/`** — Reads each dataset in its original format and returns samples in the shared structure.
- **`configs/`** — Stores dataset and output paths so the code can run with different file locations.
- **`tests/`** — Checks that reading, conversion, and validation behave as expected; some tests create plots for manual inspection.
- **`scripts/`** — Contains supporting utilities, such as downloading test data.
