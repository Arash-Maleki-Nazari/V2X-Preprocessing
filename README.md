# Cooperative V2X Dataset Preprocessing

## Purpose

This project is part of a master's thesis on **cooperative vehicle-to-everything (V2X) perception**.

V2X datasets contain information collected by vehicles and roadside infrastructure, such as LiDAR point clouds, camera images, sensor positions, and object annotations. However, each dataset stores and organizes this information differently.

**The goal is to make these datasets accessible through one common structure, regardless of which dataset the data comes from.** Instead of writing different processing code for every dataset, we can access them in the same way and use their standardized output for later analysis, visualization, and machine-learning experiments.

The project currently works with four datasets: **UrbanIng-V2X, V2XPnP, TUMTraf-V2X, and DeepAccident**.

## How the project works

The design has **three conceptual layers**:

| Layer | Responsibility |
| --- | --- |
| **1. Native datasets** | The original data, stored in each dataset's own files, formats, and coordinate systems. |
| **2. Dataset adapters** | Read the original files and translate their information into our shared representation. Each dataset has its own adapter because its native format is different. |
| **3. Canonical schema** | Defines the common output structure (`CanonicalV2XSample`), including agents, available sensors, coordinate transforms, and annotations. It also checks that samples follow the expected structure. |

In short:

**Original dataset → Dataset-specific adapter → Common V2X sample → Later use**

The original dataset files are not changed. Each adapter presents their information through the same interface, so later code does not need to understand every native file format.

## Project structure

| File or directory | Purpose |
| --- | --- |
| `schema.py` | Defines the common V2X data structure and its validation rules. |
| `adapters/base_adapter.py` | Defines how all dataset adapters must be accessed. |
| `adapters/urbaning_adapter.py` | Reads and converts UrbanIng-V2X data. |
| `adapters/v2xpnp_adapter.py` | Reads and converts V2XPnP data. |
| `adapters/tumtraf_adapter.py` | Reads and converts TUMTraf-V2X data. |
| `adapters/deepaccident_adapter.py` | Reads DeepAccident metadata and builds provisional canonical samples. |
| `project_paths.py` and `configs/` | Manage dataset locations without hard-coded machine-specific paths. |
| `tests/` | Check sample structure, data loading, coordinate calculations, and portability. |
| `scripts/` | Contains supporting utilities. |
| `requirements.txt` and `pyproject.toml` | Define the project's Python dependencies and configuration. |
| `ADAPTERS_EXPLAINED.md` | Briefly explains the reasoning behind each adapter. |
| `TESTS_EXPLAINED.md` | Explains what the tests check and the limits of their results. |

## What has been developed

- A **shared V2X schema** so different datasets can return the same type of sample.
- A **common adapter interface** so samples can be accessed consistently.
- **Four dataset adapters** to handle the differences between the original datasets.
- **Validation and test scripts** to check data structure, geometry, and project portability.

**Current scope:** The common interface is established, but not every sensor or annotation has been fully converted and physically verified. Where conversion is uncertain, the original information is preserved rather than presented as validated canonical data. Further real-data and cross-dataset validation is still needed before relying on the outputs for model training.
