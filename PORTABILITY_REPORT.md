# Portability refactor report

Completed in the existing workspace on 2026-10-13. No raw dataset contents were
changed, moved, renamed or deleted. `schema.py`, `adapters/urbaning_adapter.py`
and `adapters/tumtraf_adapter.py` were left unchanged. DeepAccident was not added.

## Files changed

Paths are relative to the repository root. New files are marked **new**.

| Exact path | Modification |
| --- | --- |
| `adapters/base_adapter.py` | Resolve constructor-relative roots against the project root using pathlib. |
| `adapters/v2xpnp_adapter.py` | Resolve roots consistently; discover a scenario root itself as well as nested scenarios. Preserve constructor defaults, transforms and parsing. |
| `adapters/__init__.py` | **new**: explicit adapter package without eager devkit imports. |
| `project_paths.py` | **new**: YAML loading, project-relative/absolute roots, CLI overrides, missing-path errors and output creation. |
| `configs/dataset_paths.yaml` | **new**: requested default dataset roots and outputs root. |
| `tests/__init__.py` | **new**: explicit test package. |
| `tests/cli.py` | **new**: shared CLI options, available UrbanIng reference selection and metadata-based complete V2XPnP pair selection. |
| `tests/conftest.py` | **new**: dataset-root/config/output pytest options, default-data skips and headless plotting. |
| `tests/test_portability.py` | **new**: 12 dataset-free checks including parametrized invalid configurations. |
| `tests/test_tumtraf_adapter.py` | Replace former machine-specific root with YAML/CLI selection; preserve sample checks without fixed agent-count assertions; add pytest entry. |
| `tests/test_urbaning.py` | Move execution into main; configure sequence, sample and reference agent; fix project-root assumption; add pytest entry. |
| `tests/test_urbaning_geometry.py` | Preserve geometry helpers and plots; configure sample/root/output and optional interactive display; no import-time execution; add pytest entry. |
| `tests/test_urbaning_multiframe.py` | Configure roots, sequences and explicit indices; spread default indices across available frames; compare relative timestamps only within a sequence; add pytest entry. |
| `tests/test_v2xpnp_adapter.py` | Move execution into main; configure root, reference, split, scenario/frame and index; use existing automatic reference selection by default in CLI; add pytest entry. |
| `tests/test_v2xpnp_geometry.py` | Remove fixed scenario/frame/agent constants; select via CLI/native metadata; pass context explicitly to pose loader; add pytest entry. |
| `tests/test_v2xpnp_lidar_alignment.py` | Fix package import, configure selection/output, pass loader context explicitly and retain existing math/plots; add pytest entry. |
| `tests/test_v2xpnp_overlap_alignment.py` | Fix package import, configure selection/output and explicit loader/frame context; preserve occupancy diagnostics/plots; add pytest entry. |
| `scripts/__init__.py` | **new**: explicit scripts package. |
| `scripts/download_urbaning_test_data.py` | Add import-safe main and configurable destination/config/sequence; preserve official downloading routine. |
| `requirements.txt` | **new**: NumPy, SciPy, PyYAML, matplotlib, Pillow, official UrbanIng devkit and pytest. |
| `pyproject.toml` | **new**: editable repository installation, Python >=3.10, package/module discovery and pytest configuration. |
| `.gitignore` | **new**: ignore raw data, virtual environments, caches, generated outputs, build artifacts and local configs. |
| `README.md` | **new**: full structure, installation, paths, adapter usage, tests, geometry scripts, downloader and limitations. |
| `PORTABILITY_REPORT.md` | **new**: this audit and verification record. |

## Hardcoding and fragile assumptions removed

The pre-edit audit identified these issues:

- TUMTraf smoke script default `D:\TUM\data`.
- Test-module `Path(__file__).parent` treated as the repository root after files
  moved into `tests/`, producing incorrect dataset and plot destinations.
- Working-directory-relative UrbanIng download destination and unconditional
  download on import.
- UrbanIng smoke/geometry and V2XPnP smoke workflows executed during import.
- V2XPnP alignment imports depended on root-level sibling modules and imported
  fixed scenario/frame/agent constants.
- Fixed UrbanIng sequence `20241126_0017_crossing1_00`, V2XPnP scenario
  `2023-03-17-16-12-12_3_0`, frame `000001`, agent IDs `1`/`-1`, and fixed
  UrbanIng multiframe indices ending at 199.
- V2XPnP smoke always read index 1 and required agent 1; CLI now selects index 0
  and passes None to the existing automatic-reference option unless overridden.
- TUMTraf smoke asserted a universal two-agent count.
- Geometry outputs went into source directories; outputs now use YAML/CLI paths.
- No package initialization, dependency list, README or ignore rules existed.
- UrbanIng temporal validation could compare sequence-relative timestamps across
  distinct sequences; it now compares within each sequence.

The final AST audit parsed all 21 project-owned Python files and found no
machine-specific absolute paths or `sys.path` workarounds. Raw datasets, `.venv`,
external packages, caches and generated output directories were excluded.
Old selection identifiers remain only in usage examples or synthetic fixtures,
where they reproduce an explicit scenario or exercise native parsing.

## Imports and configuration

Alignment scripts now import helpers from `tests.test_v2xpnp_geometry`.
Their helpers receive the selected scenario, frame and expected metadata type
explicitly; there are no imported mutable scenario/frame constants. Existing
`from schema ...` and `from adapters...` imports remain valid.

`PROJECT_ROOT` comes from `project_paths.py`'s location. All relative config,
dataset and output paths resolve against this root, independent of process cwd.
Absolute external paths are accepted unchanged apart from normalization.
`--root` and `--output` override their corresponding YAML entries. Missing
configured roots raise a clear error without searching another dataset location.
Constructors keep their public signatures; relative constructor paths now follow
this documented project-relative policy. Use an absolute path for a location
relative to an external caller's cwd.

Editable installation makes module execution work from other directories.
Default YAML remains exactly as requested; the existing `datasets/TumTraf`
folder was not renamed or silently substituted for `datasets/TUMTraf-V2X`.

## Verification actually performed

Commands used the workspace virtual environment's Python.

| Command/check | Actual result |
| --- | --- |
| `python -m compileall -q schema.py project_paths.py adapters tests scripts` | Passed, exit 0, all project Python sources compile. |
| `python -m pytest tests/ --tumtraf-root datasets/TumTraf -q` | 20 passed in 14.42 seconds. Includes all eight existing dataset validation workflows and 12 portability checks. |
| `python -m pytest tests/ -q` after final CLI changes | 19 passed, 1 skipped in 14.59 seconds. TUMTraf skipped because the requested default `datasets/TUMTraf-V2X` does not exist; no fallback was used. |
| From `scripts/`, `python -m pytest <absolute-repository-tests-path> --tumtraf-root datasets/TumTraf -q` | 20 passed in 14.69 seconds. Demonstrates full-suite independence from the terminal cwd. |
| `python -m tests.test_tumtraf_adapter --root datasets/TumTraf` | Passed. Actual sample had 2 agents, 27 boxes, 131072 infrastructure points and 43730 valid vehicle points; none of these counts are universal assertions. |
| `python -m tests.test_v2xpnp_geometry` | Passed on automatically selected frame 000000 with vehicle 1 and infrastructure -2. |
| `python -m tests.test_v2xpnp_geometry --scenario 2023-03-17-16-12-12_3_0 --frame 000001 --vehicle-id 1 --infra-id -1` | Passed, reproducing the previous explicit selection. |
| Dataset-free external-root regression | Synthetic V2XPnP data outside the repository loaded through both parent/scenario roots; three non-default agent IDs supported; canonical reflected coordinates retained. |
| Different-cwd subprocess checks | All project modules imported without executing workflows; all nine CLI help commands succeeded; geometry CLI loaded an external root with explicit non-default IDs. |
| PCD regression | Nonfinite XYZ/intensity rows were removed with the same mask for XYZ, intensity and extra ring attributes. |
| Downloader regression | Import caused no download; mocked official function received the exact configured destination and sequence. |
| AST source audit | 21 owned Python files parsed; no machine-specific path or sys.path findings. |

Pytest, wheel and the local editable project were installed in the existing virtual
environment to perform these checks. Installation first encountered restricted
network/filesystem access and was completed with approved escalation. An initial
suite run found a plot-title reference to the removed global FRAME; it was fixed
and the final suites passed. Generated integration plots are under `outputs/pytest/`.

No real UrbanIng download was performed; server availability, curl execution and
archive extraction were not end-to-end tested. The existing official URLs and
routine were preserved. No Linux/macOS runtime was available; those platforms
were not tested. Existing Windows UrbanIng, V2XPnP and TUMTraf data were available
and their sample workflows were exercised.

## Remaining limitations and necessary native assumptions

- DeepAccident remains unimplemented; no placeholder adapter was created.
- TUMTraf's existing mini/OpenLABEL, named sensor directories and ASCII PCD
  constraints remain. Its camera paths stay in extras without verified calibration.
- UrbanIng uses official devkit directory/field conventions and retains its
  constructor's `vehicle1` default; CLI selection is configurable/automatic.
- V2XPnP retains native numeric agent directories, metadata keys, float32 Nx4
  binaries, transformation conventions and nominal frame-rate behavior. Its
  constructor retains default reference ID `"1"` for compatibility; the CLI uses
  its existing None option for automatic selection.
- V2XPnP camera construction already lacks required mappings for the current
  `CameraData` schema. A frame with matching camera images/metadata may fail;
  camera projection support was not claimed or repaired using guessed calibration.
- V2XPnP geometry scripts inspect a vehicle/infrastructure pair in native OpenCOOD
  coordinates. Matrix consistency and occupancy plots cannot prove physical
  registration or canonical handedness. Camera and box-overlay validation remain
  incomplete, as does cross-dataset class taxonomy.
- Plot ranges, deterministic sampling limits, numerical tolerances and occupancy
  grid parameters remain diagnostic constants. They do not require specific frame
  counts, agent IDs or point counts.
- Packaging is intended for editable use of a copied repository. Linux/macOS
  devkit dependencies and external curl/7-Zip availability need platform checks.

## User actions

1. Use `--root datasets/TumTraf` / pytest `--tumtraf-root datasets/TumTraf`, or update
   `datasets.tumtraf` in YAML to the current folder. No dataset movement is required.
2. On another computer, recreate a virtual environment, install
   `requirements.txt`, then run `python -m pip install -e . --no-deps` in the copied
   repository. Do not transfer the existing virtual environment.
3. Provide extracted native datasets at the configured roots, or pass absolute
   external directories. Use `--config` / pytest `--dataset-config` for a local YAML.
4. Inspect generated plots for physical alignment; automatic numerical checks do
   not replace that inspection.
