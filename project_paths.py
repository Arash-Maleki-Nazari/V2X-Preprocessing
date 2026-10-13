"""Project-relative paths, independent of the process working directory."""
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "dataset_paths.yaml"


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def load_config(config: str | Path | None = None) -> dict:
    path = resolve_path(config) if config is not None else DEFAULT_CONFIG
    with path.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, dict):
        raise ValueError(f"Expected a mapping in configuration: {path}")
    return values


def dataset_path(name: str, root: str | Path | None = None,
                 config: str | Path | None = None, *, must_exist: bool = True) -> Path:
    if root is None:
        try:
            root = load_config(config)["datasets"][name]
        except (KeyError, TypeError) as error:
            raise ValueError(f"Missing datasets.{name} in configuration") from error
    if not isinstance(root, (str, Path)) or not str(root).strip():
        raise ValueError(f"Invalid dataset path for {name}: {root!r}")
    path = resolve_path(root)
    if must_exist and not path.is_dir():
        raise FileNotFoundError(
            f"Dataset {name!r} directory does not exist: {path}. "
            "Set --root or update configs/dataset_paths.yaml."
        )
    return path


def output_path(output: str | Path | None = None,
                config: str | Path | None = None) -> Path:
    if output is None:
        try:
            output = load_config(config)["outputs"]["root"]
        except (KeyError, TypeError) as error:
            raise ValueError("Missing outputs.root in configuration") from error
    if not isinstance(output, (str, Path)) or not str(output).strip():
        raise ValueError(f"Invalid output directory: {output!r}")
    path = resolve_path(output)
    path.mkdir(parents=True, exist_ok=True)
    return path
