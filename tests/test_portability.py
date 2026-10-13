"""Dataset-free regression checks for paths, imports and native selections."""
import importlib
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from project_paths import PROJECT_ROOT, dataset_path, output_path, resolve_path
from adapters.v2xpnp_adapter import V2XPnPAdapter
from adapters.tumtraf_adapter import read_ascii_pcd
from tests.cli import parser_for, select_v2xpnp


def test_relative_configuration_is_project_relative(tmp_path, monkeypatch):
    config = tmp_path / "paths.yaml"
    config.write_text("datasets:\n  v2xpnp: datasets/V2XPnP\noutputs:\n  root: outputs\n")
    monkeypatch.chdir(tmp_path)
    assert dataset_path("v2xpnp", config=config, must_exist=False) == PROJECT_ROOT / "datasets/V2XPnP"
    assert resolve_path("outputs") == PROJECT_ROOT / "outputs"


def test_absolute_external_root_and_cli_override(tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    config = tmp_path / "paths.yaml"
    config.write_text(yaml.safe_dump({"datasets": {"v2xpnp": "missing-dataset"}}))
    assert dataset_path("v2xpnp", external, config) == external
    with pytest.raises(FileNotFoundError, match="directory does not exist"):
        dataset_path("v2xpnp", config=config)


@pytest.mark.parametrize("text", ["[]", "datasets: {}", "datasets: {v2xpnp: null}"])
def test_invalid_configuration_reports_error(tmp_path, text):
    config = tmp_path / "invalid.yaml"
    config.write_text(text)
    with pytest.raises(ValueError):
        dataset_path("v2xpnp", config=config)


def test_output_configuration_creates_parents(tmp_path):
    destination = tmp_path / "nested" / "plots"
    config = tmp_path / "paths.yaml"
    config.write_text(yaml.safe_dump({"outputs": {"root": str(destination)}}))
    assert output_path(config=config) == destination
    assert destination.is_dir()


@pytest.fixture
def external_v2xpnp(tmp_path):
    root = tmp_path / "external_data"
    # IDs intentionally differ from the former 1/-1 pair. Metadata is authoritative.
    for agent, infra in (("7", False), ("-9", True), ("8", False)):
        folder = root / "scenario" / agent
        folder.mkdir(parents=True)
        (folder / "000042.yaml").write_text(yaml.safe_dump({
            "infra": infra, "lidar_pose": [0, 0, 0, 0, 0, 0], "vehicles": {}}))
        np.array([[1, 2, 3, 4]], dtype="<f4").tofile(folder / "000042.bin")
    return root


def test_native_selection_and_external_adapter(external_v2xpnp, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    args = parser_for("v2xpnp").parse_args(["--root", str(external_v2xpnp)])
    scenario, frame, vehicle, infra = select_v2xpnp(args)
    assert (frame, vehicle, infra) == ("000042", "7", "-9")
    for root in (external_v2xpnp, scenario):
        adapter = V2XPnPAdapter(root, reference_agent_id=None)
        sample = adapter[0]
        assert len(sample.agents) == 3
        assert sample.reference_agent_id == "7"
        np.testing.assert_array_equal(sample.agents[0].lidar.xyz, [[1, -2, 3]])
    args.frame = "000001"
    with pytest.raises(ValueError, match="No complete"):
        select_v2xpnp(args)


def test_pcd_nonfinite_filter_preserves_all_attributes(tmp_path):
    path = tmp_path / "cloud.pcd"
    path.write_text("FIELDS x y z intensity ring\nPOINTS 4\nDATA ascii\n"
                    "1 2 3 10 1\nnan 2 3 20 2\n4 5 6 nan 3\n7 8 9 40 4\n")
    lidar = read_ascii_pcd(path)
    np.testing.assert_array_equal(lidar.xyz, [[1, 2, 3], [7, 8, 9]])
    np.testing.assert_array_equal(lidar.intensity, [10, 40])
    np.testing.assert_array_equal(lidar.extras["pcd_fields"]["ring"], [1, 4])
    assert lidar.extras["num_invalid_points_removed"] == 2


def test_imports_from_other_directory_have_no_execution(tmp_path):
    modules = ["schema", "adapters.base_adapter", "adapters.urbaning_adapter",
               "adapters.tumtraf_adapter", "adapters.v2xpnp_adapter",
               "scripts.download_urbaning_test_data"] + [
        f"tests.{path.stem}" for path in (PROJECT_ROOT / "tests").glob("test_*.py")]
    code = "import importlib; " + "; ".join(
        f"importlib.import_module({name!r})" for name in modules)
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert list(tmp_path.iterdir()) == []


def test_all_module_cli_help_from_other_directory(tmp_path):
    modules = [f"tests.{path.stem}" for path in (PROJECT_ROOT / "tests").glob("test_*.py")
               if path.stem != "test_portability"] + ["scripts.download_urbaning_test_data"]
    for module in modules:
        result = subprocess.run([sys.executable, "-m", module, "--help"], cwd=tmp_path,
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert "--root" in result.stdout


def test_geometry_cli_external_root_from_other_directory(tmp_path, external_v2xpnp):
    result = subprocess.run([
        sys.executable, "-m", "tests.test_v2xpnp_geometry", "--root", str(external_v2xpnp),
        "--scenario", "scenario", "--frame", "000042", "--vehicle-id", "8", "--infra-id", "-9"
    ], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Vehicle (8)" in result.stdout


def test_downloader_destination_is_explicit_and_import_is_safe(tmp_path):
    with patch("urbaning.data.download_one_sequence") as download:
        module = importlib.import_module("scripts.download_urbaning_test_data")
        importlib.reload(module)
        download.assert_not_called()
        destination = tmp_path / "download"
        module.main(["--destination", str(destination), "--sequence", "example"])
        download.assert_called_once_with(download_dir=str(destination), sequence_name="example")
        assert destination.is_dir()
