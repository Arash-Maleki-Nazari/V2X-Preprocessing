"""Dataset fixtures: unavailable defaults skip; explicit missing paths fail."""
import matplotlib
matplotlib.use("Agg")
import pytest

from project_paths import dataset_path, output_path


def pytest_addoption(parser):
    parser.addoption("--dataset-config", default=None)
    for name in ("urbaning", "v2xpnp", "tumtraf"):
        parser.addoption(f"--{name}-root", default=None)
    parser.addoption("--validation-output", default=None)


@pytest.fixture
def cli_args(request):
    def arguments(dataset):
        config = request.config.getoption("--dataset-config")
        root = request.config.getoption(f"--{dataset}-root")
        try:
            resolved = dataset_path(dataset, root, config)
        except FileNotFoundError:
            if root is not None or config is not None:
                raise
            pytest.skip(f"Default {dataset} dataset unavailable; supply --{dataset}-root")
        output = output_path(request.config.getoption("--validation-output"), config)
        argv = ["--root", str(resolved)]
        if config:
            argv += ["--config", config]
        if dataset != "tumtraf":
            argv += ["--output", str(output / "pytest")]
        return argv
    return arguments
