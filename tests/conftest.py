from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


def _artifact_dir(config: pytest.Config) -> Path:
    """Return the directory that holds generated visualization artifacts.

    The path is anchored at the rootdir rather than the current working directory so artifacts land in the same place
    regardless of where pytest was started from.

    Args:
        config: Active pytest configuration.

    Returns:
        Directory for generated artifacts.
    """
    return config.rootpath / "output"


def pytest_configure(config: pytest.Config) -> None:
    """Ensure common output directory exists for generated GIFs.

    Args:
        config: Active pytest configuration.
    """
    _artifact_dir(config).mkdir(exist_ok=True)


@pytest.fixture(scope="session")
def output_dir(pytestconfig: pytest.Config) -> Path:
    """Return the output directory path, creating it if missing.

    Args:
        pytestconfig: Active pytest configuration.

    Returns:
        Directory for generated artifacts.
    """
    d: Path = _artifact_dir(pytestconfig)
    d.mkdir(exist_ok=True)
    return d
