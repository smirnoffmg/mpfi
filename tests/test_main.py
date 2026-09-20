"""Tests for the main module."""

import pytest

from pypi_package_uv_template import __version__, main


def test_version() -> None:
    """Check that version is defined."""
    assert __version__ == "0.1.0"


def test_main(capsys: pytest.CaptureFixture[str]) -> None:
    """Check that main prints the expected message."""
    main()
    captured = capsys.readouterr()
    assert "Hello from pypi-package-uv-template!" in captured.out
