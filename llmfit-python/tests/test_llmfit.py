"""Tests for the llmfit Python module (llmfit-python/src/llmfit/)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

import llmfit
from llmfit import find_llmfit_bin


def test_find_llmfit_bin_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that find_llmfit_bin raises a BinaryNotFoundError when the binary is missing."""
    monkeypatch.setattr(Path, "is_file", lambda _: False)
    with pytest.raises(llmfit.BinaryNotFoundError):
        find_llmfit_bin()


def test_binary_path_is_path() -> None:
    """Tests that find_llmfit_bin returns a Path object."""
    assert isinstance(find_llmfit_bin(), Path)


def test_binary_runs() -> None:
    """Tests that the llmfit binary runs successfully."""
    result = subprocess.run([find_llmfit_bin(), "--help"], capture_output=True, check=False)
    assert result.returncode == 0


def test_version() -> None:
    """Tests that llmfit.__version__ is a valid semantic version."""
    assert re.match(r"^\d+\.\d+\.\d+$", llmfit.__version__) is not None
