from __future__ import annotations

import sys
import sysconfig
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    __version__ = version("llmfit")
except PackageNotFoundError:
    __version__ = "unknown"


class LlmfitError(Exception):
    """Base class for llmfit exceptions."""


class BinaryNotFoundError(FileNotFoundError, LlmfitError):
    """Exception raised when the llmfit binary cannot be found."""

    def __init__(self, candidate: Path) -> None:
        super().__init__(
            f"llmfit binary not found at {candidate}. This may indicate a corrupt or incomplete installation."
        )


def find_llmfit_bin() -> Path:
    """Return the path to the llmfit binary installed with this package."""
    bin_name = "llmfit.exe" if sys.platform == "win32" else "llmfit"
    candidate = Path(sysconfig.get_path("scripts")) / bin_name
    if not candidate.is_file():
        raise BinaryNotFoundError(candidate)
    return candidate
