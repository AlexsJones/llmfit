"""
hatch_build.py  -  Hatchling build hook for llmfit.

Injects the pre-built llmfit binary into each wheel via ``shared_scripts`` so
that the installer places it in the environment's scripts directory (e.g.
``.venv/bin/llmfit``).  Also overrides the wheel platform tag so that wheels
built on different CI runners get the correct platform-specific name.

For editable installs (``uv sync``, ``uv run``), the locally compiled debug
binary (from ``cargo build``) is used.  For release installs (``uv build``),
the release binary (from ``cargo build --release``) is used.

Release builds also support setting ``LLMFIT_PYTHON_PLATFORM_TAG``, in which
case the release binary for the corresponding Rust target is used instead
(e.g. after ``cargo build --release --target aarch64-unknown-linux-gnu``).

Environment variables
---------------------
LLMFIT_PYTHON_PLATFORM_TAG
    Wheel platform tag to target (e.g. ``manylinux_2_17_x86_64``).
    Not supported for editable installs.
LLMFIT_VERSION
    Override the version read from ``Cargo.toml`` (e.g. ``0.9.8``).
    Supported for both release and editable installs.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import tomli
from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from hatchling.metadata.plugin.interface import MetadataHookInterface
from packaging.tags import sys_tags

# wheel_platform_tag -> (upstream_target, binary_name)
TARGET_CONFIGS: dict[str, tuple[str, str]] = {
    "manylinux_2_17_x86_64": ("x86_64-unknown-linux-gnu", "llmfit"),
    "manylinux_2_17_aarch64": ("aarch64-unknown-linux-gnu", "llmfit"),
    "musllinux_1_2_x86_64": ("x86_64-unknown-linux-musl", "llmfit"),
    "musllinux_1_2_aarch64": ("aarch64-unknown-linux-musl", "llmfit"),
    "macosx_10_12_x86_64": ("x86_64-apple-darwin", "llmfit"),
    "macosx_11_0_arm64": ("aarch64-apple-darwin", "llmfit"),
    "win_amd64": ("x86_64-pc-windows-msvc", "llmfit.exe"),
    "win_arm64": ("aarch64-pc-windows-msvc", "llmfit.exe"),
}


class LlmfitMetadataHook(MetadataHookInterface):
    """Hatchling metadata hook that sets version and license-expression dynamically."""

    PLUGIN_NAME = "llmfit version and license"

    def update(self, metadata: dict) -> None:
        """Populate ``version`` and ``license-expression`` from ``Cargo.toml``.

        Version resolution order:

        1. ``LLMFIT_VERSION`` environment variable (e.g. ``0.9.8``).
        2. The ``version`` field in ``[workspace.package]`` from ``Cargo.toml``.
        """
        with (Path(self.root).parent / "Cargo.toml").open("rb") as f:
            workspace_package: dict[str, str] = tomli.load(f)["workspace"]["package"]
        version: str = os.environ.get("LLMFIT_VERSION") or workspace_package["version"]
        if not re.match(r"^\d+\.\d+\.\d+$", version):
            raise ValueError(f"Invalid version: {version!r}")
        metadata["version"] = version
        metadata["license-expression"] = workspace_package["license"]


class LlmfitBinaryBuildHook(BuildHookInterface):
    """Hatchling build hook that injects the llmfit binary into each wheel."""

    PLUGIN_NAME = "llmfit binary"

    @staticmethod
    def _detect_platform() -> str:
        """Return the best platform tag for the current machine."""
        best = next((t.platform for t in sys_tags() if t.platform in TARGET_CONFIGS), None)
        if best is not None:
            return best
        first = next(t.platform for t in sys_tags())
        raise RuntimeError(f"No suitable wheel platform found for runtime platform {first!r}.")

    @staticmethod
    def _find_binary_for_target(llmfit_root: Path, py_target: str) -> Path:
        """Find the binary compiled for a specific Rust target.

        Looks in ``target/{upstream_target}/release/``, which is where Cargo
        places the binary when built with ``--target``.
        """
        upstream_target, binary_name = TARGET_CONFIGS[py_target]
        bin_path = llmfit_root / "target" / upstream_target / "release" / binary_name
        if not bin_path.is_file():
            raise FileNotFoundError(
                f"Binary not found at {bin_path}. Expected it to be built for target {upstream_target!r}.",
            )
        return bin_path

    @staticmethod
    def _find_local_binary(llmfit_root: Path) -> Path:
        """Find the locally compiled binary in default Cargo output locations.

        Checks ``target/debug/`` first (from ``make build``), then
        ``target/release/`` (from ``make release``).
        """
        binary_name = "llmfit.exe" if sys.platform == "win32" else "llmfit"
        candidates = [
            llmfit_root / "target" / "debug" / binary_name,
            llmfit_root / "target" / "release" / binary_name,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            "No compiled binary found. Checked:\n"
            + "\n".join(f"  {c}" for c in candidates)
            + "\nRun 'make build' first.",
        )

    @staticmethod
    def _check_binary_version(bin_path: Path, expected_version: str) -> None:
        """Run the binary with ``--version`` and verify it matches the expected version.

        Raises ``RuntimeError`` on a mismatch — this indicates a stale build.
        """
        result = subprocess.run(
            [str(bin_path), "--version"],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
        output = result.stdout.strip()  # e.g. "llmfit 0.9.8"
        match = re.match(r"^llmfit v?(\d+\.\d+\.\d+)$", output)
        if not match:
            raise RuntimeError(f"Unexpected output from '{bin_path} --version': {output!r}")
        binary_version = match.group(1)
        if binary_version != expected_version:
            raise RuntimeError(
                f"Binary version mismatch: binary at {bin_path} reports {binary_version!r} "
                f"but Cargo.toml (or LLMFIT_VERSION) says {expected_version!r}. "
                "Run 'make build' to recompile.",
            )
        print(f"  Binary version OK ({binary_version})")

    def initialize(self, version: str, build_data: dict) -> None:
        """Locate the platform binary and configure the wheel before it is built."""
        py_target_from_env = os.environ.get("LLMFIT_PYTHON_PLATFORM_TAG")
        if version == "editable" and py_target_from_env:
            raise ValueError(
                "LLMFIT_PYTHON_PLATFORM_TAG is not supported for editable installs. "
                "Let the build system detect the host platform instead.",
            )
        running_platform = self._detect_platform()
        py_target = py_target_from_env or running_platform
        if py_target not in TARGET_CONFIGS:
            raise ValueError(
                f"Unknown LLMFIT_PYTHON_PLATFORM_TAG={py_target!r}. Must be one of: {sorted(TARGET_CONFIGS)}",
            )

        upstream_target, binary_name = TARGET_CONFIGS[py_target]
        pypi_version: str = self.metadata.version

        print(f"  target={upstream_target}  version={pypi_version}  wheel tag=py3-none-{py_target}")

        llmfit_root = Path(self.root).parent
        if version == "editable":
            # For editable installs, look for target/debug/llmfit or target/release/llmfit (or llmfit.exe on Windows).
            bin_path = self._find_local_binary(llmfit_root)
        elif version == "standard":
            # For release installs, look for e.g. target/x86_64-unknown-linux-gnu/release/llmfit on Linux.
            bin_path = self._find_binary_for_target(llmfit_root, py_target)
        else:
            raise ValueError(f"Unknown version: {version!r}")

        # Always check that the binary exists.
        if not bin_path.is_file():
            raise FileNotFoundError(
                f"Binary not found at {bin_path}. The binary selection logic selected a file that does not exist.",
            )

        # If possible, check the self-reported version of the binary.
        # If the binary was built for a different platform then it's not possible.
        if py_target == running_platform:
            self._check_binary_version(bin_path, pypi_version)

        # Place the binary in the wheel's scripts directory so that the
        # installer puts it in .venv/bin/ (or Scripts/ on Windows).
        build_data["shared_scripts"][str(bin_path.absolute())] = binary_name

        # Override the platform tag so the wheel gets the correct platform-specific name.
        build_data["tag"] = f"py3-none-{py_target}"
        build_data["pure_python"] = False
