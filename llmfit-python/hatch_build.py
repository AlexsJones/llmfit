"""
hatch_build.py  -  Hatchling build hook for llmfit.

Downloads the pre-built binary for the target platform from GitHub Releases,
verifies its SHA256 checksum, and injects it into the wheel via shared_scripts
so that the installer places it in the environment's scripts directory
(e.g. ``.venv/bin/llmfit``).  Also overrides the wheel platform tag so that
cross-platform wheels can be produced from a single Linux CI runner.

For editable installs (``uv sync``, ``uv run``), the same mechanism is used
so that ``find_llmfit_bin()`` works via ``sysconfig.get_path("scripts")``.

Environment variables
---------------------
LLMFIT_PYTHON_PLATFORM_TAG
    Wheel platform tag to build for (e.g. ``manylinux_2_17_x86_64``).
    If unset, the current machine's platform is auto-detected.  Set this
    explicitly when building for a platform other than the host.
LLMFIT_UPSTREAM_VERSION
    Upstream release tag to fetch, including the leading ``v``
    (e.g. ``v0.8.6``).  If unset, the latest GitHub release is fetched
    automatically.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import NewType

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from hatchling.metadata.plugin.interface import MetadataHookInterface
from packaging.tags import sys_tags

GITHUB_API_LATEST = "https://api.github.com/repos/AlexsJones/llmfit/releases/latest"
GITHUB_DOWNLOAD_URL = "https://github.com/AlexsJones/llmfit/releases/download/{version_tag}/{filename}"
GITHUB_LICENSE_API_URL = "https://api.github.com/repos/AlexsJones/llmfit/license?ref={ref}"

# The SPDX identifier we claim in our LICENSE file for the upstream binary.
# If the upstream ever relicenses, this check will catch it and the build will fail.
CLAIMED_UPSTREAM_SPDX_ID = "MIT"

# wheel_platform_tag -> (upstream_target, binary_name, is_zip)
TARGET_CONFIGS: dict[str, tuple[str, str, bool]] = {
    "manylinux_2_17_x86_64": ("x86_64-unknown-linux-gnu", "llmfit", False),
    "manylinux_2_17_aarch64": ("aarch64-unknown-linux-gnu", "llmfit", False),
    "musllinux_1_2_x86_64": ("x86_64-unknown-linux-musl", "llmfit", False),
    "musllinux_1_2_aarch64": ("aarch64-unknown-linux-musl", "llmfit", False),
    "macosx_10_12_x86_64": ("x86_64-apple-darwin", "llmfit", False),
    "macosx_11_0_arm64": ("aarch64-apple-darwin", "llmfit", False),
    "win_amd64": ("x86_64-pc-windows-msvc", "llmfit.exe", True),
    "win_arm64": ("aarch64-pc-windows-msvc", "llmfit.exe", True),
}

# Use the NewType system to prevent confusion between upstream and PyPI version strings.
UpstreamVersion = NewType("UpstreamVersion", str)  # e.g. "v0.8.6"
PyPIVersion = NewType("PyPIVersion", str)  # e.g. "0.8.6"


def _validate_upstream_version(upstream_version: str) -> UpstreamVersion:
    if not re.match(r"^v\d+\.\d+\.\d+$", upstream_version):
        raise ValueError(f"Invalid upstream version: {upstream_version!r}")
    return UpstreamVersion(upstream_version)


def _upstream_to_pypi(upstream_version: UpstreamVersion) -> PyPIVersion:
    return PyPIVersion(upstream_version.lstrip("v"))


def _pypi_to_upstream(pypi_version: PyPIVersion) -> UpstreamVersion:
    return UpstreamVersion(f"v{pypi_version}")


class LlmfitMetadataHook(MetadataHookInterface):
    """Hatchling metadata hook that sets version and license dynamically."""

    PLUGIN_NAME = "llmfit version and license information"

    @staticmethod
    def _get_upstream_version() -> UpstreamVersion:
        """Return the upstream llmfit version to package.

        Resolution order:

        1. ``LLMFIT_UPSTREAM_VERSION`` environment variable — must be an upstream version tag with
           a leading ``v`` (e.g. ``v0.8.6``).  ``build_wheels.py`` sets this.
        2. Latest upstream release fetched from the GitHub API.
        """
        v = os.environ.get("LLMFIT_UPSTREAM_VERSION")
        if v:
            return _validate_upstream_version(v)
        print("  LLMFIT_UPSTREAM_VERSION not set; fetching latest release tag from GitHub")
        with urllib.request.urlopen(GITHUB_API_LATEST) as resp:
            tag = json.loads(resp.read())["tag_name"]
        return _validate_upstream_version(tag)

    @staticmethod
    def _verify_upstream_license(version_tag: UpstreamVersion) -> str | None:
        """Fetch the upstream license via the GitHub API and confirm it matches our claim.

        Returns ``None`` on success (and prints a confirmation message).  Returns a
        non-empty warning message string on any failure: network errors, unidentified
        licenses, or mismatches.  Never raises.
        """
        url = GITHUB_LICENSE_API_URL.format(ref=version_tag)
        print(f"  GET {url}")
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return f"Could not retrieve upstream license from {url}: {exc}"

        spdx_id = (data.get("license") or {}).get("spdx_id", "NOASSERTION")
        if spdx_id == "NOASSERTION":
            return (
                f"GitHub could not identify the upstream license at tag {version_tag!r}. "
                f"Cannot verify our claim of {CLAIMED_UPSTREAM_SPDX_ID!r}."
            )
        if spdx_id != CLAIMED_UPSTREAM_SPDX_ID:
            return (
                f"Upstream license mismatch at tag {version_tag!r}: "
                f"we claim {CLAIMED_UPSTREAM_SPDX_ID!r} but GitHub reports {spdx_id!r}. "
                "Update LICENSE and CLAIMED_UPSTREAM_SPDX_ID to resolve this."
            )
        print(f"  License OK (upstream SPDX: {spdx_id})")
        return None

    def update(self, metadata: dict) -> None:
        """Populate ``version``, ``license-expression``, and ``license-files`` in the project metadata table."""
        upstream_version = self._get_upstream_version()
        metadata["version"] = _upstream_to_pypi(upstream_version)

        license_warning = self._verify_upstream_license(upstream_version)
        if license_warning is not None:
            metadata["license-expression"] = license_warning
            metadata["license-files"] = []
        else:
            metadata["license-expression"] = CLAIMED_UPSTREAM_SPDX_ID
            metadata["license-files"] = ["LICENSE"]


class LlmfitBinaryBuildHook(BuildHookInterface):
    """Hatchling build hook that injects the llmfit binary into each wheel.

    Fails a release build if the upstream license cannot be verified. For an editable build, a
    warning is printed instead.
    """

    PLUGIN_NAME = "llmfit binary from GitHub releases"

    @staticmethod
    def _detect_platform() -> str:
        """Return the best platform tag for the current machine."""
        first = next(t.platform for t in sys_tags())
        best = next((t.platform for t in sys_tags() if t.platform in TARGET_CONFIGS), None)
        if best is not None:
            return best
        raise RuntimeError(f"No suitable wheel platform found for runtime platform {first!r}.")

    @staticmethod
    def _download(url: str) -> bytes:
        print(f"  GET {url}")
        with urllib.request.urlopen(url) as resp:
            return resp.read()

    @staticmethod
    def _extract(archive_bytes: bytes, binary_name: str, *, is_zip: bool) -> bytes:
        if is_zip:
            with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
                for name in zf.namelist():
                    if Path(name).name == binary_name:
                        return zf.read(name)
            raise FileNotFoundError(f"{binary_name!r} not found in zip archive")
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tf:
            for member in tf.getmembers():
                if Path(member.name).name == binary_name:
                    f = tf.extractfile(member)
                    if f is not None:
                        return f.read()
        raise FileNotFoundError(f"{binary_name!r} not found in tar.gz archive")

    def _fetch_binary(self, upstream_version: UpstreamVersion, py_target: str) -> Path:
        """Download, verify, and extract the binary for the given version and target."""
        upstream_target, binary_name, is_zip = TARGET_CONFIGS[py_target]
        ext = ".zip" if is_zip else ".tar.gz"
        archive_filename = f"llmfit-{upstream_version}-{upstream_target}{ext}"
        sha256_filename = f"{archive_filename}.sha256"

        archive_url = GITHUB_DOWNLOAD_URL.format(version_tag=upstream_version, filename=archive_filename)
        sha256_url = GITHUB_DOWNLOAD_URL.format(version_tag=upstream_version, filename=sha256_filename)

        # Write archive and binary to a subdirectory of artifacts/ (gitignored).
        archive_dir = Path(self.root) / "artifacts" / "archives"
        bin_dir = Path(self.root) / "artifacts" / "binaries" / upstream_version
        archive_dir.mkdir(parents=True, exist_ok=True)
        bin_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / archive_filename
        bin_path = bin_dir / f"llmfit-{upstream_target}{'.exe' if binary_name.endswith('.exe') else ''}"

        if archive_path.is_file():
            print(f"  archive: cached ({archive_filename})")
            archive_bytes = archive_path.read_bytes()
        else:
            archive_bytes = self._download(archive_url)

        sha256_content = self._download(sha256_url).decode()
        expected_hash = sha256_content.split()[0]
        actual_hash = hashlib.sha256(archive_bytes).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                f"SHA256 mismatch for {archive_filename}:\n  expected: {expected_hash}\n  actual:   {actual_hash}",
            )
        print(f"  SHA256 OK ({actual_hash[:16]}...)")

        binary_data = self._extract(archive_bytes, binary_name, is_zip=is_zip)
        print(f"  Extracted {binary_name} ({len(binary_data):,} bytes)")

        if not archive_path.is_file():
            archive_path.write_bytes(archive_bytes)
        bin_path.write_bytes(binary_data)
        bin_path.chmod(0o755)
        return bin_path

    def initialize(self, version: str, build_data: dict) -> None:
        """Download the platform binary and configure the wheel before it is built."""
        py_target = os.environ.get("LLMFIT_PYTHON_PLATFORM_TAG") or self._detect_platform()
        if py_target not in TARGET_CONFIGS:
            raise ValueError(
                f"Unknown LLMFIT_PYTHON_PLATFORM_TAG={py_target!r}. Must be one of: {sorted(TARGET_CONFIGS)}",
            )

        upstream_target, binary_name, _ = TARGET_CONFIGS[py_target]
        upstream_version: UpstreamVersion = _pypi_to_upstream(PyPIVersion(self.metadata.version))

        print(f"  target={upstream_target}  version={upstream_version}  wheel tag=py3-none-{py_target}")

        license_expression = self.metadata.core_raw_metadata.get("license-expression", "")
        if license_expression != CLAIMED_UPSTREAM_SPDX_ID:
            if version == "editable":
                print(f"  {license_expression}")
            else:  # version == "release"
                raise RuntimeError(f"{license_expression} Refusing to build.")

        bin_path = self._fetch_binary(upstream_version, py_target)

        # Place the binary in the wheel's scripts directory so that the
        # installer puts it in .venv/bin/ (or Scripts/ on Windows).
        build_data["shared_scripts"][str(bin_path)] = binary_name

        # Override the platform tag so cross-platform wheels get the right name.
        build_data["tag"] = f"py3-none-{py_target}"
        build_data["pure_python"] = False
