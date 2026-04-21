from __future__ import annotations

import os
import sys

from llmfit import find_llmfit_bin


def main() -> None:
    """Entry point for the llmfit CLI."""
    bin_path = str(find_llmfit_bin())
    args = [bin_path, *sys.argv[1:]]
    if sys.platform == "win32":
        import subprocess  # noqa: PLC0415

        try:
            completed_process = subprocess.run(args, check=False)
        except KeyboardInterrupt:
            sys.exit(2)
        sys.exit(completed_process.returncode)
    else:
        os.execv(bin_path, args)  # noqa: S606 # arguments are sufficiently validated


if __name__ == "__main__":
    main()
