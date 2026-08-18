#!/bin/sh
set -e

# Web UI / Frontend server explicit triggers
if [ "$1" = "web" ] || [ "$1" = "ui" ]; then
    echo "[INFO] Starting llmfit web interface..."
    exec npm run start -- --host 0.0.0.0 --port 8787
fi

# Pass through explicit CLI arguments or subcommands to the Rust binary
if [ "$#" -gt 0 ]; then
    exec llmfit "$@"
fi

# Default fallback when no arguments are supplied:
# Output JSON recommendations as expected by the default container use case
exec llmfit recommend --json
