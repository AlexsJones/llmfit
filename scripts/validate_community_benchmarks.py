#!/usr/bin/env python3
"""Validate community benchmark submissions (llmfit-core/data/community/).

Run by CI on every PR touching the community data directory, and runnable
locally:

    python3 scripts/validate_community_benchmarks.py            # all files
    python3 scripts/validate_community_benchmarks.py FILE...    # specific files

Checks, per file:
  1. Path conventions — `community/<hardware-slug>/<timestamp>-<hash>.json`
     with a lowercase alphanumeric-dash slug (what `llmfit bench --share`
     generates; anything else is hand-crafted and gets a closer look).
  2. Size cap — a submission is a few KB; anything large is not a benchmark.
  3. Valid JSON that conforms to community/schema.json (draft-07).
  4. Cross-field sanity the schema cannot express: minTps <= avgTps <= maxTps,
     plausible hardware bounds, and a submission timestamp that is neither
     before the feature existed nor in the future.

Exit code 0 when every file passes; 1 with one line per problem otherwise.

Requires: jsonschema (`pip install jsonschema`).
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import jsonschema

REPO_ROOT = Path(__file__).resolve().parent.parent
COMMUNITY_DIR = REPO_ROOT / "llmfit-core" / "data" / "community"
SCHEMA_PATH = COMMUNITY_DIR / "schema.json"

MAX_FILE_BYTES = 64 * 1024
MAX_RESULTS_PER_FILE = 100

# `bench --share` names files after the local store entry: unix timestamp +
# 8-hex content hash (an optional -<n> suffix appeared in early builds).
FILENAME_RE = re.compile(r"^\d{9,12}-[0-9a-f]{8}(-\d+)?\.json$")
SLUG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

# The share feature shipped in July 2026; nothing genuine predates it.
MIN_SUBMITTED_AT = 1_782_864_000  # 2026-07-01 00:00:00 UTC
FUTURE_SLACK_SECS = 24 * 3600

# Upper bounds that the schema leaves open. Generous on purpose: these catch
# nonsense (a 4 TB GPU, a million-core CPU), not exotic-but-real hardware.
MAX_RAM_GB = 8192
MAX_VRAM_GB = 2048
MAX_CPU_CORES = 1024
MAX_GPU_COUNT = 64


def check_file(path: Path, validator: jsonschema.Draft7Validator) -> list[str]:
    problems: list[str] = []
    rel = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path

    def bad(msg: str) -> None:
        problems.append(f"{rel}: {msg}")

    # 1. Path conventions.
    if path.parent.parent != COMMUNITY_DIR:
        bad("must live directly under community/<hardware-slug>/")
    else:
        slug = path.parent.name
        if not SLUG_RE.match(slug):
            bad(f"hardware slug {slug!r} must be lowercase alphanumeric-dash")
    if not FILENAME_RE.match(path.name):
        bad(f"file name {path.name!r} must match <unix-timestamp>-<hash>.json")

    # 2. Size cap.
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        bad(f"file is {size} bytes (cap {MAX_FILE_BYTES}); not a benchmark submission")
        return problems

    # 3. JSON + schema.
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        bad(f"not valid JSON: {e}")
        return problems
    schema_errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.path))
    for err in schema_errors:
        where = "/".join(str(p) for p in err.path) or "(root)"
        bad(f"schema violation at {where}: {err.message}")
    if schema_errors:
        return problems

    # 4. Cross-field sanity.
    submitted = payload["submittedAtUnix"]
    if submitted < MIN_SUBMITTED_AT:
        bad(f"submittedAtUnix {submitted} predates the share feature (min {MIN_SUBMITTED_AT})")
    if submitted > time.time() + FUTURE_SLACK_SECS:
        bad(f"submittedAtUnix {submitted} is in the future")

    hw = payload["hardware"]
    if hw["ramGb"] > MAX_RAM_GB:
        bad(f"ramGb {hw['ramGb']} exceeds sanity cap {MAX_RAM_GB}")
    if (hw.get("vramGb") or 0) > MAX_VRAM_GB:
        bad(f"vramGb {hw['vramGb']} exceeds sanity cap {MAX_VRAM_GB}")
    if hw["cpuCores"] > MAX_CPU_CORES:
        bad(f"cpuCores {hw['cpuCores']} exceeds sanity cap {MAX_CPU_CORES}")
    if hw["gpuCount"] > MAX_GPU_COUNT:
        bad(f"gpuCount {hw['gpuCount']} exceeds sanity cap {MAX_GPU_COUNT}")

    results = payload["results"]
    if len(results) > MAX_RESULTS_PER_FILE:
        bad(f"{len(results)} results in one submission (cap {MAX_RESULTS_PER_FILE})")
    for i, r in enumerate(results):
        if not r["minTps"] <= r["avgTps"] <= r["maxTps"]:
            bad(
                f"results[{i}] ({r['model']}): tps ordering violated "
                f"(min {r['minTps']}, avg {r['avgTps']}, max {r['maxTps']})"
            )
        if r["avgTps"] <= 0:
            bad(f"results[{i}] ({r['model']}): avgTps must be positive")
        # Multi-node layout (#1227): a result without nodeCount is
        # single-node. A multi-node result must say how the model was split.
        if r.get("nodeCount", 1) > 1 and "tpSize" not in r and "ppSize" not in r:
            bad(f"results[{i}] ({r['model']}): nodeCount > 1 needs tpSize and/or ppSize")
        if "interconnect" in r and not r["interconnect"].strip():
            bad(f"results[{i}] ({r['model']}): interconnect must not be blank")

    return problems


def main() -> int:
    if len(sys.argv) > 1:
        files = [Path(a).resolve() for a in sys.argv[1:]]
    else:
        files = sorted(
            p for p in COMMUNITY_DIR.rglob("*.json") if p != SCHEMA_PATH
        )

    if not files:
        print("no community submissions to validate")
        return 0

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft7Validator.check_schema(schema)
    validator = jsonschema.Draft7Validator(schema)

    problems: list[str] = []
    for f in files:
        if not f.exists():
            # A PR that deletes/moves a file passes paths of removed entries.
            continue
        problems.extend(check_file(f, validator))

    if problems:
        print(f"❌ {len(problems)} problem(s) across {len(files)} file(s):")
        for p in problems:
            print(f"  {p}")
        return 1
    print(f"✅ {len(files)} community submission(s) valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
