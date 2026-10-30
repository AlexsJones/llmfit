# Community benchmarks

This directory collects benchmark results contributed by llmfit users via:

```sh
llmfit bench --all --share
```

`--share` runs a benchmark sweep, shows you the exact JSON payloads, asks for
confirmation, then opens a pull request adding the files here — **without
needing the `gh` CLI**. Authentication uses the GitHub OAuth *device flow* (the
same mechanism `gh auth login` uses); a `GITHUB_TOKEN` / `GH_TOKEN` env var is
used automatically when present (e.g. in CI).

Bench runs made **without** `--share` are kept in a local store on the user's
machine; a later `llmfit bench --share` (with nothing else to bench) uploads
that stored backlog in one PR, so declining to share never discards data.

Preview what would be submitted without contacting GitHub:

```sh
llmfit bench --all --share --dry-run
```

## Layout

```
community/
  <hardware-slug>/
    <unix-timestamp>-<hash>.json
```

Files are namespaced by hardware and carry a content hash so concurrent
submissions never collide. Each file name mirrors the contributor's local
store entry, which makes submissions idempotent: if a contributor already has
an open benchmark PR, new results are appended to it (instead of opening
another PR), and a retry after a partial failure skips files that already
landed rather than duplicating them.

## What happens after merge

Submissions are aggregated by `llmfit-core/build.rs` and **embedded into the
binary**, so every merged submission ships in the next release. Users on
identical hardware (same CPU + GPU) then get:

- your runs on their **benchmark page**, attributed `llmfit community`
- **measured ✓ tok/s** in the fit table for the models you benched
  (below their own local runs, above localmaxxing medians and estimates)
- **calibrated estimates** for every other model, anchored on your runs —
  a fresh install benefits before its user ever benchmarks anything

## Validation

Every PR touching this directory runs the **Community Benchmarks** workflow
(`scripts/validate_community_benchmarks.py`): schema conformance against
[`schema.json`](./schema.json), path/naming conventions, and cross-field
sanity checks (tps ordering, plausible hardware bounds, submission
timestamps). Hand-crafted submissions are welcome as long as they pass the
same checks the generated ones do.

## Format

Each file conforms to [`schema.json`](./schema.json). Example:

`hardware.pcieLinks` records one link per detected card. The field is absent
when detection cannot provide every card's link width. Matching compares the
width of each card. It ignores the current generation because it can fall at
idle. A record without `pcieLinks` matches only when the current machine also
lacks complete link widths. Old records remain valid under schema version 1.
Users must run a new benchmark to calibrate a GPU with known link widths.
Mixed GPU models keep the old matching rule until the payload identifies each GPU.

```json
{
  "schemaVersion": 1,
  "submittedAtUnix": 1752127200,
  "tool": { "name": "llmfit", "version": "1.0.0" },
  "hardware": {
    "hwClass": "DISCRETE_GPU",
    "hardwareName": "NVIDIA GeForce RTX 4090",
    "memTierGb": 24,
    "vramGb": 24.0,
    "gpuCount": 1,
    "unifiedMemory": false,
    "cpu": "AMD Ryzen 9 7950X",
    "cpuCores": 32,
    "ramGb": 64.0,
    "os": "linux"
  },
  "results": [
    {
      "model": "llama3.1:8b",
      "provider": "ollama",
      "numRuns": 3,
      "avgTps": 128.4,
      "minTps": 121.0,
      "maxTps": 133.7,
      "avgTtftMs": 41.2,
      "avgTotalMs": 812.5,
      "avgOutputTokens": 104.0
    }
  ]
}
```

Submissions are validated against the schema and sanity-checked (measurements
within physical limits for the reported hardware) before merge.
