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
    <unix-timestamp>-<hash>-<n>.json
```

Files are namespaced by hardware and carry a content hash so concurrent
submissions never collide.

## Format

Each file conforms to [`schema.json`](./schema.json). Example:

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
