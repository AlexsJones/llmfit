# llmfit REST API Guide

This document is for agent/client builders integrating with `llmfit serve`.

## Purpose

`llmfit serve` exposes node-local model fit analysis (same core data used by TUI/CLI) over HTTP and serves a local web dashboard.

Primary use case:
- Query each node in a cluster for top runnable models.
- Aggregate externally (scheduler/controller/UI) for placement decisions.

## Start the server

```sh
llmfit serve --port 8787
```

Global flags still apply:

```sh
llmfit --memory 24G --max-context 8192 serve --port 8787
```

## Base URL

Default local base URL:

```text
http://127.0.0.1:8787
```

To expose outside localhost, pass `--host 0.0.0.0`.

If you are building from source and want the dashboard embedded in `llmfit`, build web assets first:

```sh
cd llmfit-web && npm ci && npm run build
```

## Endpoints

### `GET /`
Web dashboard entrypoint (same-origin UI for fit exploration).

### `GET /health`
Liveness probe.

Example response:

```json
{
  "status": "ok",
  "node": {
    "name": "worker-1",
    "os": "linux"
  }
}
```

---

### `GET /api/v1/system`
Returns node identity + detected hardware.

Example response shape:

```json
{
  "node": {
    "name": "worker-1",
    "os": "linux"
  },
  "system": {
    "total_ram_gb": 62.23,
    "available_ram_gb": 41.08,
    "cpu_cores": 14,
    "cpu_name": "Intel(R) Core(TM) Ultra 7 165U",
    "has_gpu": false,
    "gpu_vram_gb": null,
    "gpu_name": null,
    "gpu_count": 0,
    "unified_memory": false,
    "backend": "CPU (x86)",
    "gpus": []
  }
}
```

---

### `GET /api/v1/models`
Returns filtered/sorted model-fit rows for this node.

Envelope shape:

```json
{
  "node": { "name": "worker-1", "os": "linux" },
  "system": { "...": "..." },
  "total_models": 23,
  "returned_models": 10,
  "filters": { "...": "echo of query state" },
  "models": [
    {
      "name": "Qwen/Qwen2.5-Coder-7B-Instruct",
      "provider": "Qwen",
      "parameter_count": "7B",
      "params_b": 7.0,
      "context_length": 32768,
      "use_case": "Coding",
      "category": "Coding",
      "release_date": "2025-03-14",
      "is_moe": false,
      "fit_level": "good",
      "fit_label": "Good",
      "run_mode": "gpu",
      "run_mode_label": "GPU",
      "score": 86.5,
      "score_components": {
        "quality": 87.0,
        "speed": 81.2,
        "fit": 90.1,
        "context": 88.0
      },
      "estimated_tps": 42.5,
      "runtime": "llamacpp",
      "runtime_label": "llama.cpp",
      "best_quant": "Q5_K_M",
      "memory_required_gb": 5.8,
      "memory_available_gb": 12.0,
      "utilization_pct": 48.3,
      "notes": [],
      "gguf_sources": []
    }
  ]
}
```

---

### `GET /api/v1/models/top`
Key scheduling endpoint. Same schema as `/api/v1/models`, but defaults to top 5 runnable entries.

Important behavior:
- Defaults `limit=5`.
- Excludes `too_tight` rows unless explicitly overridden (and top endpoint still keeps runnable semantics).

---

### `GET /api/v1/models/{name}`
Path-constrained search. Equivalent to a text search scoped by `{name}`.

Useful for:
- Client-side drilldown after selecting a model family.

## Query parameters

Supported on `/api/v1/models` and `/api/v1/models/top` (also `/api/v1/models/{name}`):

- `limit` (or alias `n`): max rows returned.
- `perfect`: `true|false` (when `true`, only perfect fits).
- `min_fit`: `perfect|good|marginal|too_tight`.
- `runtime`: `any|mlx|llamacpp`.
- `use_case`: `general|coding|reasoning|chat|multimodal|embedding`.
- `provider`: provider substring filter.
- `search`: free-text filter (name/provider/params/use-case/category).
- `sort`: `score|tps|params|mem|ctx|date|use_case`.
- `include_too_tight`: include unrunnable rows (defaults true for `/models`, false for `/models/top`).
- `max_context`: per-request context cap used by memory estimation.
- `force_runtime`: `mlx|llamacpp|vllm` — override automatic runtime selection during analysis (e.g. get llama.cpp recommendations on Apple Silicon instead of MLX).

## Error handling

Invalid filter values return HTTP 400:

```json
{
  "error": "invalid min_fit value: use perfect|good|marginal|too_tight"
}
```

Server errors return HTTP 500 with `{"error": "..."}`.

## Client integration recommendations

### 1) Polling pattern for schedulers
For each node agent:
1. Call `/health`.
2. Call `/api/v1/system`.
3. Call `/api/v1/models/top?limit=K&min_fit=good`.
4. Attach node metadata and forward to your central scheduler.

### 2) Conservative placement defaults
For production placement, prefer:

```text
min_fit=good
include_too_tight=false
sort=score
limit=5..20
```

### 3) Per-workload targeting
Examples:
- Coding workloads: `use_case=coding`
- Embedding workloads: `use_case=embedding`
- Runtime constrained to llama.cpp fleet: `runtime=llamacpp`

### 4) Stable parsing
Treat unknown fields as forward-compatible additions:
- Parse required fields you depend on.
- Ignore unknown fields.

## Curl examples

```sh
curl http://127.0.0.1:8787/health
curl http://127.0.0.1:8787/api/v1/system
curl "http://127.0.0.1:8787/api/v1/models?limit=20&min_fit=marginal&sort=score"
curl "http://127.0.0.1:8787/api/v1/models/top?limit=5&min_fit=good&use_case=coding"
curl "http://127.0.0.1:8787/api/v1/models/Mistral?runtime=any"
```

## Versioning notes

Current API prefix is `v1`.

If you build long-lived clients, pin to `/api/v1/...` and validate behavior with the local test script in `scripts/test_api.py`.
