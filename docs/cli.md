# CLI & Automation

Classic table output, the REST API for cluster scheduling, hardware overrides, context caps, and JSON output for scripts and agents.

[← Back to README](../README.md)

### CLI mode

Use `--cli` or any subcommand to get classic table output:

```sh
# Table of all models ranked by fit
llmfit --cli

# Only perfectly fitting models, top 5
llmfit fit --perfect -n 5

# Show detected system specs
llmfit system

# Hardware diagnostic report for bug reports (raw nvidia-smi/rocm-smi/sysfs
# output + what llmfit detected) — paste into a GitHub issue
llmfit doctor

# List all models in the database
llmfit list

# Search by name, provider, or size
llmfit search "llama 8b"

# Detailed view of a single model
llmfit info "Mistral-7B"

# Top 5 recommendations (JSON, for agent/script consumption)
llmfit recommend --json --limit 5

# Recommendations filtered by use case
llmfit recommend --json --use-case coding --limit 3

# Force a specific runtime (bypass automatic MLX selection on Apple Silicon)
llmfit recommend --force-runtime llamacpp
llmfit recommend --force-runtime llamacpp --use-case coding --limit 3

# Plan required hardware for a specific model configuration
llmfit plan "Qwen/Qwen3-4B-MLX-4bit" --context 8192
llmfit plan "Qwen/Qwen3-4B-MLX-4bit" --context 8192 --quant mlx-4bit
llmfit plan "Qwen/Qwen3-4B-MLX-4bit" --context 8192 --target-tps 25 --json

# Run as a node-level REST API (for cluster schedulers / aggregators)
llmfit serve --host 0.0.0.0 --port 8787
```

### REST API (`llmfit serve`)

`llmfit serve` starts an HTTP API that exposes the same fit/scoring data used by TUI/CLI, including filtering and top-model selection for a node.

```sh
# Liveness
curl http://localhost:8787/health

# Node hardware info
curl http://localhost:8787/api/v1/system

# Full fit list with filters
curl "http://localhost:8787/api/v1/models?min_fit=marginal&runtime=llamacpp&sort=score&limit=20"

# Key scheduling endpoint: top runnable models for this node
curl "http://localhost:8787/api/v1/models/top?limit=5&min_fit=good&use_case=coding"

# Search by model name/provider text
curl "http://localhost:8787/api/v1/models/Mistral?runtime=any"
```

Supported query params for `models`/`models/top`:

- `limit` (or `n`): max number of rows returned
- `perfect`: `true|false` (forces perfect-only when `true`)
- `min_fit`: `perfect|good|marginal|too_tight`
- `runtime`: `any|mlx|llamacpp`
- `use_case`: `general|coding|reasoning|chat|multimodal|embedding`
- `provider`: provider text filter (substring)
- `search`: free-text filter across name/provider/size/use-case
- `sort`: `score|tps|params|mem|ctx|date|use_case`
- `include_too_tight`: include non-runnable rows (default `false` on `/top`, `true` on `/models`)
- `max_context`: per-request context cap for memory estimation
- `force_runtime`: `mlx|llamacpp|vllm` — override automatic runtime selection during analysis

Validate API behavior locally:

```sh
# spawn server automatically and run endpoint/schema/filter assertions
python3 scripts/test_api.py --spawn

# or test an already-running server
python3 scripts/test_api.py --base-url http://127.0.0.1:8787
```

### Hardware overrides

Hardware autodetection can fail on some systems (e.g. broken `nvidia-smi`, VMs, passthrough setups), or you may want to evaluate model fit against different target hardware. Use `--memory`, `--ram`, and `--cpu-cores` to override detected values:

```sh
# Override GPU VRAM
llmfit --memory=32G

# Override system RAM
llmfit --ram=128G

# Override CPU core count
llmfit --cpu-cores=16

# Combine overrides to simulate target hardware
llmfit --memory=24G --ram=64G --cpu-cores=8 fit
llmfit --memory=24G --ram=64G system --json

# Works with all modes: TUI, CLI, and subcommands
llmfit --memory=24G --cli
llmfit --memory=24G fit --perfect -n 5
llmfit --ram=64G recommend --json
```

Accepted suffixes for `--memory` and `--ram`: `G`/`GB`/`GiB` (gigabytes), `M`/`MB`/`MiB` (megabytes), `T`/`TB`/`TiB` (terabytes). Case-insensitive. If no GPU was detected, `--memory` creates a synthetic GPU entry so models are scored for GPU inference. On unified-memory systems (Apple Silicon), `--ram` also updates VRAM; use `--memory` to override VRAM independently.

### Context-length cap for estimation

Use `--max-context` to cap context length used for memory estimation (without changing each model's advertised maximum context):

```sh
# Estimate memory fit at 4K context
llmfit --max-context 4096 --cli

# Works with subcommands
llmfit --max-context 8192 fit --perfect -n 5
llmfit --max-context 16384 recommend --json --limit 5
```

If `--max-context` is not set, llmfit will use `OLLAMA_CONTEXT_LENGTH` when available.

### JSON output

Add `--json` to any subcommand for machine-readable output:

```sh
llmfit --json system     # Hardware specs as JSON
llmfit --json fit -n 10  # Top 10 fits as JSON
llmfit recommend --json  # Top 5 recommendations (JSON is default for recommend)
llmfit plan "Qwen/Qwen2.5-Coder-0.5B-Instruct" --context 8192 --json
```

`plan` JSON includes stable fields for:
- request (`context`, `quantization`, `target_tps`)
- estimated minimum/recommended hardware
- per-path feasibility (`gpu`, `cpu_offload`, `cpu_only`)
- upgrade deltas

---
