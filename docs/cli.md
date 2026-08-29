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

### Contributing benchmarks (`bench --share`)

`llmfit bench` measures inference performance against a running provider
(Ollama, vLLM, MLX, or llama-server). llama-server is auto-detected on port
8080 via its `/props` endpoint (override with `LLAMA_SERVER_HOST` for a full
URL, or `LLAMA_SERVER_PORT`), or select it explicitly with
`--provider llamacpp`. Add `--share` to contribute your results back to the
project as a pull request — **no `gh` CLI and no account on a third-party
service required**:

```sh
# Benchmark every discovered model and open a PR with the results
llmfit bench --all --share

# Preview the exact JSON payloads without contacting GitHub
llmfit bench --all --share --dry-run

# Skip the confirmation prompt (e.g. for automation)
llmfit bench --all --share --yes

# Upload previously stored local benchmarks without benchmarking again
llmfit bench --share
```

**Every successful bench run is also saved locally** (under
`~/.local/share/llmfit/benchmarks/pending/` on Linux; override the location
with `LLMFIT_BENCH_STORE`), so skipping `--share` never discards data. These
local results appear at the top of the TUI leaderboard as “you (local)”, and
they feed back into the fit table: a model you benched shows your measured
tok/s instead of the estimate, and runs on trustworthy models (≥ 1B params,
dense) calibrate the formula estimates for **every other model** on the same
hardware (shown as “Calibrated ×N from your own llmfit bench run(s)” in the
estimate basis). Runs recorded on a different CPU/GPU configuration are
ignored.
Sharing later — `llmfit bench --share` on its own, or the share toggle in the
TUI — offers to contribute **all** stored benchmarks in a single PR; uploaded
files move to `.../benchmarks/shared/` so they are kept as history but never
submitted twice.

**Merged submissions ship in the next release.** Community files are embedded
into the binary at build time, so anyone on identical hardware (same CPU +
GPU) sees them on the benchmark page as `llmfit community` rows, gets
measured ✓ tok/s for those models, and gets calibrated estimates everywhere
else — a fresh install benefits before its user ever runs a benchmark. Trust
order everywhere: your own runs > llmfit community on identical hardware >
localmaxxing medians on matching presets > formula estimate.

Authentication uses the GitHub **device flow** (the same mechanism
`gh auth login` uses): llmfit prints a short code and a URL, you approve it in
your browser once, and the token is cached under `~/.config/llmfit/` for next
time. If a `GITHUB_TOKEN` or `GH_TOKEN` environment variable is set (or you use
CI), that token is used automatically and no browser step is needed. With
`--share`, credentials are resolved and verified **before** any benchmark
starts, so a missing or expired token fails fast instead of after minutes of
benching.

`--share` then forks the repo, commits one result file per stored submission
under `llmfit-core/data/community/<hardware>/`, and opens a pull request — or,
if you already have an open benchmark PR, **appends the new results to it**
instead of opening another. Submissions are idempotent: file names mirror your
local store, so retrying after a partial failure skips anything that already
landed. Nothing is submitted until you confirm, and `--dry-run` never touches
the network.

> Interactive login ships enabled — the public OAuth App client id is baked
> into the binary (the device flow needs no client secret, so this is safe by
> design). `LLMFIT_GH_CLIENT_ID` overrides it (e.g. when running a fork
> against your own OAuth App); set it to an empty string to disable
> interactive login entirely and rely on `GITHUB_TOKEN` / `GH_TOKEN`.

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

### Hardware profiles

The single-field overrides fix a bad detection, but they can't answer "what would this model do on *that* box": throughput is driven by memory bandwidth and fp16 matmul throughput, neither of which is a capacity. A **hardware profile** names a whole machine — capacity, unified-memory topology, memory bandwidth, DDR bandwidth, fp16 throughput — so the answer is reproducible instead of a pile of flags:

```sh
# What fits, and how fast, on a 256 GB/s unified APU with 128 GB
llmfit --profile ryzen-ai-max-plus-395 fit -n 10

# Profiles work with any analysis subcommand
llmfit --profile nvidia-rtx-4090 recommend --json
llmfit --profile apple-m3-max-128gb info "Qwen/Qwen3-4B-MLX-4bit"

# Or point at a file you wrote yourself
llmfit --profile ./my-workstation.json fit --json
```

`--profile` conflicts with `--memory`, `--ram`, and `--cpu-cores`: a profile describes the whole machine, so mixing the two would leave the result half-detected. An unresolvable profile is an error rather than a warning — a profile that quietly failed to load would score every model against the wrong machine.

Manage profiles with the `hardware` subcommand:

```sh
# Every selectable profile (bundled and user), with --json for scripts
llmfit hardware list
llmfit hardware list --json

# One profile's fields, plus what it would change on this machine
llmfit hardware show ryzen-ai-max-plus-395

# Strictly validate files you wrote (unknown keys are errors here)
llmfit hardware validate ./my-workstation.json

# Where to drop your own profiles
llmfit hardware path
```

Profiles bundled with llmfit are embedded in the binary. Your own go in the directory printed by `llmfit hardware path` (override it with `LLMFIT_HARDWARE_PROFILES`), one `<name>.json` per profile, where `name` inside the file must match the file stem. A user profile shadows a bundled one of the same name.

Loading **tolerates** unknown keys so a profile written for a newer llmfit still works, which means a misspelled key silently does nothing — `llmfit hardware validate` rejects them, so run it after hand-editing:

```console
$ llmfit hardware validate ./my-workstation.json
FAIL  ./my-workstation.json: unknown key(s): hardware.gpu_bandwith_gbps
```

The full field list, bounds, and bundled-profile provenance live in [`llmfit-core/data/hardware/README.md`](../llmfit-core/data/hardware/README.md) with the schema next to it. Two current limitations:

- `calibration[]` entries are parsed and validated but **not applied** to estimates in `schema_version` 1; they ship as reviewable data.
- `--profile` cannot be combined with `recommend --force-runtime` yet.

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
