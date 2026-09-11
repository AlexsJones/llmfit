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
(Ollama, vLLM, Ferrum, MLX, or llama-server). vLLM and Ferrum are distinguished
by the `owned_by` identity in `/v1/models`; set `FERRUM_HOST` to override
Ferrum's default `http://localhost:8000` endpoint. llama-server is
auto-detected on port 8080 via its `/props` endpoint (override with
`LLAMA_SERVER_HOST` for a full URL, or `LLAMA_SERVER_PORT`), or select it
explicitly with `--provider llamacpp`. Add `--share` to contribute your results
back to the project as a pull request — **no `gh` CLI and no account on a
third-party service required**:

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

`--memory` / `--ram` / `--cpu-cores` fix capacity. They cannot answer “how fast on *that* box?” — tok/s needs memory bandwidth (and optionally fp16 TFLOPS). A **hardware profile** is a small JSON file that describes a whole machine. Pass it with `--profile` and every analysis command scores against that machine instead of the host you are sitting on.

#### Try a bundled profile (30 seconds)

```sh
llmfit hardware list
llmfit hardware show ryzen-ai-max-plus-395

llmfit --profile ryzen-ai-max-plus-395 fit -n 10
llmfit --profile ryzen-ai-max-plus-395 plan openai/gpt-oss-120b
llmfit --profile nvidia-rtx-4090 recommend --json
llmfit --profile apple-m3-max-128gb info "Qwen/Qwen3-4B-MLX-4bit"
```

#### Simulate unreleased (or any) hardware

You do not need the machine in front of you. Write a profile, validate it, then score models against it.

**1. See where user profiles live**

```sh
llmfit hardware path
# e.g. ~/.local/share/llmfit/hardware
# override with: LLMFIT_HARDWARE_PROFILES=/tmp/my-hw
```

**2. Write a profile** (file stem must match `"name"`)

```sh
mkdir -p "$(llmfit hardware path)"
cat > "$(llmfit hardware path)/m5ultra512.json" <<'EOF'
{
  "schema_version": 1,
  "name": "m5ultra512",
  "match": { "gpu_name_contains": "M5 Ultra" },
  "hardware": {
    "total_ram_gb": 512.0,
    "unified_memory": true,
    "gpu_memory_bandwidth_gbps": 1200.0,
    "ddr_bandwidth_gbps": 1200.0
  }
}
EOF
```

Or keep a one-off file and pass the path — no install needed:

```sh
llmfit --profile ./m5ultra512.json fit --json
```

**3. Validate, list, inspect**

```sh
llmfit hardware validate "$(llmfit hardware path)/m5ultra512.json"
llmfit hardware list
llmfit hardware show m5ultra512
```

**4. Score as if you owned that box**

```sh
llmfit --profile m5ultra512 fit -n 20
llmfit --profile m5ultra512 plan --quant Q4_K_M openai/gpt-oss-120b
llmfit --profile m5ultra512 recommend --json
```

| Field | Effect |
| --- | --- |
| `total_ram_gb` | Capacity (and VRAM when `unified_memory` is true) |
| `unified_memory` | Shared pool (Apple / APU) vs discrete GPU |
| `gpu_memory_bandwidth_gbps` | Decode / estimated tok/s |
| `ddr_bandwidth_gbps` | CPU / offload path |
| `gpu_compute_tflops_fp16` | Prefill / TTFT; omit → honest `null` |

#### Managing profiles

```sh
llmfit hardware list          # bundled + user
llmfit hardware list --json
llmfit hardware show <NAME>   # fields + what would change on this host
llmfit hardware validate <file>
llmfit hardware path
```

Bundled profiles are embedded in the binary. Your own live under `llmfit hardware path` (or `LLMFIT_HARDWARE_PROFILES`). Same `name` → user file wins.

Loading **tolerates** unknown keys (forward-compatible). `hardware validate` **rejects** them so typos do not silently no-op:

```console
$ llmfit hardware validate ./my-workstation.json
FAIL  ./my-workstation.json: unknown key(s): hardware.gpu_bandwith_gbps
```

`--profile` conflicts with `--memory` / `--ram` / `--cpu-cores` (whole machine vs one field). An unresolvable profile is a hard error.

Full field list and bundled provenance: [`llmfit-core/data/hardware/README.md`](../llmfit-core/data/hardware/README.md).

Limitations today:

- `calibration[]` is stored for review but **not** applied to estimates (schema v1).
- `--profile` cannot combine with `--force-runtime` yet.
- `doctor` rejects `--profile` (it diagnoses *this* host). Use `hardware show` instead.

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

### Model library storage

Use `storage` to estimate SSD capacity for models you keep on disk and switch
between. It selects runnable models using the shared hardware fit analysis.
Each full catalog ID counts once; different repositories or format variants
remain distinct library entries.

```sh
# Top three models by fit score, with default storage allowances
llmfit storage --keep 3

# Conservative sizing: the largest three fitting models
llmfit --memory 128G --ram 128G --cpu-cores 18 \
  storage --keep 3 --selection largest --json

# Apply a hardware profile, context cap, and model search
llmfit --profile ryzen-ai-max-plus-395 --max-context 8192 \
  storage --search qwen --perfect --keep 3 --json

# Explicit allowances: 150 GB for OS/apps, 200 GB scratch, 20% free space
llmfit storage --keep 5 --os-reserve 150GB --scratch 200GB --headroom 20
```

Hardware flags (`--memory`, `--ram`, `--cpu-cores`, `--profile`) and
`--max-context` go **before** the subcommand. Capacity overrides retain the
host's backend and bandwidth; profiles apply their existing topology and
calculation settings. `OLLAMA_CONTEXT_LENGTH` supplies the context cap when
`--max-context` is absent.

| Option | Default | Meaning |
|---|---|---|
| `--keep N` | `3` | Maximum distinct catalog models to retain; must be positive |
| `--selection score\|largest` | `score` | Existing fit score ranking, or descending weight storage |
| `--os-reserve SIZE` | `100G` | Allowance for OS, apps, and other files |
| `--scratch auto\|SIZE` | `auto` | One largest-selected-model download; a size replaces this allowance |
| `--headroom PERCENT` | `15` | Percentage of suggested SSD capacity to leave free, 0–99 |
| `--perfect` | off | Only Perfect models; normally Good and Marginal also qualify |
| `--search QUERY` | none | Case-insensitive name, provider, or parameter-size filter |
| `--json` | off | Structured output; text is the default, CSV is unsupported |

The calculation uses each model's selected `best_quant`:

```text
library_gb          = sum(selected disk_size_gb)
download_scratch_gb = largest selected disk_size_gb, or the explicit allowance
need_gb             = os_reserve_gb + library_gb + download_scratch_gb
target_capacity_gb  = need_gb / (1 - headroom_percent / 100)
```

`minimum_ssd_gb` rounds `need_gb` up to the first suitable tier;
`suggested_ssd_gb` rounds `target_capacity_gb` up. The generic tiers are
256, 512, 1000, 2000, 4000, 8000, and 16000 **decimal GB**. For example, a
500 GB requirement fits a 512 GB minimum, but 15% free headroom increases
the suggested capacity to 1000 GB. These tiers are capacity categories;
availability depends on the device. The reserve and headroom are adjustable
planning policies, not measured requirements.

Storage sizes use decimal `M`/`MB`, `G`/`GB`, and `T`/`TB`; bare numbers are
GB. Explicit `MiB`, `GiB`, and `TiB` suffixes use binary bytes converted to
decimal GB: `1TB` is 1000 GB, `1TiB` is approximately 1099.51 GB. Suffixes
are case-insensitive. This storage parser intentionally differs from the
legacy hardware memory parser. Zero reserve and zero explicit scratch are
allowed. Calculations keep full precision before selecting a tier.

JSON contains `system` (the usual hardware summary) and `storage`, including
the selected `models`, `selection`, `keep_requested`, `selected_count`,
`eligible_count`, the numeric fields above, `minimum_ssd_gb`,
`suggested_ssd_gb`, `perfect`, `estimate_notice`, and `warnings`.
`scratch_policy` is `{"mode":"auto"}` or `{"mode":"fixed","size_gb":200.0}`.
Each model includes `name`, `best_quant`, `fit_level`, `runtime`, `score`,
`disk_size_gb`, and `effective_context_length`. Runtime values use the core
enum names (`Mlx`, `LlamaCpp`, `Vllm`).

When fewer than N models qualify, the report includes the available models
and a warning. When none qualify, weights and scratch are zero, `need_gb`
contains only the reserve, and both SSD recommendations are null. When a
requirement exceeds the largest tier, the affected recommendation is null
with a warning; an undersized drive is never recommended. These reports exit
successfully. Invalid storage sizes, selection data, or output options exit
1 and use the usual JSON error envelope when `--json` is set; malformed CLI
syntax exits 2. Hardware/profile errors retain the existing CLI behavior.

Weights are approximate and include all MoE experts. Existing catalog
quantization estimates are reused; actual downloaded artifacts, auxiliary
files, and runtime caches may differ. Installed models still count toward
the total. The command models one library copy with sequential model use;
it does not scan free disk space, download files, estimate concurrent
serving capacity, or calculate storage replicated across cluster nodes.
Automatic scratch covers one comparable extra download; use an explicit
allowance for larger future models or conversion caches.

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
- `disk_size_gb`: estimated weight storage in decimal GB at the planned quantization
- estimated minimum/recommended hardware
- per-path feasibility (`gpu`, `cpu_offload`, `cpu_only`)
- upgrade deltas

`disk_size_gb` excludes KV cache, inference buffers, and download scratch.
MoE models include all stored experts. `fit`, `recommend`, and `info` report
disk size at their selected `best_quant`; `plan` uses `--quant` or the model's
catalog default. Compare the same quantization when comparing these outputs.
The estimate uses the existing parameter-count and quantization formula, so
actual downloaded files and auxiliary assets can differ.

---
