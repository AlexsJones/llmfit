# llmfit

<p align="center">
  <img src="assets/icon.svg" alt="llmfit icon" width="128" height="128">
</p>

<p align="center">
  <b>English</b> ·
  <a href="README.zh.md">中文</a>
</p>

<p align="center">
  <a href="https://github.com/AlexsJones/llmfit/actions/workflows/ci.yml"><img src="https://github.com/AlexsJones/llmfit/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://crates.io/crates/llmfit"><img src="https://img.shields.io/crates/v/llmfit.svg" alt="Crates.io"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="https://about.signpath.io"><img src="https://img.shields.io/badge/SignPath-signed-brightgreen?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiIgZmlsbD0id2hpdGUiIHZpZXdCb3g9IjAgMCAxNiAxNiI+PHBhdGggZD0iTTEwLjA2NyA0LjU2N2wtNC43MzQgNC43MzMtMS40LTEuNGExIDEgMCAwIDAtMS40MTQgMS40MTRsMi4xIDIuMWExIDEgMCAwIDAgMS40MTQgMGw1LjQ0LTUuNDRhMSAxIDAgMCAwLTEuNDE0LTEuNDE0eiIvPjwvc3ZnPg==" alt="Signed with SignPath"></a>
</p>

> **Windows binaries are now code-signed!** All Windows releases are signed via [SignPath Foundation](https://signpath.org), providing Authenticode signatures so you can trust that downloads haven't been tampered with.

**Hundreds of models & providers. One command to find what runs on your hardware.**

A terminal tool that right-sizes LLM models to your system's RAM, CPU, and GPU. Detects your hardware, scores each model across quality, speed, fit, and context dimensions, and tells you which ones will actually run well on your machine.

Ships with an interactive TUI (default) and a classic CLI mode. Supports multi-GPU setups, MoE architectures, dynamic quantization selection, speed estimation, and local runtime providers (Ollama, llama.cpp, MLX, Docker Model Runner, LM Studio).

**New: [Download Manager](#download-manager-d) (`D`), [Advanced Configuration](#advanced-configuration-a) (`A`), and [Hardware Simulation](#hardware-simulation-s)** — Press `D` to manage downloads, view history, delete models, and configure the download directory. Press `A` to tune TPS efficiency, run mode factors, and scoring weights. Press `S` to simulate different hardware.

> **Sister projects:**
> - [sympozium](https://github.com/sympozium-ai/sympozium/) — managing agents in Kubernetes.
> - [llmserve](https://github.com/AlexsJones/llmserve) — a simple TUI for serving local LLM models. Pick a model, pick a backend, serve it.
> - [llama-panel](https://github.com/AlexsJones/llama-panel) — a native macOS app for managing local llama-server instances.

![demo](demo.gif)

---

## Install

### Windows
```sh
scoop install llmfit
```

If Scoop is not installed, follow the [Scoop installation guide](https://scoop.sh/).

### macOS / Linux

#### Homebrew
```sh
brew install llmfit
```

### MacPorts
```sh
port install llmfit
```

#### Quick install
```sh
curl -fsSL https://llmfit.axjns.dev/install.sh | sh
```

Downloads the latest release binary from GitHub and installs it to `/usr/local/bin` (or `~/.local/bin` if no sudo).

**Install to `~/.local/bin` without sudo:**
```sh
curl -fsSL https://llmfit.axjns.dev/install.sh | sh -s -- --local
```

### Docker / Podman
```sh
docker run ghcr.io/alexsjones/llmfit
```
This prints JSON from `llmfit recommend` command. The JSON could be further queried with `jq`.
```
podman run ghcr.io/alexsjones/llmfit recommend --use-case coding | jq '.models[].name'
```

### From source
```sh
git clone https://github.com/AlexsJones/llmfit.git
cd llmfit
cargo build --release
# binary is at target/release/llmfit
```

---

## Usage

### TUI (default)

```sh
llmfit
```

Launches the interactive terminal UI. Your system specs (CPU, RAM, GPU name, VRAM, backend) are shown at the top. Models are listed in a scrollable table sorted by composite score. Each row shows the model's score, estimated tok/s, best quantization for your hardware, run mode, memory usage, and use-case category.

| Key                        | Action                                                                |
|----------------------------|-----------------------------------------------------------------------|
| `Up` / `Down` or `j` / `k` | Navigate models                                                       |
| `/`                        | Enter search mode (partial match on name, provider, params, use case) |
| `Esc` or `Enter`           | Exit search mode                                                      |
| `Ctrl-U`                   | Clear search                                                          |
| `f`                        | Cycle fit filter: All, Runnable, Perfect, Good, Marginal              |
| `a`                        | Cycle availability filter: All, GGUF Avail, Installed                 |
| `s`                        | Cycle sort column: Score, Params, Mem%, Ctx, Date, Use Case           |
| `v`                        | Enter Visual mode (select multiple models)                            |
| `V`                        | Enter Select mode (column-based filtering)                            |
| `t`                        | Cycle color theme (saved automatically)                               |
| `p`                        | Open Plan mode for selected model (hardware planning)                 |
| `P`                        | Open provider filter popup                                            |
| `U`                        | Open use-case filter popup                                            |
| `C`                        | Open capability filter popup                                          |
| `L`                        | Open license filter popup                                             |
| `R`                        | Open runtime/backend filter popup (llama.cpp, MLX, vLLM)             |
| `S`                        | Open hardware simulation popup (override RAM/VRAM/CPU)                |
| `A`                        | Open advanced configuration popup (tune efficiency, run mode factors) |
| `h`                        | Open help popup (all key bindings)                                    |
| `m`                        | Mark selected model for compare                                       |
| `c`                        | Open compare view (marked vs selected)                                |
| `x`                        | Clear compare mark                                                    |
| `i`                        | Toggle installed-first sorting (any detected runtime provider)        |
| `d`                        | Download selected model (provider picker when multiple are available) |
| `D`                        | Open Download Manager (history, deletion, config)                     |
| `r`                        | Refresh installed models from runtime providers                       |
| `Enter`                    | Toggle detail view for selected model                                 |
| `PgUp` / `PgDn`            | Scroll by 10                                                          |
| `g` / `G`                  | Jump to top / bottom                                                  |
| `q`                        | Quit                                                                  |

### Vim-like modes

The TUI uses Vim-inspired modes shown in the bottom-left status bar. The current mode determines which keys are active.

#### Normal mode

The default mode. Navigate, search, filter, and open views. All keys in the table above apply here.

#### Visual mode (`v`)

Select a contiguous range of models for bulk comparison. Press `v` to anchor at the current row, then navigate with `j`/`k` or arrow keys to extend the selection. Selected rows are highlighted.

| Key                 | Action                                                 |
|---------------------|--------------------------------------------------------|
| `j` / `k` or arrows | Extend selection up/down                               |
| `c`                 | Compare all selected models (opens multi-compare view) |
| `m`                 | Mark current model for two-model compare               |
| `Esc` or `v`        | Exit Visual mode                                       |

The multi-compare view displays a table where rows are attributes (Score, tok/s, Fit, Mem%, Params, Mode, Context, Quant, etc.) and columns are models. Best values are highlighted. Use `h`/`l` or arrow keys to scroll horizontally if more models are selected than fit on screen.

#### Select mode (`V`)

Column-based actions. Press `V` (shift-v) to enter Select mode, then use `h`/`l` or arrow keys to move between column headers. The active column is visually highlighted. Press `Enter` or `Space` to trigger that column's current action.

| Column                        | Filter action                                                             |
|-------------------------------|---------------------------------------------------------------------------|
| Inst                          | Cycle availability filter                                                 |
| Model                         | Enter search mode                                                         |
| Provider                      | Open provider popup                                                       |
| Params                        | Open parameter-size bucket popup (<3B, 3-7B, 7-14B, 14-30B, 30-70B, 70B+) |
| Score, tok/s, Mem%, Ctx, Date | Sort by that column                                                       |
| Quant                         | Open quantization popup                                                   |
| Mode                          | Open run-mode popup (GPU, MoE, CPU+GPU, CPU)                              |
| Fit                           | Cycle fit filter                                                          |
| Use Case                      | Open use-case popup                                                       |

Row navigation still works in Select mode so you can see the effect of actions as you apply them: `j`/`k`, arrow keys, `Ctrl-U`, `Ctrl-D`, `PageUp`, `PageDown`, `Home`, and `End`. Press `Esc` to return to Normal mode.

### TUI Plan mode (`p`)

Plan mode inverts normal fit analysis: instead of asking "what fits my hardware?", it estimates "what hardware is needed for this model config?".

Use `p` on a selected row, then:

| Key                    | Action                                                    |
|------------------------|-----------------------------------------------------------|
| `Tab` / `j` / `k`      | Move between editable fields (Context, Quant, Target TPS) |
| `Left` / `Right`       | Move cursor in current field                              |
| Type                   | Edit current field                                        |
| `Backspace` / `Delete` | Remove characters                                         |
| `Ctrl-U`               | Clear current field                                       |
| `Esc` or `q`           | Exit Plan mode                                            |

Plan mode shows estimates for:
- minimum and recommended VRAM/RAM/CPU cores
- feasible run paths (GPU, CPU offload, CPU-only)
- upgrade deltas to reach better fit targets

### Hardware Simulation (`S`)

Press `S` to open the hardware simulation popup. Override RAM, VRAM, and CPU core count to see which models would fit on different target hardware. All model scores, fit levels, and speed estimates are recalculated instantly against the simulated specs.

![Hardware Simulation](simulation.png)

| Key                    | Action                                  |
|------------------------|-----------------------------------------|
| `Tab` / `j` / `k`      | Switch between RAM, VRAM, CPU fields    |
| Type digits            | Edit the selected field                 |
| `Enter`               | Apply simulation                        |
| `Ctrl-R`              | Reset to real detected hardware         |
| `Esc`                 | Cancel and close                        |

When simulation is active, a `SIM` badge appears in the system bar and status bar. The entire model table reflects the simulated hardware until you reset.

### Advanced Configuration (`A`)

Press `A` to open the Advanced Configuration popup. This panel lets you tune the parameters behind TPS estimation, run mode penalties, and composite scoring — addressing [issue #449](https://github.com/AlexsJones/llmfit/issues/449) where tok/s was overestimated for certain models (e.g., Qwen3 30B).

All changes are applied immediately and the model table is recalculated. Close with `Esc` to accept or `Ctrl-R` to reset to defaults.

| Field              | Description                                                             | Default |
|--------------------|-------------------------------------------------------------------------|---------|
| **Efficiency**     | Global efficiency factor for bandwidth-based TPS. Accounts for overhead | `0.55`  |
| **GPU factor**     | Speed multiplier for pure GPU inference                                 | `1.0`   |
| **CPU Offload**    | Speed multiplier when weights spill to system RAM                       | `0.5`   |
| **MoE Offload**    | Speed multiplier for Mixture-of-Experts expert switching                | `0.8`   |
| **Tensor Par**     | Speed multiplier for tensor-parallel inference                          | `0.9`   |
| **CPU Only**       | Speed multiplier for CPU-only execution                                 | `0.3`   |
| **Context cap**    | Max context length used for memory estimation (leave blank for default) | `auto`  |

| Key                    | Action                                  |
|------------------------|-----------------------------------------|
| `Tab` / `j` / `k`      | Switch between fields                   |
| Type digits / `.`      | Edit the selected field                 |
| `Left` / `Right`       | Move cursor within the field            |
| `Backspace` / `Delete` | Remove characters                       |
| `Ctrl-U`               | Clear the current field                 |
| `Enter`                | Apply changes and recalculate all scores|
| `Esc` / `q`            | Close without applying                  |

### Download Manager (`D`)

Press `D` to open the Download Manager view. This full-screen view replaces the main model table and provides three sections:

- **Active Download** — shows the current download in progress with a progress bar, model name, and status message.
- **Config** — displays (and allows editing) the GGUF models directory. The configured path persists across sessions.
- **History** — a navigable list of past downloads (newest first) with model name, provider, status, and date. Failed downloads can be removed from history, and successful downloads can be deleted from the provider.

Use `Tab` / `Shift-Tab` to cycle focus between sections.

| Key                    | Action                                           |
|------------------------|--------------------------------------------------|
| `Tab` / `Shift-Tab`   | Cycle focus: Active → Config → History           |
| `j` / `k` or arrows   | Navigate the history list (when History focused)  |
| `x`                   | Delete selected model (prompts for confirmation)  |
| `y` / `n`             | Confirm or cancel deletion                        |
| `e`                   | Edit download directory (when Config focused)     |
| `Enter`               | Confirm directory edit                            |
| `Esc` / `D` / `q`    | Close and return to the model table               |

For failed downloads (e.g. 404 errors), `x` removes the entry from history. For successful downloads, it deletes the model from the provider (supported for Ollama and llama.cpp).

### Themes

Press `t` to cycle through 10 built-in color themes. Your selection is saved automatically to `~/.config/llmfit/theme` and restored on next launch.

| Theme                    | Description                                       |
|--------------------------|---------------------------------------------------|
| **Default**              | Original llmfit colors                            |
| **Dracula**              | Dark purple background with pastel accents        |
| **Solarized**            | Ethan Schoonover's Solarized Dark palette         |
| **Nord**                 | Arctic, cool blue-gray tones                      |
| **Monokai**              | Monokai Pro warm syntax colors                    |
| **Gruvbox**              | Retro groove palette with warm earth tones        |
| **Catppuccin Latte**     | 🌻 Light theme — harmonious pastel inversion      |
| **Catppuccin Frappé**    | 🪴 Low-contrast dark — muted, subdued aesthetic   |
| **Catppuccin Macchiato** | 🌺 Medium-contrast dark — gentle, soothing tones  |
| **Catppuccin Mocha**     | 🌿 Darkest variant — cozy with color-rich accents |

### Web dashboard

When you run `llmfit` in non-JSON mode, it automatically starts a background web dashboard on `0.0.0.0:8787`. Open it in any browser on the same network:

```
http://<your-machine-ip>:8787
```

Override the host or port with environment variables:

```sh
LLMFIT_DASHBOARD_HOST=0.0.0.0 LLMFIT_DASHBOARD_PORT=9000 llmfit
```

| Variable | Default | Description |
|---|---|---|
| `LLMFIT_DASHBOARD_HOST` | `0.0.0.0` | Interface to bind the dashboard server |
| `LLMFIT_DASHBOARD_PORT` | `8787` | Port to bind the dashboard server |

To disable the auto-started dashboard, pass `--no-dashboard`:

```sh
llmfit --no-dashboard
```

### CLI mode

Use `--cli` or any subcommand to get classic table output:

```sh
# Table of all models ranked by fit
llmfit --cli

# Only perfectly fitting models, top 5
llmfit fit --perfect -n 5

# Show detected system specs
llmfit system

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

## How it works

1. **Hardware detection** -- Reads total/available RAM via `sysinfo`, counts CPU cores, and probes for GPUs:
   - **NVIDIA** -- Multi-GPU support via `nvidia-smi`. Aggregates VRAM across all detected GPUs. Falls back to VRAM estimation from GPU model name if reporting fails.
   - **AMD** -- Detected via `rocm-smi`.
   - **Intel Arc** -- Discrete VRAM via sysfs, integrated via `lspci`.
   - **Apple Silicon** -- Unified memory via `system_profiler`. VRAM = system RAM.
   - **Ascend** -- Detected via `npu-smi`.
   - **Backend detection** -- Automatically identifies the acceleration backend (CUDA, Metal, ROCm, SYCL, CPU ARM, CPU x86, Ascend) for speed estimation.

2. **Model database** -- Hundreds models sourced from the HuggingFace API, stored in `data/hf_models.json` and embedded at compile time. Memory requirements are computed from parameter counts across a quantization hierarchy (Q8_0 through Q2_K). VRAM is the primary constraint for GPU inference; system RAM is the fallback for CPU-only execution.

   **MoE support** -- Models with Mixture-of-Experts architectures (Mixtral, DeepSeek-V2/V3) are detected automatically. Only a subset of experts is active per token, so the effective VRAM requirement is much lower than total parameter count suggests. For example, Mixtral 8x7B has 46.7B total parameters but only activates ~12.9B per token, reducing VRAM from 23.9 GB to ~6.6 GB with expert offloading.

3. **Dynamic quantization** -- Instead of assuming a fixed quantization, llmfit tries the best quality quantization that fits your hardware. It walks a hierarchy from Q8_0 (best quality) down to Q2_K (most compressed), picking the highest quality that fits in available memory. If nothing fits at full context, it tries again at half context.

4. **Multi-dimensional scoring** -- Each model is scored across four dimensions (0–100 each):

   | Dimension   | What it measures                                                               |
   |-------------|--------------------------------------------------------------------------------|
   | **Quality** | Parameter count, model family reputation, quantization penalty, task alignment |
   | **Speed**   | Estimated tokens/sec based on backend, params, and quantization                |
   | **Fit**     | Memory utilization efficiency (sweet spot: 50–80% of available memory)         |
   | **Context** | Context window capability vs target for the use case                           |

   Dimensions are combined into a weighted composite score. Weights vary by use-case category (General, Coding, Reasoning, Chat, Multimodal, Embedding). For example, Chat weights Speed higher (0.35) while Reasoning weights Quality higher (0.55). Models are ranked by composite score, with unrunnable models (Too Tight) always at the bottom.

5. **Speed estimation** -- Token generation in LLM inference is memory-bandwidth-bound: each token requires reading the full model weights once from VRAM. When the GPU model is recognized, llmfit uses its actual memory bandwidth to estimate throughput:

   Formula: `(bandwidth_GB_s / model_size_GB) × efficiency_factor`

   The efficiency factor (0.55) and per-mode speed multipliers are tunable via the Advanced Configuration popup (`A` in the TUI). The defaults account for kernel overhead, KV-cache reads, and memory controller effects. This approach is validated against published benchmarks from llama.cpp ([Apple Silicon](https://github.com/ggml-org/llama.cpp/discussions/4167), [NVIDIA T4](https://github.com/ggml-org/llama.cpp/discussions/4225)) and real-world measurements.

   The bandwidth lookup table covers ~80 GPUs across NVIDIA (consumer + datacenter), AMD (RDNA + CDNA), and Apple Silicon families.

   For unrecognized GPUs, llmfit falls back to per-backend speed constants:

   | Backend      | Speed constant |
   |--------------|----------------|
   | CUDA         | 220            |
   | Metal        | 160            |
   | ROCm         | 180            |
   | SYCL         | 100            |
   | CPU (ARM)    | 90             |
   | CPU (x86)    | 70             |
   | NPU (Ascend) | 390            |

   Fallback formula: `K / params_b × quant_speed_multiplier`, with per-mode penalties tunable via the Advanced Configuration popup (`A` in the TUI).

6. **Fit analysis** -- Each model is evaluated for memory compatibility:

   **Run modes:**
   - **GPU** -- Model fits in VRAM. Fast inference.
   - **MoE** -- Mixture-of-Experts with expert offloading. Active experts in VRAM, inactive in RAM.
   - **CPU+GPU** -- VRAM insufficient, spills to system RAM with partial GPU offload.
   - **CPU** -- No GPU. Model loaded entirely into system RAM.

   **Fit levels:**
   - **Perfect** -- Recommended memory met on GPU. Requires GPU acceleration.
   - **Good** -- Fits with headroom. Best achievable for MoE offload or CPU+GPU.
   - **Marginal** -- Tight fit, or CPU-only (CPU-only always caps here).
   - **Too Tight** -- Not enough VRAM or system RAM anywhere.

---

## Model database

The model list is generated by `scripts/scrape_hf_models.py`, a standalone Python script (stdlib only, no pip dependencies) that queries the HuggingFace REST API. Hundreds models & providers including Meta Llama, Mistral, Qwen, Google Gemma, Microsoft Phi, DeepSeek, IBM Granite, Allen Institute OLMo, xAI Grok, Cohere, BigCode, 01.ai, Upstage, TII Falcon, HuggingFace, Zhipu GLM, Moonshot Kimi, Baidu ERNIE, and more. The scraper automatically detects MoE architectures via model config (`num_local_experts`, `num_experts_per_tok`) and known architecture mappings.

Model categories span general purpose, coding (CodeLlama, StarCoder2, WizardCoder, Qwen2.5-Coder, Qwen3-Coder), reasoning (DeepSeek-R1, Orca-2), multimodal/vision (Llama 3.2 Vision, Llama 4 Scout/Maverick, Qwen2.5-VL), chat, enterprise (IBM Granite), and embedding (nomic-embed, bge).

See [MODELS.md](MODELS.md) for the full list.

The model database is embedded at compile time, so **end users** get updates by upgrading llmfit itself (`brew upgrade llmfit`, `scoop update llmfit`, or downloading a newer release). The commands below are for **contributors** refreshing the database from source:

To refresh the model database:

```sh
# Automated update (recommended)
make update-models

# Or run the script directly
./scripts/update_models.sh

# Or manually
python3 scripts/scrape_hf_models.py
cargo build --release
```

The scraper writes `data/hf_models.json`, which is baked into the binary via `include_str!`. The automated update script backs up existing data, validates JSON output, and rebuilds the binary.

By default, the scraper enriches models with known GGUF download sources from providers like [unsloth](https://huggingface.co/unsloth) and [bartowski](https://huggingface.co/bartowski). Results are cached in `data/gguf_sources_cache.json` (7-day TTL) to avoid repeated API calls. Use `--no-gguf-sources` to skip enrichment for a faster scrape.

---

## Project structure

```
src/
  main.rs         -- CLI argument parsing, entrypoint, TUI launch
  hardware.rs     -- System RAM/CPU/GPU detection (multi-GPU, backend identification)
  models.rs       -- Model database, quantization hierarchy, dynamic quant selection
  fit.rs          -- Multi-dimensional scoring (Q/S/F/C), speed estimation, MoE offloading
  providers.rs    -- Runtime provider integration (Ollama, llama.cpp, MLX, Docker Model Runner, LM Studio), install detection, pull/download
  display.rs      -- Classic CLI table rendering + JSON output
  tui_app.rs      -- TUI application state, filters, navigation
  tui_ui.rs       -- TUI rendering (ratatui)
  tui_events.rs   -- TUI keyboard event handling (crossterm)
data/
  hf_models.json  -- Model database (206 models)
skills/
  llmfit-advisor/ -- OpenClaw skill for hardware-aware model recommendations
scripts/
  scrape_hf_models.py        -- HuggingFace API scraper
  update_models.sh            -- Automated database update script
  install-openclaw-skill.sh   -- Install the OpenClaw skill
Makefile           -- Build and maintenance commands
```

---

## Publishing to crates.io

The `Cargo.toml` already includes the required metadata (description, license, repository). To publish:

```sh
# Dry run first to catch issues
cargo publish --dry-run

# Publish for real (requires a crates.io API token)
cargo login
cargo publish
```

Before publishing, make sure:

- The version in `Cargo.toml` is correct (bump with each release).
- A `LICENSE` file exists in the repo root. Create one if missing:

```sh
# For MIT license:
curl -sL https://opensource.org/license/MIT -o LICENSE
# Or write your own. The Cargo.toml declares license = "MIT".
```

- `data/hf_models.json` is committed. It is embedded at compile time and must be present in the published crate.
- The `exclude` list in `Cargo.toml` keeps `target/`, `scripts/`, and `demo.gif` out of the published crate to keep the download small.

To publish updates:

```sh
# Bump version
# Edit Cargo.toml: version = "0.2.0"
cargo publish
```

---

## Dependencies

| Crate                  | Purpose                                          |
|------------------------|--------------------------------------------------|
| `clap`                 | CLI argument parsing with derive macros          |
| `sysinfo`              | Cross-platform RAM and CPU detection             |
| `serde` / `serde_json` | JSON deserialization for model database          |
| `tabled`               | CLI table formatting                             |
| `colored`              | CLI colored output                               |
| `ureq`                 | HTTP client for runtime/provider API integration |
| `ratatui`              | Terminal UI framework                            |
| `crossterm`            | Terminal input/output backend for ratatui        |

---

## Runtime provider integration

llmfit supports multiple local runtime providers:

- **Ollama** (daemon/API based pulls)
- **llama.cpp** (direct GGUF downloads from Hugging Face + local cache detection)
- **MLX** (Apple Silicon / mlx-community model cache + optional server) — MLX downloads map to `mlx-community/*` repos on HuggingFace, not the original model publisher
- **Docker Model Runner** (Docker Desktop's built-in model serving)
- **LM Studio** (local model server with REST API for model management + downloads)

When more than one compatible provider is available for a model, pressing `d` in the TUI opens a provider picker modal.

### Ollama integration

llmfit integrates with [Ollama](https://ollama.com) to detect which models you already have installed and to download new ones directly from the TUI.

### Requirements

- **Ollama must be installed and running** (`ollama serve` or the Ollama desktop app)
- llmfit connects to `http://localhost:11434` (Ollama's default API port)
- No configuration needed — if Ollama is running, llmfit detects it automatically

### Remote Ollama instances

To connect to Ollama running on a different machine or port, set the `OLLAMA_HOST` environment variable:

```sh
# Connect to Ollama on a specific IP and port
OLLAMA_HOST="http://192.168.1.100:11434" llmfit

# Connect via hostname  
OLLAMA_HOST="http://ollama-server:666" llmfit

# Works with all TUI and CLI commands
OLLAMA_HOST="http://192.168.1.100:11434" llmfit --cli
OLLAMA_HOST="http://192.168.1.100:11434" llmfit fit --perfect -n 5
```

This is useful for:
- Running llmfit on one machine while Ollama serves from another (e.g., GPU server + laptop client)
- Connecting to Ollama running in Docker containers with custom ports
- Using Ollama behind reverse proxies or load balancers

### How it works

On startup, llmfit queries `GET /api/tags` to list your installed Ollama models. Each installed model gets a green **✓** in the **Inst** column of the TUI. The system bar shows `Ollama: ✓ (N installed)`.

When you press `d` on a model, llmfit sends `POST /api/pull` to Ollama to download it. The row highlights with an animated progress indicator showing download progress in real-time. Once complete, the model is immediately available for use with Ollama.

If Ollama is not running, Ollama-specific operations are skipped; the TUI still supports other providers like llama.cpp where available.

### llama.cpp integration

llmfit integrates with [llama.cpp](https://github.com/ggml-org/llama.cpp) as a runtime/download provider in both TUI and CLI.

Requirements:

- `llama-cli` or `llama-server` available in `PATH` (for runtime detection)
- network access to Hugging Face for GGUF downloads

How it works:

- llmfit maps HF models to known GGUF repos (with heuristic fallbacks)
- downloads GGUF files into the local llama.cpp model cache
- marks models installed when matching GGUF files are present locally

#### Environment variables

| Variable | Default | Description |
|---|---|---|
| `LLAMA_CPP_PATH` | *(none)* | Directory containing llama.cpp binaries (`llama-cli`, `llama-server`). Checked before `PATH` lookup. |
| `LLAMA_SERVER_PORT` | `8080` | Port used when probing a running `llama-server` health endpoint for runtime detection. |

If llama.cpp is installed in a non-standard location, set `LLAMA_CPP_PATH` so llmfit can find it without requiring it in your `PATH`.

### Docker Model Runner integration

llmfit integrates with [Docker Model Runner](https://docs.docker.com/desktop/features/model-runner/), Docker Desktop's built-in model serving feature.

Requirements:

- Docker Desktop with Model Runner enabled
- Default endpoint: `http://localhost:12434`

How it works:

- llmfit queries `GET /engines` to list models available in Docker Model Runner
- models are matched to the HF database using Ollama-style tag mapping (Docker Model Runner uses `ai/<tag>` naming)
- pressing `d` in the TUI pulls via `docker model pull`

### Remote Docker Model Runner instances

To connect to Docker Model Runner on a different host or port, set the `DOCKER_MODEL_RUNNER_HOST` environment variable:

```sh
DOCKER_MODEL_RUNNER_HOST="http://192.168.1.100:12434" llmfit
```

### LM Studio integration

llmfit integrates with [LM Studio](https://lmstudio.ai) as a local model server with built-in model download capabilities.

Requirements:

- LM Studio must be running with its local server enabled
- Default endpoint: `http://127.0.0.1:1234`

How it works:

- llmfit queries `GET /v1/models` to list models available in LM Studio
- pressing `d` in the TUI triggers a download via `POST /api/v1/models/download`
- download progress is tracked by polling `GET /api/v1/models/download-status`
- LM Studio accepts HuggingFace model names directly, so no name mapping is needed

### Remote LM Studio instances

To connect to LM Studio on a different host or port, set the `LMSTUDIO_HOST` environment variable:

```sh
LMSTUDIO_HOST="http://192.168.1.100:1234" llmfit
```

### Model name mapping

llmfit's database uses HuggingFace model names (e.g. `Qwen/Qwen2.5-Coder-14B-Instruct`) while Ollama uses its own naming scheme (e.g. `qwen2.5-coder:14b`). llmfit maintains an accurate mapping table between the two so that install detection and pulls resolve to the correct model. Each mapping is exact — `qwen2.5-coder:14b` maps to the Coder model, not the base `qwen2.5:14b`.

---

## Platform support

- **Linux** -- Full support. GPU detection via `nvidia-smi` (NVIDIA), `rocm-smi` (AMD), sysfs/`lspci` (Intel Arc) and `npu-smi` (Ascend).
- **macOS (Apple Silicon)** -- Full support. Detects unified memory via `system_profiler`. VRAM = system RAM (shared pool). Models run via Metal GPU acceleration.
- **macOS (Intel)** -- RAM and CPU detection works. Discrete GPU detection if `nvidia-smi` available.
- **Windows** -- RAM and CPU detection works. NVIDIA GPU detection via `nvidia-smi` if installed.
- **Android / Termux / PRoot** -- CPU and RAM detection usually work, but GPU autodetection is not currently supported. Mobile GPUs such as Adreno typically are not visible through the desktop/server probing interfaces llmfit uses.

### GPU support

| Vendor                 | Detection method              | VRAM reporting                 |
|------------------------|-------------------------------|--------------------------------|
| NVIDIA                 | `nvidia-smi`                  | Exact dedicated VRAM           |
| AMD                    | `rocm-smi`                    | Detected (VRAM may be unknown) |
| Intel Arc (discrete)   | sysfs (`mem_info_vram_total`) | Exact dedicated VRAM           |
| Intel Arc (integrated) | `lspci`                       | Shared system memory           |
| Apple Silicon          | `system_profiler`             | Unified memory (= system RAM)  |
| Ascend                 | `npu-smi`                     | Detected (VRAM may be unknown) |

If autodetection fails or reports incorrect values, use `--memory`, `--ram`, or `--cpu-cores` to override (see [Hardware overrides](#hardware-overrides) above).

### Android / Termux note

On Android setups such as **Termux + PRoot**, llmfit usually cannot see mobile GPUs through the standard Linux detection paths (`nvidia-smi`, `rocm-smi`, DRM/sysfs, `lspci`, etc.). In those environments, "no GPU detected" is expected with the current implementation.

If you still want GPU-style recommendations on a unified-memory phone or tablet, use a manual memory override:

```sh
llmfit --memory=8G fit -n 20
llmfit recommend --json --memory=8G --limit 10
```

This is a workaround for recommendation/scoring only; it does not provide true Android GPU runtime detection.

---

## Contributing

Contributions are welcome, especially new models.

### Before submitting a PR

Please run `cargo fmt` before pushing your changes. Most CI check failures are caused by unformatted code:

```sh
cargo fmt
```

### Adding a model

1. Add the model's HuggingFace repo ID (e.g., `meta-llama/Llama-3.1-8B`) to the `TARGET_MODELS` list in `scripts/scrape_hf_models.py`.
2. If the model is gated (requires HuggingFace authentication to access metadata), add a fallback entry to the `FALLBACKS` list in the same script with the parameter count and context length.
3. Run the automated update script:
   ```sh
   make update-models
   # or: ./scripts/update_models.sh
   ```
4. Verify the updated model list: `./target/release/llmfit list`
5. Update [MODELS.md](MODELS.md) by running: `python3 << 'EOF' < scripts/...` (see commit history for the generator script)
6. Open a pull request.

See [MODELS.md](MODELS.md) for the current list and [AGENTS.md](AGENTS.md) for architecture details.

---

## OpenClaw integration

llmfit ships as an [OpenClaw](https://github.com/openclaw/openclaw) skill that lets the agent recommend hardware-appropriate local models and auto-configure Ollama/vLLM/LM Studio providers.

### Install the skill

```sh
# From the llmfit repo
./scripts/install-openclaw-skill.sh

# Or manually
cp -r skills/llmfit-advisor ~/.openclaw/skills/
```

Once installed, ask your OpenClaw agent things like:

- "What local models can I run?"
- "Recommend a coding model for my hardware"
- "Set up Ollama with the best models for my GPU"

The agent will call `llmfit recommend --json` under the hood, interpret the results, and offer to configure your `openclaw.json` with optimal model choices.

### How it works

The skill teaches the OpenClaw agent to:

1. Detect your hardware via `llmfit --json system`
2. Get ranked recommendations via `llmfit recommend --json`
3. Map HuggingFace model names to Ollama/vLLM/LM Studio tags
4. Configure `models.providers.ollama.models` in `openclaw.json`

See [skills/llmfit-advisor/SKILL.md](skills/llmfit-advisor/SKILL.md) for the full skill definition.

---

## Alternatives

If you're looking for a different approach, check out [llm-checker](https://github.com/Pavelevich/llm-checker) -- a Node.js CLI tool with Ollama integration that can pull and benchmark models directly. It takes a more hands-on approach by actually running models on your hardware via Ollama, rather than estimating from specs. Good if you already have Ollama installed and want to test real-world performance. Note that it doesn't support MoE (Mixture-of-Experts) architectures -- all models are treated as dense, so memory estimates for models like Mixtral or DeepSeek-V3 will reflect total parameter count rather than the smaller active subset.

---

## License

MIT
