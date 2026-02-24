# llmfit

<p align="center">
  <img src="assets/icon.svg" alt="llmfit icon" width="128" height="128">
</p>

**206 models. 57 providers. One command to find what runs on your hardware.**

A terminal tool that right-sizes LLM models to your system's RAM, CPU, and GPU. Detects your hardware, scores each model across quality, speed, fit, and context dimensions, and tells you which ones will actually run well on your machine.

Ships with an interactive TUI (default) and a classic CLI mode. Supports multi-GPU setups, MoE architectures, dynamic quantization selection, and speed estimation.

### Quick install (macOS / Linux)

```sh
curl -fsSL https://llmfit.axjns.dev/install.sh | sh
```
_Downloads the latest release binary from GitHub and installs it to `/usr/local/bin` (or `~/.local/bin`)_

Or
```sh
brew tap AlexsJones/llmfit
brew install llmfit
```

Windows users: see the **Install** section below.

![demo](demo.gif)

Example of a medium performance home laptop

![home](home_laptop.png)


Example of models with Mixture-of-Experts architectures

![moe](moe.png)

Downloading a model via Ollama integration

![download](download.gif)
---

## Install

### Cargo (Windows / macOS / Linux)

```sh
cargo install llmfit
```

If `cargo` is not installed yet, install Rust via [rustup](https://rustup.rs/).

### macOS / Linux

#### Homebrew

```sh
brew tap AlexsJones/llmfit
brew install llmfit
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

| Key | Action |
|---|---|
| `Up` / `Down` or `j` / `k` | Navigate models |
| `/` | Enter search mode (partial match on name, provider, params, use case) |
| `Esc` or `Enter` | Exit search mode |
| `Ctrl-U` | Clear search |
| `f` | Cycle fit filter: All, Runnable, Perfect, Good, Marginal |
| `s` | Cycle sort column: Score, Params, Mem%, Ctx, Date, Use Case |
| `t` | Cycle color theme (saved automatically) |
| `p` | Open provider filter popup |
| `i` | Toggle installed-first sorting (Ollama only) |
| `d` | Pull/download selected model via Ollama |
| `r` | Refresh installed models from Ollama |
| `1`-`9` | Toggle provider visibility |
| `Enter` | Toggle detail view for selected model |
| `PgUp` / `PgDn` | Scroll by 10 |
| `g` / `G` | Jump to top / bottom |
| `q` | Quit |

### Themes

Press `t` to cycle through 6 built-in color themes. Your selection is saved automatically to `~/.config/llmfit/theme` and restored on next launch.

| Theme | Description |
|---|---|
| **Default** | Original llmfit colors |
| **Dracula** | Dark purple background with pastel accents |
| **Solarized** | Ethan Schoonover's Solarized Dark palette |
| **Nord** | Arctic, cool blue-gray tones |
| **Monokai** | Monokai Pro warm syntax colors |
| **Gruvbox** | Retro groove palette with warm earth tones |

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
```

### GPU memory override

GPU VRAM autodetection can fail on some systems (e.g. broken `nvidia-smi`, VMs, passthrough setups). Use `--memory` to manually specify your GPU's VRAM:

```sh
# Override with 32 GB VRAM
llmfit --memory=32G

# Megabytes also work (32000 MB ≈ 31.25 GB)
llmfit --memory=32000M

# Works with all modes: TUI, CLI, and subcommands
llmfit --memory=24G --cli
llmfit --memory=24G fit --perfect -n 5
llmfit --memory=24G system
llmfit --memory=24G info "Llama-3.1-70B"
llmfit --memory=24G recommend --json
```

Accepted suffixes: `G`/`GB`/`GiB` (gigabytes), `M`/`MB`/`MiB` (megabytes), `T`/`TB`/`TiB` (terabytes). Case-insensitive. If no GPU was detected, the override creates a synthetic GPU entry so models are scored for GPU inference.

### JSON output

Add `--json` to any subcommand for machine-readable output:

```sh
llmfit --json system     # Hardware specs as JSON
llmfit --json fit -n 10  # Top 10 fits as JSON
llmfit recommend --json  # Top 5 recommendations (JSON is default for recommend)
```

---

## How it works

1. **Hardware detection** -- Reads total/available RAM via `sysinfo`, counts CPU cores, and probes for GPUs:
   - **NVIDIA** -- Multi-GPU support via `nvidia-smi`. Aggregates VRAM across all detected GPUs. Falls back to VRAM estimation from GPU model name if reporting fails.
   - **AMD** -- Detected via `rocm-smi`.
   - **Intel Arc** -- Discrete VRAM via sysfs, integrated via `lspci`.
   - **Apple Silicon** -- Unified memory via `system_profiler`. VRAM = system RAM.
   - **Backend detection** -- Automatically identifies the acceleration backend (CUDA, Metal, ROCm, SYCL, CPU ARM, CPU x86) for speed estimation.

2. **Model database** -- 206 models sourced from the HuggingFace API, stored in `data/hf_models.json` and embedded at compile time. Memory requirements are computed from parameter counts across a quantization hierarchy (Q8_0 through Q2_K). VRAM is the primary constraint for GPU inference; system RAM is the fallback for CPU-only execution.

   **MoE support** -- Models with Mixture-of-Experts architectures (Mixtral, DeepSeek-V2/V3) are detected automatically. Only a subset of experts is active per token, so the effective VRAM requirement is much lower than total parameter count suggests. For example, Mixtral 8x7B has 46.7B total parameters but only activates ~12.9B per token, reducing VRAM from 23.9 GB to ~6.6 GB with expert offloading.

3. **Dynamic quantization** -- Instead of assuming a fixed quantization, llmfit tries the best quality quantization that fits your hardware. It walks a hierarchy from Q8_0 (best quality) down to Q2_K (most compressed), picking the highest quality that fits in available memory. If nothing fits at full context, it tries again at half context.

4. **Multi-dimensional scoring** -- Each model is scored across four dimensions (0–100 each):

   | Dimension | What it measures |
   |---|---|
   | **Quality** | Parameter count, model family reputation, quantization penalty, task alignment |
   | **Speed** | Estimated tokens/sec based on backend, params, and quantization |
   | **Fit** | Memory utilization efficiency (sweet spot: 50–80% of available memory) |
   | **Context** | Context window capability vs target for the use case |

   Dimensions are combined into a weighted composite score. Weights vary by use-case category (General, Coding, Reasoning, Chat, Multimodal, Embedding). For example, Chat weights Speed higher (0.35) while Reasoning weights Quality higher (0.55). Models are ranked by composite score, with unrunnable models (Too Tight) always at the bottom.

5. **Speed estimation** -- Estimated tokens per second using backend-specific constants:

   | Backend | Speed constant |
   |---|---|
   | CUDA | 220 |
   | Metal | 160 |
   | ROCm | 180 |
   | SYCL | 100 |
   | CPU (ARM) | 90 |
   | CPU (x86) | 70 |

   Formula: `K / params_b × quant_speed_multiplier`, with penalties for CPU offload (0.5×), CPU-only (0.3×), and MoE expert switching (0.8×).

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

The model list is generated by `scripts/scrape_hf_models.py`, a standalone Python script (stdlib only, no pip dependencies) that queries the HuggingFace REST API. 206 models across 57 providers including Meta Llama, Mistral, Qwen, Google Gemma, Microsoft Phi, DeepSeek, IBM Granite, Allen Institute OLMo, xAI Grok, Cohere, BigCode, 01.ai, Upstage, TII Falcon, HuggingFace, Zhipu GLM, Moonshot Kimi, Baidu ERNIE, and more. The scraper automatically detects MoE architectures via model config (`num_local_experts`, `num_experts_per_tok`) and known architecture mappings.

Model categories span general purpose, coding (CodeLlama, StarCoder2, WizardCoder, Qwen2.5-Coder, Qwen3-Coder), reasoning (DeepSeek-R1, Orca-2), multimodal/vision (Llama 3.2 Vision, Llama 4 Scout/Maverick, Qwen2.5-VL), chat, enterprise (IBM Granite), and embedding (nomic-embed, bge).

See [MODELS.md](MODELS.md) for the full list.

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

---

## Project structure

```
src/
  main.rs         -- CLI argument parsing, entrypoint, TUI launch
  hardware.rs     -- System RAM/CPU/GPU detection (multi-GPU, backend identification)
  models.rs       -- Model database, quantization hierarchy, dynamic quant selection
  fit.rs          -- Multi-dimensional scoring (Q/S/F/C), speed estimation, MoE offloading
  providers.rs    -- Runtime provider integration (Ollama), model install detection, pull/download
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

| Crate | Purpose |
|---|---|
| `clap` | CLI argument parsing with derive macros |
| `sysinfo` | Cross-platform RAM and CPU detection |
| `serde` / `serde_json` | JSON deserialization for model database |
| `tabled` | CLI table formatting |
| `colored` | CLI colored output |
| `ureq` | HTTP client for Ollama API integration |
| `ratatui` | Terminal UI framework |
| `crossterm` | Terminal input/output backend for ratatui |

---

## Ollama integration

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

If Ollama is not running, the `d`, `i`, and `r` keybindings are hidden from the status bar and disabled — the TUI works normally without Ollama, you just can't see install status or pull models.

### Model name mapping

llmfit's database uses HuggingFace model names (e.g. `Qwen/Qwen2.5-Coder-14B-Instruct`) while Ollama uses its own naming scheme (e.g. `qwen2.5-coder:14b`). llmfit maintains an accurate mapping table between the two so that install detection and pulls resolve to the correct model. Each mapping is exact — `qwen2.5-coder:14b` maps to the Coder model, not the base `qwen2.5:14b`.

---

## Platform support

- **Linux** -- Full support. GPU detection via `nvidia-smi` (NVIDIA), `rocm-smi` (AMD), and sysfs/`lspci` (Intel Arc).
- **macOS (Apple Silicon)** -- Full support. Detects unified memory via `system_profiler`. VRAM = system RAM (shared pool). Models run via Metal GPU acceleration.
- **macOS (Intel)** -- RAM and CPU detection works. Discrete GPU detection if `nvidia-smi` available.
- **Windows** -- RAM and CPU detection works. NVIDIA GPU detection via `nvidia-smi` if installed.

### GPU support

| Vendor | Detection method | VRAM reporting |
|---|---|---|
| NVIDIA | `nvidia-smi` | Exact dedicated VRAM |
| AMD | `rocm-smi` | Detected (VRAM may be unknown) |
| Intel Arc (discrete) | sysfs (`mem_info_vram_total`) | Exact dedicated VRAM |
| Intel Arc (integrated) | `lspci` | Shared system memory |
| Apple Silicon | `system_profiler` | Unified memory (= system RAM) |

If autodetection fails or reports incorrect values, use `--memory=<SIZE>` to override (see [GPU memory override](#gpu-memory-override) above).

---

## Contributing

Contributions are welcome, especially new models.

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
