# TUI Guide

Everything about llmfit's interactive terminal UI: navigation, planning, simulation, configuration, downloads, the community leaderboard, benchmarking, themes, and the web dashboard.

[← Back to README](../README.md)

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
| `P`                        | Open provider filter popup (type to fuzzy-filter providers)          |
| `U`                        | Open use-case filter popup                                            |
| `C`                        | Open capability filter popup                                          |
| `L`                        | Open license filter popup                                             |
| `R`                        | Open runtime/backend filter popup (llama.cpp, MLX, vLLM)             |
| `S`                        | Open hardware simulation popup (override RAM/VRAM/CPU)                |
| `A`                        | Open advanced configuration popup (tune efficiency, run mode factors) |
| `b`                        | Open community leaderboard view (localmaxxing.com)                    |
| `I`                        | Open inference bench view (local quality scoring against your models) |
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

![Hardware Simulation](../assets/simulation.png)

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

### Community Leaderboard (`b`)

Press `b` to open the Community Leaderboard view. Instead of relying solely on llmfit's theoretical speed estimates, this view shows **real-world performance data** from other users with the same hardware — actual measured tok/s, time-to-first-token, and peak VRAM usage.

![Community Leaderboard](../assets/benchmark.jpeg)

Data is sourced from [localmaxxing.com](https://localmaxxing.com), a community benchmark database. When you open the view, llmfit auto-detects your hardware (GPU model, VRAM tier, Apple Silicon chip family, OS) and queries for matching results.

| Column       | Description                                              |
|--------------|----------------------------------------------------------|
| **Model**    | HuggingFace model ID                                     |
| **Engine**   | Inference runtime used (llama.cpp, vLLM, Ollama, MLX...) |
| **Quant**    | Quantization format (Q4_K_M, Q8_0, etc.)                |
| **tok/s**    | Measured output token generation speed                   |
| **Total t/s**| Total throughput (prompt + generation)                   |
| **TTFT**     | Time to first token (latency)                            |
| **VRAM**     | Peak memory usage during inference                       |
| **Ctx**      | Context length used in the benchmark                     |
| **User**     | Submitter (verified users marked with `*`)               |

| Key                    | Action                                  |
|------------------------|-----------------------------------------|
| `j` / `k` or arrows   | Navigate results                        |
| `H`                    | Open hardware picker (browse any GPU)   |
| `r`                    | Refresh / re-fetch from API             |
| `b` / `q` / `Esc`     | Close and return to model table         |

Press `H` to open the hardware picker — a scrollable list of 27 popular GPUs and chips (RTX 5090 through CPU-only, plus Apple Silicon M1–M4 variants, AMD RX/MI series, and NVIDIA datacenter cards). Select one to instantly load benchmarks for that hardware, even if it's not what you're running on. Select "My Hardware (auto-detect)" to go back to your own system.

#### API key setup

Public benchmarks work without authentication. For full access, provide your [localmaxxing.com](https://localmaxxing.com) API key:

```sh
# Via environment variable (recommended)
export LOCALMAXXING_API_KEY="bhk_your_key_here"
llmfit

# Or via CLI flag
llmfit --api-key "bhk_your_key_here"
```

| Variable | Description |
|---|---|
| `LOCALMAXXING_API_KEY` | Bearer token for localmaxxing.com API |

### Inference Bench (`I`)

Press `I` (uppercase) to open the Inference Bench view. This runs **live inference benchmarks against your locally running providers** — Ollama, vLLM, and MLX — measuring time-to-first-token (TTFT), tokens per second (TPS), and total latency with real inference requests.

Unlike the Community Leaderboard (which shows crowd-sourced data from other users), Inference Bench measures your actual hardware with your actual models.

#### TUI usage

| Key | Action |
|-----|--------|
| `I` | Open inference bench (auto-detects provider and runs benchmarks) |
| `I` (again) | Rerun benchmarks from within the bench view |
| `j` / `k` or arrows | Navigate model results |
| `Enter` | Open detail view for selected model |
| `r` | Switch to routing matrix view |
| `q` / `Esc` | Close bench view |

Results are cached to `~/.config/llmfit/bench-cache.json` and loaded instantly on subsequent opens.

#### CLI usage

```sh
# Auto-detect provider and benchmark
llmfit bench

# Benchmark all discovered models across all running providers
llmfit bench --all

# Benchmark a specific model via Ollama
llmfit bench --provider ollama llama3.2

# Override endpoint URL
llmfit bench --provider ollama --url http://my-server:11434 llama3.2

# Override vLLM endpoint
llmfit bench --provider vllm --url http://localhost:8000

# Output as JSON (for scripting)
llmfit bench --json

# Run quality benchmarks (role-based scoring for routing)
llmfit bench --quality

# Output routing matrix
llmfit bench --quality --routing
```

#### Environment variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API base URL |
| `VLLM_PORT` | `8000` | vLLM server port (used as `http://localhost:$VLLM_PORT`) |

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
