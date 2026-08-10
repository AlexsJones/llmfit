# AGENTS.md

Instructions for AI agents contributing to this codebase.

---

## Project overview

`llmfit` is a Rust CLI/TUI tool that matches LLM models against local system hardware (RAM, CPU, GPU). It detects system specs, loads a model database from embedded JSON, scores each model's fit, and presents results in an interactive terminal UI or classic table output.

## Language and toolchain

- Rust, edition 2024.
- Build with `cargo build`. Run with `cargo run`.
- No nightly features required. Stable toolchain only.
- Minimum supported Rust version: whatever edition 2024 requires (1.85+).

## Architecture

```
llmfit-core/      Shared Rust library. It owns hardware detection, model data,
                 fit analysis, planning, providers, benchmarks, quality checks,
                 model updates, diagnostics, claims, and result sharing.

llmfit-tui/       Main `llmfit` binary. It provides the CLI, ratatui TUI,
                 Axum HTTP API, embedded Web dashboard, and stdio MCP server.
                 `main.rs` parses all clap flags and selects an interface.

llmfit-desktop/   Tauri desktop application. Tauri commands call llmfit-core
                 for hardware detection, fit analysis, and Ollama downloads.

llmfit-web/       React 18 and Vite dashboard. It calls `/api/v1/*` endpoints
                 from llmfit-tui. The llmfit-tui build script embeds `dist/`.
                 This directory is not a Cargo workspace member.

llmfit-python/    Python package wrapper. Its wheel includes the compiled Rust
                 binary. `python -m llmfit` forwards arguments to that binary.
                 It does not expose llmfit-core through a native Python API.
```

The Cargo workspace contains `llmfit-core`, `llmfit-tui`, and
`llmfit-desktop`. The default members are `llmfit-core` and `llmfit-tui`.

Source modules in `llmfit-core/src/`:

- `analysis.rs`: Builds model-fit results. It marks installed models and applies
  local, community, and measured benchmark calibration.
- `bench.rs`: Runs throughput benchmarks against Ollama and OpenAI-compatible
  endpoints. It also discovers available benchmark targets.
- `benchmarks.rs`: Loads embedded and remote benchmark data. It builds measured
  throughput indexes and hardware leaderboard queries.
- `claim.rs`: Calculates model resource bounds. It renders Kubernetes DRA
  `ResourceClaim` and `ResourceClaimTemplate` manifests.
- `doctor.rs`: Collects installation, hardware, runtime, and model diagnostics.
- `fit.rs`: Calculates fit level, run mode, runtime, quantization, score, and
  estimated throughput.
- `hardware.rs`: Detects RAM, CPU, GPUs, unified memory, clusters, and memory
  bandwidth.
- `models.rs`: Defines model metadata. It loads embedded HF and ONNX catalogs,
  custom models, and the update cache.
- `plan.rs`: Estimates memory, throughput, run paths, and hardware upgrade needs
  for a requested model setup.
- `providers.rs`: Integrates Ollama, MLX, llama.cpp, Docker Model Runner,
  LM Studio, vLLM, and RamaLama.
- `quality.rs`: Runs response quality tests. It scores roles, builds routing
  recommendations, and compares results with baselines.
- `share.rs`: Stores local benchmark results. It handles GitHub authentication
  and submits community benchmark data.
- `task_bench.rs`: Provides task benchmark scores for model and task pairs.
- `update.rs`: Fetches model metadata and manages the local model update cache.

Source modules in `llmfit-tui/src/`:

- `main.rs`: Owns CLI parsing, hardware overrides, command execution, and
  interface dispatch.
- `display.rs`: Renders classic CLI tables, model plans, JSON, and CSV output.
- `download_history.rs`: Stores persistent model download records.
- `events.rs`: Publishes optional NATS system events and periodic snapshots.
- `filter_config.rs`: Loads and saves persistent TUI filter settings.
- `mcp_server.rs`: Exposes hardware, model, runtime, and planning MCP tools.
- `serve_api.rs`: Serves the embedded Web dashboard and JSON API with Axum.
- `serve_shared.rs`: Converts shared core types into API and MCP JSON values.
- `theme.rs`: Defines TUI color themes and stores the selected theme.
- `tui_app.rs`: Owns TUI state, model results, filters, downloads, and selection.
- `tui_events.rs`: Handles crossterm input and mutates TUI state.
- `tui_ui.rs`: Renders TUI views, tables, details, plans, and popups with
  ratatui.

## Data flow

All interfaces use the same core analysis flow:

1. `SystemSpecs::detect()` detects CPU, RAM, GPU, unified-memory, and cluster
   information. CLI hardware overrides can replace detected values.
2. `ModelDatabase::new()` loads the embedded HF and ONNX catalogs.
3. Custom models replace matching embedded models. The update cache appends
   models that are not already present.
4. `build_model_fits()` removes backend-incompatible models. It calls
   `ModelFit::analyze_with_forced_runtime()` for each remaining model.
5. Fit analysis selects a runtime, quantization, and run mode. It calculates
   memory use, throughput, fit level, score components, and notes.
6. Local benchmark results, community results, and measured presets can replace
   or calibrate formula-based throughput estimates.
7. Each interface applies its own filters, sorting, limits, and presentation.

`ModelFit::analyze()` is the default analysis wrapper. Use
`analyze_with_context_limit()` for a context cap. Use
`analyze_with_forced_runtime()` for runtime selection. Use
`analyze_with_config()` for custom calculation parameters. These methods share
the private `analyze_inner()` implementation.

Interface-specific flow:

- CLI: `main.rs` dispatches a subcommand. The command calls llmfit-core and
  writes a table, JSON, or CSV result.
- TUI: `App` owns model and filter state. `tui_events` changes that state.
  `apply_filters()` updates visible indices. `tui_ui` renders the current state.
- Web: React calls `/api/v1/*`. Axum handlers in `serve_api.rs` call llmfit-core
  and return JSON. The same server returns the embedded React assets.
- MCP: `LlmfitMcpServer` receives stdio tool calls. Each tool calls shared core
  analysis or planning logic and returns JSON text.
- Desktop: Tauri commands call llmfit-core and serialize results for the desktop
  UI. Ollama pull state stays in the Tauri application state.
- Python: The Python entry point locates the installed `llmfit` binary. It then
  replaces the process on Unix or starts a subprocess on Windows.

## Model database

- Source: `llmfit-core/data/hf_models.json` (33 models).
- Generated by `scripts/scrape_hf_models.py` (Python, stdlib only, no pip deps).
- Embedded at compile time via `include_str!("../data/hf_models.json")`.
- Schema per entry: name, provider, parameter_count, min_ram_gb, recommended_ram_gb, min_vram_gb, quantization, context_length, use_case.
- `min_vram_gb` is VRAM needed for GPU inference. `min_ram_gb` is system RAM needed for CPU inference. Both are derived from the same parameter count.
- RAM formula: `params * 0.5 bytes (Q4_K_M) / 1024^3 * 1.2 overhead`.
- VRAM formula: `params * 0.5 bytes (Q4_K_M) / 1024^3 * 1.1 activation overhead`.
- Recommended RAM: `model_size * 2.0`.

Do not manually edit `hf_models.json`. Regenerate it by running the scraper:

```sh
python3 scripts/scrape_hf_models.py
```

The scraper has hardcoded fallback entries for gated models that require authentication.

## Conventions

- No `unsafe` code.
- No `.unwrap()` on user-facing paths. Use proper error handling or `expect()` with a descriptive message for internal invariants only.
- Fit levels are ordered: Perfect > Good > Marginal > TooTight. Do not add levels without updating `rank_models_by_fit()` sort logic.
- Fit is VRAM-first. `RunMode` has five execution paths: `Gpu`, `MoeOffload`,
  `CpuOffload`, `CpuOnly`, and `TensorParallel`.
- `Gpu` keeps the model in VRAM. `MoeOffload` keeps active experts in VRAM and
  inactive experts in RAM. `CpuOffload` splits work between VRAM and RAM.
  `CpuOnly` uses system RAM. `TensorParallel` distributes work across nodes.
- `min_vram_gb` is the VRAM needed to load model weights on GPU. `min_ram_gb` is the system RAM needed for CPU-only inference (same weights, loaded into RAM instead). They represent the same workload on different hardware paths.
- On Apple Silicon (unified memory), VRAM = system RAM. The `CpuOffload` path is skipped because there is no separate RAM pool to spill to. `SystemSpecs::unified_memory` tracks this.
- TUI rendering is stateless. `tui_ui::draw()` must not mutate `App`. Pass `&mut App` only for `TableState` widget requirements -- do not use it to change application state.
- Event handling in `tui_events.rs` is the sole place that mutates `App` in the TUI loop.
- Keep `display.rs` and `tui_*.rs` independent. The CLI path must work without initializing any TUI state.

## Adding a new model to the database

1. Add the model's HuggingFace repo ID to `TARGET_MODELS` in `scripts/scrape_hf_models.py`.
2. If the model is gated (requires HF auth), add a fallback entry to the `FALLBACK` dict in the same script.
3. Run `python3 scripts/scrape_hf_models.py`.
4. Verify the output in `llmfit-core/data/hf_models.json`.
5. Run `cargo build` to verify compilation.

## Adding a new filter

1. Add the filter state to `App` in `tui_app.rs`.
2. Add filtering logic inside `apply_filters()`.
3. Add the keybinding in `tui_events.rs` (Normal mode handler).
4. Add the UI widget in `tui_ui.rs` (`draw_search_and_filters()` function).
5. Update the status bar help text in `draw_status_bar()`.

## Adding a new CLI subcommand

1. Add a variant to the `Commands` enum in `main.rs`.
2. Add the match arm in the `main()` function's command dispatch.
3. Use `display.rs` functions for output, or add new ones as needed.

## Testing

The project has Rust, Web, and Python test suites.

- Rust unit tests live beside code in `llmfit-core/src/` and
  `llmfit-tui/src/`.
- Core integration tests in `llmfit-core/tests/` validate catalog schemas and
  ONNX model data.
- CLI integration tests in `llmfit-tui/tests/` use `assert_cmd` against the
  compiled `llmfit` binary.
- HTTP API tests exercise Axum routers and JSON responses in `serve_api.rs`.
- TUI tests focus on state transitions, filters, event handling, and render
  output. Keep production rendering stateless.
- Web tests use Vitest, jsdom, and Testing Library. They cover API query
  construction, localization, filtering, and dashboard interactions.
- Python tests use pytest. They cover binary discovery, package versioning, and
  invocation of the packaged Rust binary.

Run the default Rust test set:

```sh
cargo test
```

Run all Rust workspace members, including the desktop crate:

```sh
cargo test --workspace
```

Run one Rust package:

```sh
cargo test -p llmfit-core
cargo test -p llmfit
```

Run the Web tests:

```sh
npm --prefix llmfit-web test
```

Run the Python tests and quality checks:

```sh
uv run --project llmfit-python pytest llmfit-python/tests
make -C llmfit-python check
```

## Dependencies policy

- Prefer crates that are well-maintained and have minimal transitive dependencies.
- `sysinfo` is the system detection crate. Do not replace it with raw platform calls.
- `ureq` is the blocking HTTP client for providers, benchmarks, updates, quality
  tests, and sharing. Do not add a second core HTTP client without a concrete need.
- `which` locates installed runtime binaries. Keep runtime discovery in
  `providers.rs` instead of adding manual `PATH` parsing.
- `regex` supports response scoring and text parsing. `serde_yml` parses quality
  test configuration. `base64` encodes benchmark submissions for GitHub.
- `objc2-metal` reads the effective Metal working-set limit on macOS. Keep it a
  macOS-only dependency. Do not replace it with raw platform calls.
- `ratatui` + `crossterm` is the TUI stack. Do not mix in `termion` or `ncurses`.
- `clap` with derive feature for CLI parsing. Do not use manual arg parsing.
- The Python scraper uses only stdlib (`urllib`, `json`). Do not add pip dependencies.

## Common tasks

```sh
# Build
cargo build

# Run TUI
cargo run

# Run CLI mode
cargo run -- --cli

# Run specific subcommand
cargo run -- system
cargo run -- fit --perfect -n 5
cargo run -- search "llama"

# Refresh model database
python3 scripts/scrape_hf_models.py && cargo build

# Check for compilation issues
cargo check

# Format code
cargo fmt

# Lint
cargo clippy
```

## Platform notes

- GPU detection shells out to `nvidia-smi` (NVIDIA) and `rocm-smi` (AMD). These are best-effort and fail silently if unavailable.
- Apple Silicon detection uses `system_profiler SPDisplaysDataType`. On unified memory Macs, VRAM is reported as available system RAM (same pool).
- `sysinfo` handles cross-platform RAM/CPU. No conditional compilation needed.
- The TUI uses crossterm which works on Linux, macOS, and Windows terminals.
