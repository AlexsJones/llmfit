# llmfit — Project Planning & Improvement Roadmap

## Project Overview

llmfit is a Rust workspace CLI/TUI tool that right-sizes LLM models to local hardware. It detects system specs (RAM, CPU, GPU/VRAM), scores hundreds of models across quality/speed/fit/context dimensions, and presents results in an interactive TUI or classic CLI.

**Current version:** 0.9.3  
**Language:** Rust (edition 2024), React (web dashboard)  
**Architecture:** Cargo workspace with 3 crates + web frontend

---

## Workspace Structure

```
llmfit/
├── llmfit-core/        # Library: hardware detection, model DB, fit scoring, providers
│   └── src/
│       ├── hardware.rs     # SystemSpecs::detect(), GPU probing (nvidia-smi, rocm-smi, etc.)
│       ├── models.rs       # LlmModel, ModelDatabase, quantization helpers
│       ├── fit.rs          # FitLevel, RunMode, scoring engine, speed estimation
│       ├── providers.rs    # Ollama, llama.cpp, MLX, Docker Model Runner, LM Studio
│       ├── plan.rs         # Hardware planning / inverse fit estimation
│       └── update.rs       # Self-update logic
├── llmfit-tui/         # Binary: TUI + CLI + REST API server
│   └── src/
│       ├── main.rs         # CLI arg parsing (clap), entrypoint
│       ├── tui_app.rs      # App state, filters, navigation
│       ├── tui_ui.rs       # ratatui rendering (stateless)
│       ├── tui_events.rs   # crossterm keyboard event handling
│       ├── serve_api.rs    # axum REST API (embedded web assets)
│       ├── display.rs      # CLI table rendering (tabled + colored)
│       └── theme.rs        # 10 built-in color themes
├── llmfit-desktop/     # Tauri desktop app (wraps web UI)
├── llmfit-web/         # React web dashboard (Vite, served embedded in binary)
│   └── src/
│       ├── App.jsx
│       ├── components/     # Header, SystemPanel, FilterBar, ModelTable, etc.
│       ├── contexts/       # FilterContext, ModelContext
│       └── hooks/          # useModels, useSystem
├── data/
│   └── hf_models.json      # 200+ model entries (embedded at compile time)
└── scripts/
    ├── scrape_hf_models.py     # HF API scraper (stdlib only)
    ├── update_models.sh        # Automated DB update
    └── test_api.py             # REST API integration tests
```

---

## Data Flow

```
SystemSpecs::detect()
    → ModelDatabase::new()  (embedded JSON)
        → ModelFit::analyze() × N models
            → rank_models_by_fit()
                → apply_filters()  (TUI: filtered_fits: Vec<usize>)
                    → tui_ui::draw() / display::render_table() / serve_api
```

---

## Key Algorithms

### Fit Scoring (fit.rs)
- 4 dimensions: Quality (0-100), Speed (0-100), Fit (0-100), Context (0-100)
- Weighted composite score; weights vary by use-case (e.g. Chat: Speed 0.35, Reasoning: Quality 0.55)
- Dynamic quantization: walks Q8_0 → Q2_K, picks best that fits VRAM/RAM
- MoE: only active experts counted for VRAM (e.g. Mixtral 8x7B: 46.7B total → ~6.6GB VRAM)

### Speed Estimation (fit.rs)
- Bandwidth-based: `(bandwidth_GB_s / model_size_GB) × 0.55`
- ~80 GPU bandwidth entries; fallback constants per backend (CUDA=220, Metal=160, etc.)

### Hardware Detection (hardware.rs)
- RAM/CPU: sysinfo crate
- NVIDIA: nvidia-smi (multi-GPU, aggregates VRAM)
- AMD: rocm-smi
- Intel Arc: sysfs / lspci
- Apple Silicon: system_profiler (unified memory, VRAM = RAM)
- Ascend: npu-smi

---

## Improvement Roadmap

### P0 — Critical / High Impact

1. **Test coverage** — Zero tests currently. Add:
   - Unit tests for `fit.rs` (FitLevel assertions given known specs)
   - Unit tests for `models.rs` (JSON parsing, search)
   - Integration tests for CLI subcommands via `assert_cmd`
   - Property-based tests for scoring invariants (score always 0-100, TooTight always last)

2. **GPU detection reliability** — `nvidia-smi` / `rocm-smi` shell-out is fragile:
   - Add NVML bindings (nvml-wrapper crate) as primary path, fall back to nvidia-smi
   - Cache detection result to avoid repeated subprocess spawns on each filter change
   - Better error messages when detection fails (currently silent)

3. **Model database freshness** — 200+ models hardcoded in JSON, embedded at compile time:
   - Add optional runtime refresh from a hosted endpoint (with TTL cache)
   - Version the JSON schema so old binaries can detect incompatible updates
   - Separate `docker_models.json` is duplicated in `data/` and `llmfit-core/data/` — consolidate

### P1 — Important Improvements

4. **Provider reliability**
   - Ollama pull progress uses polling; switch to streaming `/api/pull` response parsing
   - LM Studio download-status polling has no timeout — can hang indefinitely
   - llama.cpp GGUF path heuristics are fragile; add user-configurable model cache path

5. **TUI performance**
   - `apply_filters()` recomputes all fits on every keystroke — add debounce or incremental filtering
   - `tui_ui.rs` is 113K — split into focused modules (table, popups, detail, compare)
   - `tui_app.rs` is 78K — extract filter state, download state, simulation state into sub-structs

6. **Web dashboard parity**
   - Web UI lacks: hardware simulation, plan mode, download trigger, compare view
   - No WebSocket support — polling only; add SSE or WS for live download progress
   - Web dashboard auto-starts on `0.0.0.0:8787` even when not needed — make opt-in or lazy

7. **Plan mode accuracy**
   - KV-cache estimation uses fixed formula; should account for GQA (grouped-query attention)
   - No support for speculative decoding memory overhead
   - Plan output doesn't account for OS/runtime memory overhead

### P2 — Nice to Have

8. **New providers**
   - vLLM: currently listed as `InferenceRuntime` but no provider implementation
   - Jan.ai, LocalAI, Koboldcpp support
   - Remote inference APIs (Groq, Together, Fireworks) for comparison baseline

9. **Model database expansion**
   - Add Llama 4 family (Scout, Maverick) — currently missing or incomplete
   - Add Gemma 3 family
   - Add Phi-4 variants
   - Structured capability tags (function-calling, vision, tool-use) per model

10. **UX improvements**
    - `d` download key: show estimated download size before confirming
    - Add `e` key to open model's HuggingFace page in browser
    - Persist filter state across sessions (save to `~/.config/llmfit/state.json`)
    - Add `?` inline help overlay per popup (not just global `h`)

11. **CLI improvements**
    - `llmfit compare <model1> <model2>` subcommand for non-TUI comparison
    - `llmfit export --format csv` for spreadsheet workflows
    - Shell completions (clap can generate these — just not wired up)

12. **Packaging**
    - Nix flake is present but not tested in CI
    - No ARM Linux binary in release workflow
    - Desktop app (Tauri) has no CI build or release pipeline

### P3 — Architecture / Long-term

13. **Plugin system** — Allow external model databases via a simple JSON schema URL
14. **Cluster mode** — `cluster_mode` field exists in `SystemSpecs` but is unused
15. **Benchmark integration** — Pull real tok/s from llama.cpp bench output instead of estimating
16. **Config file** — `~/.config/llmfit/config.toml` for persistent overrides (RAM, VRAM, preferred runtime)

---

## Quick Wins (can be done in < 1 day each)

- [ ] Wire up shell completions: `llmfit completions bash/zsh/fish`
- [ ] Consolidate duplicate `hf_models.json` (root `data/` vs `llmfit-core/data/`)
- [ ] Add `--version` flag output to include build date and git SHA
- [ ] Fix `OLLAMA_HOST` env var: currently ignores port-only values like `:11434`
- [ ] Add `Content-Security-Policy` header to embedded web dashboard responses
- [ ] `llmfit list --json` currently undocumented — add to README
- [ ] Desktop app `build.rs` is a stub (1 line) — either implement or remove

---

## Tech Debt

| Area | Issue |
|------|-------|
| `tui_ui.rs` | 113K single file — needs splitting |
| `providers.rs` | 112K single file — each provider should be its own module |
| `hardware.rs` | 105K — GPU detection per-vendor should be sub-modules |
| Duplicate data | `hf_models.json` exists in both `data/` and `llmfit-core/data/` |
| No tests | Zero test coverage across all crates |
| `update.rs` | 23K self-update logic in core library — should be in TUI crate |
| Web assets | Embedded via build.rs codegen — no hot-reload in dev mode |

---

## CI/CD Status

| Workflow | Status |
|----------|--------|
| `ci.yml` | Builds + clippy + fmt check |
| `release.yml` | GitHub releases with binaries |
| `docker.yml` | Docker image to ghcr.io |
| `release-desktop.yml` | Tauri desktop (incomplete) |
| `release-please.yml` | Automated changelog/version bump |

Missing: test step in CI, ARM Linux build, Nix flake check, web dashboard build verification.
