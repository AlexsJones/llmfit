# Development

Project structure, publishing, and dependencies.

[← Back to README](../README.md)

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
llmfit-core/data/
  hf_models.json  -- Model database (embedded at compile time)
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

- `llmfit-core/data/hf_models.json` is committed. It is embedded at compile time and must be present in the published crate.

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
