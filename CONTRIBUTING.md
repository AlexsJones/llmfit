# Contributing to llmfit

Thanks for your interest in contributing! Whether it's a bug fix, new feature, model addition, or documentation improvement, we appreciate the help.

## Getting started

### Prerequisites

- **Rust** (stable toolchain, edition 2024, MSRV 1.85+)
- **Python 3** (for model database scripts — stdlib only, no pip dependencies)
- **Git**

### Building from source

```sh
git clone https://github.com/AlexsJones/llmfit.git
cd llmfit
make build        # debug build
make release      # release build
```

### Running

```sh
make run                      # TUI mode (default)
cargo run -- --cli            # classic table output
cargo run -- system           # show detected hardware
cargo run -- fit --perfect    # show best-fit models
cargo run -- search "llama"   # search models
```

### Useful commands

```sh
make test       # run all tests
make fmt        # format code (cargo fmt)
make clippy     # run linter (cargo clippy)
make check      # fast compilation check
```

## Project structure

llmfit is a Rust workspace with three crates:

| Crate | Description |
|-------|-------------|
| `llmfit-core` | Core library — hardware detection, model database, fit analysis |
| `llmfit-tui` | Terminal user interface (ratatui + crossterm) |
| `llmfit-desktop` | Desktop app (Tauri) |

Supporting directories:

- `scripts/` — Python utilities for scraping HuggingFace and Docker model metadata
- `data/` — Generated JSON model databases (do not edit manually)
- `llmfit-python/` — Python bindings
- `llmfit-web/` — Web interface

For a deeper dive into the architecture, see [AGENTS.md](AGENTS.md).

## How to contribute

### Reporting bugs

Open an [issue](https://github.com/AlexsJones/llmfit/issues) with:

- What you expected to happen
- What actually happened
- Your OS, hardware (GPU model, RAM), and llmfit version (`llmfit --version`)
- Steps to reproduce

### Suggesting features

Start a [discussion](https://github.com/AlexsJones/llmfit/discussions) or open an issue. We're happy to chat about ideas before you invest time coding.

### Submitting a pull request

1. **Fork** the repo and create a branch from `main`.
2. Make your changes.
3. Run `cargo fmt` — most CI failures are from unformatted code.
4. Run `make clippy` and fix any warnings.
5. Run `make test` to verify nothing is broken.
6. Open a PR against `main` with a clear description of what and why.

Keep PRs focused. One bug fix or feature per PR is easier to review than a combined change.

### Adding a new model

1. Add the model's HuggingFace repo ID (e.g., `meta-llama/Llama-3.1-8B`) to the `TARGET_MODELS` list in `scripts/scrape_hf_models.py`.
2. If the model is gated (requires HF authentication), add a fallback entry to the `FALLBACKS` dict in the same script.
3. Run `make update-models` to regenerate the database and rebuild.
4. Verify with `./target/release/llmfit list`.
5. Update [MODELS.md](MODELS.md) if needed.
6. Open a PR.

## Code guidelines

- **No `unsafe` code.**
- No `.unwrap()` on user-facing paths. Use proper error handling or `expect()` with a descriptive message for internal invariants only.
- Keep TUI rendering stateless — `tui_ui::draw()` must not mutate application state.
- Prefer well-maintained crates with minimal transitive dependencies.
- The Python scraper uses only stdlib (`urllib`, `json`). Do not add pip dependencies.

## Code of conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior by opening an issue.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
