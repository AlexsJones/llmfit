# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Start here

[AGENTS.md](AGENTS.md) is the canonical deep-dive: per-module responsibilities in
`llmfit-core/src/` and `llmfit-tui/src/`, the data-flow through
`ModelFit::analyze*`, and step-by-step recipes for adding a filter, a CLI
subcommand, or a model. Read it before non-trivial changes; this file covers the
build/test surface and the cross-cutting invariants that span crates.

Two known-stale spots, so don't trust them over the source:

- AGENTS.md says `hf_models.json` holds 33 models — it currently holds ~10,800
  entries with a much richer schema (architecture, MoE dims, capabilities,
  license, release date).
- `docs/development.md` describes a flat `src/` layout that predates the
  workspace split.

## Commands

```sh
make build            # cargo build (debug)
make release          # cargo build --release
make test             # cargo test
make check            # cargo check
make fmt              # cargo fmt
make clippy           # cargo clippy -- -D warnings  (stricter than CI)
make run              # TUI

cargo run -- --cli    # classic table output
cargo run -- system   # detected hardware
cargo run -- doctor   # hardware diagnostic dump for bug reports
```

Testing:

```sh
cargo test                          # default members: llmfit-core + llmfit (tui)
cargo test --workspace              # adds llmfit-desktop
cargo test -p llmfit-core           # one package
cargo test -p llmfit                # the TUI/CLI crate is named `llmfit`
cargo test -p llmfit-core fit::tests::name_of_test   # single test by path filter
cargo test --test cli_smoke         # one integration test binary

npm --prefix llmfit-web test        # vitest run
uv run --project llmfit-python pytest llmfit-python/tests
python3 scripts/test_api.py --spawn # REST API contract assertions (spawns `llmfit serve`)
python3 scripts/validate_community_benchmarks.py   # what the community-benchmarks CI job runs
```

`llmfit-tui/build.rs` embeds `llmfit-web/dist/`. If that directory is missing the
build still succeeds but ships a **placeholder dashboard** (emitted as a
`cargo:warning`). Before touching `serve_api.rs` or the Web UI:

```sh
cd llmfit-web && npm ci && npm run build
```

CI runs `cargo test`, `cargo clippy --all-targets --all-features` (without
`-D warnings`), `cargo fmt --all -- --check`, `cargo check`, and vitest, on
Linux/macOS/Windows. `.githooks/pre-push` runs `cargo fmt --check` but is opt-in:
`git config core.hooksPath .githooks`.

Data regeneration (never hand-edit the generated JSON):

```sh
make update-models          # scripts/update_models.sh → HF scrape + rebuild
make update-docker-models   # Docker Model Runner catalog
make update-catalogs        # both, then a release build
```

## Architecture

Rust workspace: `llmfit-core` (library) + `llmfit-tui` (the `llmfit` binary) +
`llmfit-desktop` (Tauri, not a default member). `llmfit-web` (React/Vite) and
`llmfit-python` (wheel wrapping the compiled binary) sit outside the Cargo
workspace.

The point of the layout: **six interfaces, one analysis pipeline.** CLI, TUI,
Axum REST API, embedded Web dashboard, stdio MCP server (all in `llmfit-tui`),
and Tauri commands (`llmfit-desktop`) all funnel into the same
`SystemSpecs::detect()` → `ModelDatabase::new()` → `build_model_fits()` →
`ModelFit::analyze*()` chain. Behavior differences between interfaces should only
ever be filtering, sorting, limits, and presentation — if a fix belongs in
scoring, memory estimation, or run-mode selection, it goes in `llmfit-core` so
all six inherit it. `serve_shared.rs` exists to keep API and MCP JSON derived
from the same core types.

Everything the tool needs at runtime is **embedded at compile time**: the HF and
ONNX catalogs and the Docker catalog via `include_str!`, and the community
benchmark submissions via `build.rs` → `OUT_DIR/community_benchmarks.json`. There
is no network dependency for a fit analysis; network calls (`ureq`) are confined
to providers, updates, benchmarking, and sharing.

Throughput numbers have a precedence order that matters when a number looks
wrong: local measured benchmarks (this machine) → community submissions recorded
on identical hardware → measured presets → the formula estimate in `fit.rs`. The
`estimate_basis` field in JSON output says which path produced a given number.

`llmfit-core/data/community/<hardware-slug>/<timestamp>-<hash>.json` is populated
by users running `llmfit bench --share`; PRs touching it are validated repo-wide
(not just changed files) by the community-benchmarks workflow against
`data/community/schema.json`.

## Invariants

Beyond the conventions in AGENTS.md (no `unsafe`, no `.unwrap()` on user-facing
paths, stateless `tui_ui::draw()`, mutation only in `tui_events.rs`):

- Fit is **VRAM-first**. `RunMode` has exactly five paths — `Gpu`, `MoeOffload`,
  `CpuOffload`, `CpuOnly`, `TensorParallel` — and fit levels are ordered
  Perfect > Good > Marginal > TooTight; adding either requires updating
  `rank_models_by_fit()`.
- On unified memory (Apple Silicon), VRAM == system RAM and `CpuOffload` is
  skipped; guard on `SystemSpecs::unified_memory`, not on `cfg(target_os)`.
- The CLI path must not initialize TUI state — keep `display.rs` independent of
  `tui_*.rs`.
- JSON field names in `recommend`/`fit`/`plan` output and `/api/v1/*` are a
  documented contract (see `docs/cli.md` and `API.md`); renaming one is a
  breaking change for agents and the Web UI alike.

## Releases

Conventional commits drive release-please (`release-please-config.json`,
`.release-please-manifest.json`, `version.txt`). Do not bump versions by hand —
crate versions come from the workspace and are managed by the release workflow.
