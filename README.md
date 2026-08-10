# llmfit

<p align="center">
  <img src="assets/icon.svg" alt="llmfit icon" width="128" height="128">
</p>

<p align="center">
  <b>English</b> ·
  <a href="README.zh.md">中文</a> ·
  <a href="README.ja.md">日本語</a>
</p>

<p align="center">
  <a href="https://github.com/AlexsJones/llmfit/actions/workflows/ci.yml"><img src="https://github.com/AlexsJones/llmfit/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://crates.io/crates/llmfit"><img src="https://img.shields.io/crates/v/llmfit.svg" alt="Crates.io"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="https://about.signpath.io"><img src="https://img.shields.io/badge/SignPath-signed-brightgreen?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiIgZmlsbD0id2hpdGUiIHZpZXdCb3g9IjAgMCAxNiAxNiI+PHBhdGggZD0iTTEwLjA2NyA0LjU2N2wtNC43MzQgNC43MzMtMS40LTEuNGExIDEgMCAwIDAtMS40MTQgMS40MTRsMi4xIDIuMWExIDEgMCAwIDAgMS40MTQgMGw1LjQ0LTUuNDRhMSAxIDAgMCAwLTEuNDE0LTEuNDE0eiIvPjwvc3ZnPg==" alt="Signed with SignPath"></a>
</p>

> **📊 New: benchmark & share — real numbers from your machine, better estimates for everyone.** Download a model, serve it, and measure real tok/s on your hardware — then contribute the results back to the project as a PR, straight from the TUI. No `gh` CLI, no third-party account. Every run is saved locally first, your own measurements replace estimates in the fit table, and each merged submission ships in the next release: anyone on identical hardware gets measured `✓` numbers before they ever run a benchmark. [Follow the step-by-step benchmarking guide →](docs/benchmarking.md)
>
> *Previously: [llmfit 1.0 — the release where the numbers became verifiable →](https://github.com/AlexsJones/llmfit/discussions/708)*

**Hundreds of models & providers. One command to find what runs on your hardware.**

A terminal tool that right-sizes LLM models to your system's RAM, CPU, and GPU. Detects your hardware, scores each model across quality, speed, fit, and context dimensions, and tells you which ones will actually run well on your machine.

Ships with an interactive TUI (default), a classic CLI mode, and a C-ABI shared library for native integration (C, C++, Python, Pascal, Go). Supports multi-GPU setups, MoE architectures, dynamic quantization selection, speed estimation, and local runtime providers (Ollama, llama.cpp, MLX, Docker Model Runner, LM Studio).

> **Sister projects:**
> - [sympozium](https://github.com/sympozium-ai/sympozium/) — managing agents in Kubernetes.
> - [llmserve](https://github.com/AlexsJones/llmserve) — a simple TUI for serving local LLM models. Pick a model, pick a backend, serve it.
> - [llama-panel](https://github.com/AlexsJones/llama-panel) — a native macOS app for managing local llama-server instances.

![demo](assets/demo.gif)

## Documentation

|  |  |
|---|---|
| **Get started** | [Install](#install) · [Usage](#usage) · [How it works](#how-it-works) |
| **Guides** | [TUI guide](docs/tui.md) · [Benchmarking step-by-step](docs/benchmarking.md) · [CLI & automation](docs/cli.md) · [Runtime providers](docs/providers.md) · [OpenClaw integration](docs/openclaw.md) |
| **Reference** | [How it works (full)](docs/how-it-works.md) · [Platform & GPU support](docs/platform-support.md) · [Custom models](docs/custom-models.md) · [Development](docs/development.md) |
| **Project** | [Contributing](#contributing) · [Alternatives](#alternatives) · [Code signing](#code-signing) · [License](#license) |

---

## Install

### Windows
```sh
scoop install llmfit
```

If Scoop is not installed, follow the [Scoop installation guide](https://scoop.sh/).

### macOS / Linux

#### Homebrew

Prebuilt binary (recommended, works on all macOS/Linux versions):
```sh
brew install AlexsJones/llmfit/llmfit
```

Or from the homebrew-core formula, which builds from source on macOS versions without a bottle:
```sh
brew install llmfit
```

#### MacPorts
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

### uv / pip
To install or update llmfit:
```sh
uv tool install -U llmfit
```

To run without installing:
```sh
uvx llmfit
```

You can also install llmfit as a Python package in the normal way with tools such as pip or uv.

### Docker / Podman
```sh
docker run ghcr.io/alexsjones/llmfit
```
This prints JSON from `llmfit recommend` command. The JSON could be further queried with `jq`.
```
podman run ghcr.io/alexsjones/llmfit recommend --use-case coding | jq '.models[].name'
```
To launch the interactive TUI instead, pass the global `--tui` flag:
```sh
docker run --rm -it ghcr.io/alexsjones/llmfit --tui
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

```sh
llmfit          # interactive TUI: your hardware, every model, ranked
```

The TUI shows your detected specs at the top and every model scored for fit, speed, quality, and context. See the [TUI guide](docs/tui.md) for navigation, planning, simulation, downloads, the community leaderboard, and benchmarking.

For scripts, agents, and classic terminal output:

```sh
llmfit fit                    # table of all models ranked by fit
llmfit recommend --json       # top picks as JSON (agent/script consumption)
llmfit info "<model>"         # one model: fit analysis, estimate basis, verify commands
llmfit bench                  # measure real tok/s/TTFT against your running provider
llmfit doctor                 # hardware detection report for bug reports
```

Full reference: [CLI & automation](docs/cli.md).

---

## How it works

llmfit detects your hardware (RAM, CPU, GPU/VRAM, backend), then scores every model in its catalog across four dimensions: memory fit, estimated speed, quality, and context. Speed estimates come from a memory-bandwidth model grounded in runtime sampling and real community measurements — and every estimate ships its inputs, so `llmfit info` shows exactly what a number assumes and how to verify it on your machine.

Full detail, including the estimation formulas and the model database: [How llmfit works](docs/how-it-works.md).

---

## Contributing

Contributions are welcome, especially new models.

### Before submitting a PR

Please run `cargo fmt` before pushing your changes. Most CI check failures are caused by unformatted code:

```sh
cargo fmt
```

Guides for adding models — locally (no rebuild) or to the built-in catalog: [Custom models](docs/custom-models.md).

---

## Alternatives

If you're looking for a different approach, check out [llm-checker](https://github.com/Pavelevich/llm-checker) -- a Node.js CLI tool with Ollama integration that can pull and benchmark models directly. It takes a more hands-on approach by actually running models on your hardware via Ollama, rather than estimating from specs. Good if you already have Ollama installed and want to test real-world performance. Note that it doesn't support MoE (Mixture-of-Experts) architectures -- all models are treated as dense, so memory estimates for models like Mixtral or DeepSeek-V3 will reflect total parameter count rather than the smaller active subset.

---

## Code signing

llmfit's Windows release binaries are digitally signed (Authenticode) via [SignPath.io](https://about.signpath.io/), with a free code signing certificate provided by the [SignPath Foundation](https://signpath.org/).

Signing happens automatically in the [release pipeline](.github/workflows/release.yml): only artifacts built by GitHub Actions from this repository are submitted for signing, and signing requests are approved by the project maintainer ([@AlexsJones](https://github.com/AlexsJones)).

**Code signing policy:** see the [SignPath Foundation code signing policy and terms](https://signpath.org/terms).

**Privacy:** this program will not transfer any information to other networked systems unless specifically requested by the user or the person installing or operating it. llmfit only contacts external services when you explicitly use the corresponding feature (e.g. model downloads, runtime provider queries, or the community leaderboard).

---

## License

MIT
