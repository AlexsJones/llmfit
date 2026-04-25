mod display;
mod download_history;
mod filter_config;
mod serve_api;
mod theme;
mod tui_app;
mod tui_events;
mod tui_ui;

use clap::{Parser, Subcommand};
use std::net::{TcpStream, ToSocketAddrs};
use std::process::Stdio;
use std::thread;
use std::time::Duration;

use llmfit_core::fit::{ModelFit, SortColumn, backend_compatible};
use llmfit_core::hardware::SystemSpecs;
use llmfit_core::models::ModelDatabase;
use llmfit_core::plan::{PlanRequest, estimate_model_plan, resolve_model_selector};

fn parse_positive_usize(value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("invalid positive integer: {value}"))?;
    if parsed == 0 {
        return Err("value must be at least 1".to_string());
    }
    Ok(parsed)
}

const DEFAULT_DASHBOARD_HOST: &str = "127.0.0.1";
const DEFAULT_DASHBOARD_PORT: u16 = 8787;

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
enum SortArg {
    /// Composite ranking score (default)
    Score,
    /// Estimated tokens/second
    #[value(alias = "tokens", alias = "toks", alias = "throughput")]
    Tps,
    /// Model parameter count
    Params,
    /// Memory utilization percentage
    #[value(alias = "memory", alias = "mem_pct", alias = "utilization")]
    Mem,
    /// Context window length
    #[value(alias = "context")]
    Ctx,
    /// Release date (newest first)
    #[value(alias = "release", alias = "released")]
    Date,
    /// Use-case grouping
    #[value(alias = "use_case", alias = "usecase")]
    Use,
    /// Model provider
    #[value(alias = "prov", alias = "vendor")]
    Provider,
}

impl From<SortArg> for SortColumn {
    fn from(value: SortArg) -> Self {
        match value {
            SortArg::Score => SortColumn::Score,
            SortArg::Tps => SortColumn::Tps,
            SortArg::Params => SortColumn::Params,
            SortArg::Mem => SortColumn::MemPct,
            SortArg::Ctx => SortColumn::Ctx,
            SortArg::Date => SortColumn::ReleaseDate,
            SortArg::Use => SortColumn::UseCase,
            SortArg::Provider => SortColumn::Provider,
        }
    }
}

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
enum FitArg {
    All,
    Perfect,
    Good,
    Marginal,
    Tight,
    Runnable,
}

#[derive(Parser)]
#[command(name = "llmfit")]
#[command(about = "Right-size LLM models to your system's hardware")]
#[command(long_about = "\
Right-size LLM models to your system's hardware.

llmfit detects your system's RAM, CPU, and GPU (NVIDIA, AMD, Apple Silicon),
then scores every model in its database for fit, speed, and quality. It can
recommend models, compare them side-by-side, plan hardware upgrades, download
GGUF weights, and launch inference — all from a single binary.

GLOBAL FLAGS:
  --json             Output structured JSON on every subcommand (for tool/agent
                     integration). Always exits 0 on success, 1 on error.
  --memory <SIZE>    Override GPU VRAM (e.g. \"32G\", \"32000M\", \"1.5T\").
  --ram <SIZE>       Override system RAM (e.g. \"64G\", \"128000M\").
  --cpu-cores <N>    Override detected CPU core count.
  --max-context N    Cap context length for memory estimation (tokens).
                     Falls back to OLLAMA_CONTEXT_LENGTH env var if unset.

EXIT CODES:
  0  Success
  1  Any error (hardware detection failure, model not found, network error, etc.)

ENVIRONMENT VARIABLES:
  OLLAMA_CONTEXT_LENGTH  Default context-length cap when --max-context is not set.")]
#[command(after_long_help = "For a compact summary, use -h instead of --help.")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Show only models that perfectly match recommended specs
    #[arg(short, long)]
    perfect: bool,

    /// Show only models with tool/function-call capability
    #[arg(long)]
    tool_use: bool,

    /// Limit number of results
    #[arg(short = 'n', long)]
    limit: Option<usize>,

    /// Sort column for CLI fit output
    #[arg(long, value_enum, default_value_t = SortArg::Score)]
    sort: SortArg,

    /// Use classic CLI table output instead of TUI
    #[arg(long)]
    cli: bool,

    /// Output results as JSON (for tool integration)
    #[arg(long, global = true)]
    json: bool,

    /// Output results as CSV (for spreadsheet / data analysis)
    #[arg(long, global = true)]
    csv: bool,

    /// Override GPU VRAM size (e.g. "32G", "32000M", "1.5T").
    /// Useful when GPU memory autodetection fails.
    #[arg(long, value_name = "SIZE")]
    memory: Option<String>,

    /// Override system RAM (e.g. "64G", "128000M", "1T").
    /// Useful for evaluating model fit against target hardware.
    #[arg(long, value_name = "SIZE")]
    ram: Option<String>,

    /// Override detected CPU core count.
    /// Useful for evaluating model fit against target hardware.
    #[arg(long, value_name = "CORES", value_parser = parse_positive_usize)]
    cpu_cores: Option<usize>,

    /// Cap context length used for memory estimation (tokens).
    /// Falls back to OLLAMA_CONTEXT_LENGTH if not set.
    #[arg(long, value_name = "TOKENS", value_parser = clap::value_parser!(u32).range(1..))]
    max_context: Option<u32>,

    /// Do not auto-start the background dashboard server
    #[arg(long, global = true)]
    no_dashboard: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Show system hardware specifications
    #[command(long_about = "\
Show system hardware specifications.

Detects RAM, CPU, and GPU (NVIDIA via nvidia-smi, AMD via rocm-smi/sysfs,
Apple Silicon via system_profiler). On unified-memory systems (Apple Silicon),
VRAM is reported as system RAM.

PRECONDITIONS:
  None. GPU detection is best-effort and fails silently if tools are missing.

SIDE EFFECTS:
  None — read-only.

EXIT CODES:
  0  Success

AGENT USAGE:
  llmfit system --json

  JSON output fields: { system: { cpu, ram_gb, gpu_name, gpu_vram_gb,
  gpu_backend, unified_memory, os } }")]
    System,

    /// List all available LLM models
    #[command(long_about = "\
List all available LLM models.

Prints every model in the embedded database with name, provider, parameter
count, quantization, and context length. No hardware analysis is performed.

PRECONDITIONS:
  None.

SIDE EFFECTS:
  None — read-only.

EXIT CODES:
  0  Success

AGENT USAGE:
  llmfit list --json

  JSON output: array of model objects with fields: name, provider,
  parameter_count, min_ram_gb, recommended_ram_gb, min_vram_gb,
  quantization, context_length, use_case, capabilities.")]
    List {
        /// Sort models by column: date, params, ctx, mem
        #[arg(long, value_enum, default_value_t = SortArg::Date)]
        sort: SortArg,
    },

    /// Find models that fit your system (classic table output)
    #[command(long_about = "\
Find models that fit your system (classic table output).

Detects hardware, scores every model for fit/speed/quality, and prints a
ranked table. Models incompatible with the detected backend are hidden.

PRECONDITIONS:
  Requires hardware detection (GPU via nvidia-smi/rocm-smi/system_profiler).
  Use --memory to override GPU VRAM if autodetection fails.

SIDE EFFECTS:
  None — read-only.

EXIT CODES:
  0  Success
  1  Hardware detection or internal error

AGENT USAGE:
  llmfit fit --json
  llmfit fit --json --perfect -n 5
  llmfit fit --json --sort tps

  JSON output fields: { system: {...}, models: [{ name, provider,
  parameter_count, fit_level, run_mode, score, score_components,
  estimated_tps, disk_size_gb, memory_required_gb,
  memory_available_gb, utilization_pct, best_quant, use_case,
  runtime }] }")]
    Fit {
        /// Show only models that perfectly match recommended specs
        #[arg(short, long)]
        perfect: bool,

        /// Show only models with tool/function-call capability
        #[arg(long)]
        tool_use: bool,

        /// Limit number of results
        #[arg(short = 'n', long)]
        limit: Option<usize>,

        /// Sort column for fit output
        #[arg(long, value_enum, default_value_t = SortArg::Score)]
        sort: SortArg,
    },

    /// Search for specific models
    #[command(long_about = "\
Search for specific models.

Searches the embedded model database by name, provider, or parameter size.
No hardware analysis is performed.

PRECONDITIONS:
  None.

SIDE EFFECTS:
  None — read-only.

EXIT CODES:
  0  Success (even if no matches found)

AGENT USAGE:
  No --json support for this command. Use 'llmfit list --json' and filter
  client-side, or use 'llmfit info <model> --json' for a specific model.")]
    Search {
        /// Search query (model name, provider, or size)
        query: String,
    },

    /// Show detailed information about a specific model
    #[command(long_about = "\
Show detailed information about a specific model.

Looks up a model by name (or partial name) and displays full specs plus a
hardware fit analysis against the current system.

PRECONDITIONS:
  None. Hardware detection runs automatically for fit analysis.

SIDE EFFECTS:
  None — read-only.

EXIT CODES:
  0  Success
  1  No model found, or ambiguous partial match

AGENT USAGE:
  llmfit info \"llama-3.1-8b\" --json

  JSON output fields: { system: {...}, models: [{ <single model with full
  fit analysis: name, fit_level, run_mode, score, score_components,
  estimated_tps, memory_required_gb, utilization_pct, ... > }] }")]
    Info {
        /// Model name or partial name to look up
        model: String,
    },

    /// Compare two models side-by-side, or auto-compare top N filtered models
    #[command(long_about = "\
Compare two models side-by-side, or auto-compare top N filtered models.

When two model selectors are given, compares those two models. When none are
given, picks the top N models (default 2) after applying fit-level and sort
filters, and compares them.

PRECONDITIONS:
  Requires hardware detection for fit analysis. At least 2 models must pass
  the filter for auto-compare mode.

SIDE EFFECTS:
  None — read-only.

EXIT CODES:
  0  Success
  1  Model not found, ambiguous selector, fewer than 2 candidates, or
     both selectors resolve to the same model

AGENT USAGE:
  llmfit diff --json
  llmfit diff \"llama-8b\" \"qwen-7b\" --json
  llmfit diff --json --fit good --sort tps -n 3

  JSON output fields: { system: {...}, models: [{ name, fit_level,
  run_mode, score, estimated_tps, memory_required_gb, ... }] }")]
    Diff {
        /// First model selector (name or unique partial name)
        model_a: Option<String>,

        /// Second model selector (name or unique partial name)
        model_b: Option<String>,

        /// Sort column before selecting candidates
        #[arg(long, value_enum, default_value_t = SortArg::Score)]
        sort: SortArg,

        /// Fit-level filter before candidate selection
        #[arg(long, value_enum, default_value_t = FitArg::Runnable)]
        fit: FitArg,

        /// Number of top models to include when model names are omitted
        #[arg(short = 'n', long, default_value_t = 2)]
        limit: usize,
    },

    /// Plan hardware requirements for a specific model configuration
    #[command(long_about = "\
Plan hardware requirements for a specific model configuration.

Estimates VRAM/RAM requirements, expected throughput, and recommended hardware
for running a model at a given context length and quantization. Useful for
capacity planning and hardware purchasing decisions.

PRECONDITIONS:
  Model must exist in the embedded database (use 'llmfit search' to verify).

SIDE EFFECTS:
  None — read-only.

EXIT CODES:
  0  Success
  1  Model not found or invalid configuration

AGENT USAGE:
  llmfit plan \"llama-3.1-70b\" --context 8192 --json
  llmfit plan \"qwen-72b\" --context 4096 --quant Q4_K_M --target-tps 15 --json

  JSON output: PlanEstimate object with fields: model_name, context_length,
  quantization, weight_gb, kv_cache_gb, total_vram_gb, fits_in_vram,
  estimated_tps, recommended_gpu, notes.")]
    Plan {
        /// Model selector (name or unique partial name)
        model: String,

        /// Context length for estimation (tokens)
        #[arg(long, value_name = "TOKENS", value_parser = clap::value_parser!(u32).range(1..))]
        context: u32,

        /// Quantization override (e.g. Q4_K_M, Q8_0, mlx-4bit)
        #[arg(long)]
        quant: Option<String>,

        /// KV cache element representation (fp16, fp8, q8_0, q4_0, tq).
        /// Defaults to fp16. `tq` is TurboQuant, vLLM + CUDA only and
        /// experimental (not in upstream vLLM yet).
        #[arg(long, value_name = "KV")]
        kv_quant: Option<String>,

        /// Target decode speed in tokens/sec
        #[arg(long, value_name = "TOK_S")]
        target_tps: Option<f64>,
    },

    /// Recommend top models for your hardware (JSON-friendly)
    #[command(long_about = "\
Recommend top models for your hardware (JSON-friendly).

Analyzes all models against detected hardware and returns the top N ranked
recommendations. Supports filtering by use case, fit level, inference runtime,
model capabilities, and license. JSON output is enabled by default.

PRECONDITIONS:
  Requires hardware detection. Use --memory to override GPU VRAM if needed.

SIDE EFFECTS:
  None — read-only.

EXIT CODES:
  0  Success
  1  Hardware detection or internal error

AGENT USAGE:
  llmfit recommend
  llmfit recommend -n 3 --use-case coding --min-fit good
  llmfit recommend --runtime mlx --capability vision
  llmfit recommend --force-runtime llamacpp  # get llama.cpp results on Apple Silicon
  llmfit recommend --license apache-2.0,mit

  JSON output is the default. Fields: { system: {...}, models: [{ name,
  provider, parameter_count, fit_level, run_mode, score, score_components
  { quality, speed, fit, context }, estimated_tps, disk_size_gb,
  memory_required_gb, memory_available_gb, utilization_pct, best_quant,
  use_case, license, runtime, capabilities }] }")]
    Recommend {
        /// Limit number of recommendations
        #[arg(short = 'n', long, default_value = "5")]
        limit: usize,

        /// Filter by use case: general, coding, reasoning, chat, multimodal, embedding
        #[arg(long, value_name = "CATEGORY")]
        use_case: Option<String>,

        /// Filter by minimum fit level: perfect, good, marginal
        #[arg(long, default_value = "marginal")]
        min_fit: String,

        /// Filter by inference runtime: mlx, llamacpp, any
        #[arg(long, default_value = "any")]
        runtime: String,

        /// Force a specific runtime override, bypassing automatic selection
        /// (e.g. get llama.cpp recommendations on Apple Silicon instead of MLX)
        #[arg(long, value_name = "RUNTIME")]
        force_runtime: Option<String>,

        /// Filter by capability: vision, tool_use (comma-separated for multiple)
        #[arg(long, value_name = "CAPS")]
        capability: Option<String>,

        /// Filter by license (comma-separated, e.g. "apache-2.0,mit")
        #[arg(long, value_name = "LICENSE")]
        license: Option<String>,

        /// Output as JSON (default for recommend)
        #[arg(long, default_value = "true")]
        json: bool,
    },

    /// Download a GGUF model from HuggingFace for use with llama.cpp
    #[command(long_about = "\
Download a GGUF model from HuggingFace for use with llama.cpp.

Accepts a HuggingFace repo ID, a search query, or a known model name.
Automatically selects the best quantization that fits your hardware unless
--quant is specified. Use --list to browse available files without downloading.

PRECONDITIONS:
  Network access to huggingface.co. Hardware detection runs for auto quant
  selection (override with --budget or --quant).

SIDE EFFECTS:
  Downloads a GGUF file to the local model cache directory
  (~/.cache/llmfit/models/ or platform equivalent).

EXIT CODES:
  0  Success
  1  Model/repo not found, no GGUF files available, network error, or
     download failure

AGENT USAGE:
  No --json support. Parse stdout for progress and completion messages.
  Use --list to enumerate available quantizations before downloading.")]
    Download {
        /// Model to download. Can be:
        ///   - HuggingFace repo (e.g. "bartowski/Llama-3.1-8B-Instruct-GGUF")
        ///   - Search query (e.g. "llama 8b")
        ///   - Known model name (e.g. "llama-3.1-8b-instruct")
        model: String,

        /// Specific GGUF quantization to download (e.g. "Q4_K_M", "Q8_0").
        /// If omitted, selects the best quantization that fits your hardware.
        #[arg(short, long)]
        quant: Option<String>,

        /// Maximum memory budget in GB for quantization selection
        #[arg(long, value_name = "GB")]
        budget: Option<f64>,

        /// List available GGUF files in the repo without downloading
        #[arg(long)]
        list: bool,
    },

    /// Search HuggingFace for GGUF models compatible with llama.cpp
    #[command(long_about = "\
Search HuggingFace for GGUF models compatible with llama.cpp.

Queries the HuggingFace Hub API for repositories containing GGUF model files.
Results include the repository ID and model type.

PRECONDITIONS:
  Network access to huggingface.co.

SIDE EFFECTS:
  None — read-only (network query only).

EXIT CODES:
  0  Success (even if no results found)

AGENT USAGE:
  No --json support. Parse the tabular stdout output, or use the llmfit
  REST API ('llmfit serve') for programmatic access.")]
    HfSearch {
        /// Search query (model name, architecture, etc.)
        query: String,

        /// Maximum number of results
        #[arg(short = 'n', long, default_value = "10")]
        limit: usize,
    },

    /// Fetch the latest LLM models from HuggingFace and update the local cache.
    ///
    /// Models are saved to ~/.llmfit/hf_models_cache.json and automatically
    /// included the next time you run llmfit (no rebuild required).
    Update {
        /// Number of trending models to fetch
        #[arg(long, default_value = "100")]
        trending: usize,

        /// Number of top-downloaded models to fetch (0 to skip)
        #[arg(long, default_value = "50")]
        downloads: usize,

        /// HuggingFace API token for higher rate limits.
        /// Can also be supplied via the HF_TOKEN environment variable.
        #[arg(long)]
        token: Option<String>,

        /// Show cache status without fetching any new models
        #[arg(long)]
        status: bool,

        /// Delete all cached models (resets to the embedded list)
        #[arg(long)]
        clear: bool,
    },

    /// Run a downloaded GGUF model with llama-cli or llama-server
    #[command(long_about = "\
Run a downloaded GGUF model with llama-cli or llama-server.

Launches an interactive chat session (default) or an OpenAI-compatible API
server (--server). The model can be specified as a file path or a name to
search in the local cache.

PRECONDITIONS:
  llama-cli (or llama-server with --server) must be installed and in PATH.
  A GGUF model file must exist locally (use 'llmfit download' first).

SIDE EFFECTS:
  Launches an external llama.cpp process. In server mode, binds to the
  specified port.

EXIT CODES:
  0  Clean exit from llama.cpp
  1  llama-cli/llama-server not found, model not found, or process error
  *  Other codes are proxied from the llama.cpp process

AGENT USAGE:
  No --json support. For API server mode, use:
    llmfit run <model> --server --port 8080
  Then interact via the OpenAI-compatible API at http://localhost:8080.")]
    Run {
        /// Model file or name to run. If a name is given, searches the local cache.
        model: String,

        /// Run as an OpenAI-compatible API server instead of interactive chat
        #[arg(long)]
        server: bool,

        /// Port for the API server (default: 8080)
        #[arg(long, default_value = "8080")]
        port: u16,

        /// Number of GPU layers to offload (-1 = all)
        #[arg(long, short = 'g', default_value = "-1")]
        ngl: i32,

        /// Context size in tokens
        #[arg(long, short = 'c', default_value = "4096")]
        ctx_size: u32,
    },

    /// Start llmfit REST API server for cluster/node scheduling workflows
    #[command(long_about = "\
Start llmfit REST API server for cluster/node scheduling workflows.

Exposes llmfit's hardware detection and model fitting as a REST API. Useful
for multi-node clusters, CI pipelines, and orchestration systems that need
to query hardware capabilities and model recommendations programmatically.

PRECONDITIONS:
  The specified host:port must be available for binding.

SIDE EFFECTS:
  Binds an HTTP server on the specified host and port (default 127.0.0.1:8787).
  Also serves the local web dashboard at `/` on the same host/port.
  Runs until terminated.

EXIT CODES:
  0  Clean shutdown
  1  Port binding failure or runtime error

AGENT USAGE:
  llmfit serve --port 8787
  llmfit serve --host 0.0.0.0 --port 8787  # expose to other machines
  All endpoints return JSON. See API.md for the full endpoint reference.")]
    Serve {
        /// Host interface to bind
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port to listen on
        #[arg(long, default_value = "8787")]
        port: u16,
    },
}

/// Bundled hardware override options from CLI flags.
pub(crate) struct HardwareOverrides {
    pub memory: Option<String>,
    pub ram: Option<String>,
    pub cpu_cores: Option<usize>,
}

/// Detect system specs with optional hardware overrides.
/// RAM override is applied before GPU VRAM so that `--memory` takes precedence
/// on unified-memory systems where `--ram` would also update VRAM.
pub(crate) fn detect_specs(overrides: &HardwareOverrides) -> SystemSpecs {
    let mut specs = SystemSpecs::detect();

    if let Some(ram_str) = &overrides.ram {
        match llmfit_core::hardware::parse_memory_size(ram_str) {
            Some(gb) => specs = specs.with_ram_override(gb),
            None => {
                eprintln!(
                    "Warning: could not parse --ram value '{}'. Expected format: 64G, 128000M, 1T",
                    ram_str
                );
            }
        }
    }

    if let Some(mem_str) = &overrides.memory {
        match llmfit_core::hardware::parse_memory_size(mem_str) {
            Some(gb) => specs = specs.with_gpu_memory_override(gb),
            None => {
                eprintln!(
                    "Warning: could not parse --memory value '{}'. Expected format: 32G, 32000M, 1.5T",
                    mem_str
                );
            }
        }
    }

    if let Some(cores) = overrides.cpu_cores {
        specs = specs.with_cpu_core_override(cores);
    }

    specs
}

fn resolve_context_limit(max_context: Option<u32>) -> Option<u32> {
    if max_context.is_some() {
        return max_context;
    }

    let Ok(raw) = std::env::var("OLLAMA_CONTEXT_LENGTH") else {
        return None;
    };
    match raw.trim().parse::<u32>() {
        Ok(v) if v > 0 => Some(v),
        _ => {
            eprintln!(
                "Warning: could not parse OLLAMA_CONTEXT_LENGTH='{}'. Expected a positive integer.",
                raw
            );
            None
        }
    }
}

fn dashboard_pid_path() -> Option<std::path::PathBuf> {
    llmfit_core::update::cache_dir().map(|d| d.join("dashboard.pid"))
}

fn write_dashboard_pid(pid: u32) {
    let Some(path) = dashboard_pid_path() else {
        return;
    };
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(path, pid.to_string());
}

struct DashboardGuard {
    child: std::process::Child,
}

impl Drop for DashboardGuard {
    fn drop(&mut self) {
        let _ = self.child.kill();
        if let Some(path) = dashboard_pid_path() {
            let _ = std::fs::remove_file(path);
        }
    }
}

fn dashboard_target_from_env() -> (String, u16) {
    let host = std::env::var("LLMFIT_DASHBOARD_HOST")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| DEFAULT_DASHBOARD_HOST.to_string());

    let port = std::env::var("LLMFIT_DASHBOARD_PORT")
        .ok()
        .and_then(|raw| match raw.trim().parse::<u16>() {
            Ok(value) => Some(value),
            Err(_) => {
                eprintln!(
                    "Warning: invalid LLMFIT_DASHBOARD_PORT='{}'. Using {}.",
                    raw, DEFAULT_DASHBOARD_PORT
                );
                None
            }
        })
        .unwrap_or(DEFAULT_DASHBOARD_PORT);

    (host, port)
}

fn dashboard_reachable(host: &str, port: u16) -> bool {
    let Ok(mut addrs) = format!("{host}:{port}").to_socket_addrs() else {
        return false;
    };
    let Some(addr) = addrs.next() else {
        return false;
    };
    TcpStream::connect_timeout(&addr, Duration::from_millis(250)).is_ok()
}

fn ensure_dashboard_available(
    overrides: &HardwareOverrides,
    context_limit: Option<u32>,
) -> Option<DashboardGuard> {
    let (host, port) = dashboard_target_from_env();
    let _url = format!("http://{}:{}/", host, port);

    if dashboard_reachable(&host, port) {
        return None;
    }

    let exe = match std::env::current_exe() {
        Ok(path) => path,
        Err(err) => {
            eprintln!("Warning: could not resolve llmfit executable for dashboard launch: {err}");
            return None;
        }
    };

    let mut command = std::process::Command::new(exe);
    command.arg("--no-dashboard");
    if let Some(memory) = &overrides.memory {
        command.arg("--memory").arg(memory);
    }
    if let Some(ram) = &overrides.ram {
        command.arg("--ram").arg(ram);
    }
    if let Some(cores) = overrides.cpu_cores {
        command.arg("--cpu-cores").arg(cores.to_string());
    }
    if let Some(ctx) = context_limit {
        command.arg("--max-context").arg(ctx.to_string());
    }

    command
        .arg("serve")
        .arg("--host")
        .arg(&host)
        .arg("--port")
        .arg(port.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(err) => {
            eprintln!("Warning: could not start dashboard server: {err}");
            return None;
        }
    };

    write_dashboard_pid(child.id());

    for _ in 0..20 {
        if dashboard_reachable(&host, port) {
            return Some(DashboardGuard { child });
        }

        match child.try_wait() {
            Ok(Some(status)) => {
                eprintln!(
                    "Warning: dashboard server exited early (status: {}). Run `llmfit serve` to inspect logs.",
                    status
                );
                return None;
            }
            Ok(None) => {}
            Err(err) => {
                eprintln!("Warning: could not check dashboard server status: {err}");
                return None;
            }
        }

        thread::sleep(Duration::from_millis(100));
    }

    Some(DashboardGuard { child })
}

fn run_fit(
    perfect: bool,
    tool_use: bool,
    limit: Option<usize>,
    sort: SortColumn,
    json: bool,
    csv: bool,
    overrides: &HardwareOverrides,
    context_limit: Option<u32>,
) {
    use llmfit_core::providers::{
        self as provs, DockerModelRunnerProvider, LlamaCppProvider, LmStudioProvider, MlxProvider,
        ModelProvider, OllamaProvider,
    };

    let specs = detect_specs(overrides);
    let db = ModelDatabase::new();

    if !json && !csv {
        specs.display();
    }

    // Query installed models across local providers so that `fit.installed`
    // is populated in both text and JSON output — same behaviour as `recommend`.
    let ollama_installed = OllamaProvider::new().installed_models();
    let mlx_installed = MlxProvider::new().installed_models();
    let llamacpp_installed = LlamaCppProvider::new().installed_models();
    let docker_mr_installed = DockerModelRunnerProvider::new().installed_models();
    let lmstudio_installed = LmStudioProvider::new().installed_models();

    let hidden: usize = db
        .get_all_models()
        .iter()
        .filter(|m| !backend_compatible(m, &specs))
        .count();

    let mut fits: Vec<ModelFit> = db
        .get_all_models()
        .iter()
        .filter(|m| backend_compatible(m, &specs))
        .map(|m| {
            let mut fit = ModelFit::analyze_with_context_limit(m, &specs, context_limit);
            fit.installed = provs::is_model_installed(&m.name, &ollama_installed)
                || provs::is_model_installed_mlx(&m.name, &mlx_installed)
                || provs::is_model_installed_llamacpp(&m.name, &llamacpp_installed)
                || provs::is_model_installed_docker_mr(&m.name, &docker_mr_installed)
                || provs::is_model_installed_lmstudio(&m.name, &lmstudio_installed);
            fit
        })
        .collect();

    if perfect {
        fits.retain(|f| f.fit_level == llmfit_core::fit::FitLevel::Perfect);
    }

    if tool_use {
        fits.retain(|f| {
            f.model
                .capabilities
                .contains(&llmfit_core::models::Capability::ToolUse)
        });
    }

    fits = llmfit_core::fit::rank_models_by_fit_opts_col(fits, false, sort);

    if let Some(n) = limit {
        fits.truncate(n);
    }

    if csv {
        display::display_csv_fits(&fits);
    } else if json {
        display::display_json_fits(&specs, &fits);
    } else {
        if hidden > 0 {
            eprintln!(
                "({} model{} hidden — incompatible backend)",
                hidden,
                if hidden == 1 { "" } else { "s" }
            );
        }
        display::display_model_fits(&fits);
    }
}

fn fit_matches_filter(fit: &ModelFit, filter: FitArg) -> bool {
    match filter {
        FitArg::All => true,
        FitArg::Perfect => fit.fit_level == llmfit_core::fit::FitLevel::Perfect,
        FitArg::Good => fit.fit_level == llmfit_core::fit::FitLevel::Good,
        FitArg::Marginal => fit.fit_level == llmfit_core::fit::FitLevel::Marginal,
        FitArg::Tight => fit.fit_level == llmfit_core::fit::FitLevel::TooTight,
        FitArg::Runnable => fit.fit_level != llmfit_core::fit::FitLevel::TooTight,
    }
}

fn find_name_index_by_selector<T>(
    items: &[T],
    selector: &str,
    get_name: impl Fn(&T) -> &str,
) -> Result<usize, String> {
    let needle = selector.trim().to_lowercase();
    if needle.is_empty() {
        return Err("Model selector cannot be empty".to_string());
    }

    if let Some((idx, _)) = items
        .iter()
        .enumerate()
        .find(|(_, item)| get_name(item).to_lowercase() == needle)
    {
        return Ok(idx);
    }

    let matches: Vec<(usize, String)> = items
        .iter()
        .enumerate()
        .filter_map(|(i, item)| {
            let name = get_name(item);
            if name.to_lowercase().contains(&needle) {
                Some((i, name.to_string()))
            } else {
                None
            }
        })
        .collect();

    match matches.as_slice() {
        [] => Err(format!("No model found matching '{}'", selector)),
        [(idx, _)] => Ok(*idx),
        _ => {
            let names = matches
                .iter()
                .take(8)
                .map(|(_, name)| format!("  - {}", name))
                .collect::<Vec<_>>()
                .join("\n");
            Err(format!(
                "Multiple models match '{}'. Please be more specific:\n{}",
                selector, names
            ))
        }
    }
}

fn find_fit_index_by_selector(fits: &[ModelFit], selector: &str) -> Result<usize, String> {
    find_name_index_by_selector(fits, selector, |fit| fit.model.name.as_str())
}

fn run_diff(
    model_a: Option<String>,
    model_b: Option<String>,
    fit_filter: FitArg,
    sort: SortColumn,
    limit: usize,
    json: bool,
    overrides: &HardwareOverrides,
    context_limit: Option<u32>,
) {
    if limit < 2 {
        eprintln!("Error: --limit must be at least 2 for diff");
        std::process::exit(1);
    }

    if (model_a.is_some() && model_b.is_none()) || (model_a.is_none() && model_b.is_some()) {
        eprintln!("Error: provide both model selectors, or neither to auto-compare top N");
        std::process::exit(1);
    }

    let specs = detect_specs(overrides);
    let db = ModelDatabase::new();

    let mut fits: Vec<ModelFit> = db
        .get_all_models()
        .iter()
        .filter(|m| backend_compatible(m, &specs))
        .map(|m| ModelFit::analyze_with_context_limit(m, &specs, context_limit))
        .collect();

    fits.retain(|f| fit_matches_filter(f, fit_filter));
    fits = llmfit_core::fit::rank_models_by_fit_opts_col(fits, false, sort);

    let selected: Vec<ModelFit> =
        if let (Some(a), Some(b)) = (model_a.as_deref(), model_b.as_deref()) {
            let a_idx = match find_fit_index_by_selector(&fits, a) {
                Ok(i) => i,
                Err(e) => {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            };
            let b_idx = match find_fit_index_by_selector(&fits, b) {
                Ok(i) => i,
                Err(e) => {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            };

            if a_idx == b_idx {
                eprintln!("Error: both selectors resolved to the same model");
                std::process::exit(1);
            }

            vec![fits[a_idx].clone(), fits[b_idx].clone()]
        } else {
            if fits.len() < 2 {
                eprintln!("Error: need at least 2 models after filtering to compare");
                std::process::exit(1);
            }
            fits.into_iter().take(limit).collect()
        };

    if json {
        display::display_json_diff_fits(&specs, &selected);
    } else {
        specs.display();
        display::display_model_diff(&selected, sort.label());
    }
}

fn run_tui(overrides: &HardwareOverrides, context_limit: Option<u32>) -> std::io::Result<()> {
    // Setup terminal
    crossterm::terminal::enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    crossterm::execute!(
        stdout,
        crossterm::terminal::EnterAlternateScreen,
        crossterm::event::EnableMouseCapture
    )?;

    let backend = ratatui::backend::CrosstermBackend::new(stdout);
    let mut terminal = ratatui::Terminal::new(backend)?;
    draw_boot_screen(&mut terminal, "Detecting system hardware...")?;

    // Create app state
    let specs = detect_specs(overrides);
    draw_boot_screen(&mut terminal, "Loading providers and models...")?;
    let mut app = tui_app::App::with_specs_and_context(specs, context_limit);

    // Main loop
    loop {
        terminal.draw(|frame| {
            tui_ui::draw(frame, &mut app);
        })?;

        tui_events::handle_events(&mut app)?;

        if app.should_quit {
            break;
        }
    }

    // Restore terminal
    crossterm::terminal::disable_raw_mode()?;
    crossterm::execute!(
        terminal.backend_mut(),
        crossterm::terminal::LeaveAlternateScreen,
        crossterm::event::DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    Ok(())
}

fn draw_boot_screen(
    terminal: &mut ratatui::Terminal<ratatui::backend::CrosstermBackend<std::io::Stdout>>,
    message: &str,
) -> std::io::Result<()> {
    use ratatui::layout::{Constraint, Direction, Layout};
    use ratatui::style::{Modifier, Style};
    use ratatui::text::{Line, Span};
    use ratatui::widgets::{Block, Borders, Paragraph};

    terminal.draw(|frame| {
        let area = frame.area();
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(45),
                Constraint::Length(3),
                Constraint::Percentage(52),
            ])
            .split(area);

        let block = Block::default()
            .borders(Borders::ALL)
            .title(" llmfit ")
            .title_style(Style::default().add_modifier(Modifier::BOLD));
        let line = Line::from(vec![
            Span::raw(" "),
            Span::styled("Loading: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(message),
        ]);
        frame.render_widget(Paragraph::new(line).block(block), layout[1]);
    })?;
    Ok(())
}

fn run_recommend(
    limit: usize,
    use_case: Option<String>,
    min_fit: String,
    runtime_filter: String,
    force_runtime: Option<String>,
    capability: Option<String>,
    license: Option<String>,
    json: bool,
    csv: bool,
    overrides: &HardwareOverrides,
    context_limit: Option<u32>,
) {
    let specs = detect_specs(overrides);
    let db = ModelDatabase::new();

    // Parse --force-runtime into an InferenceRuntime if provided
    let forced_rt = force_runtime
        .as_deref()
        .map(|rt| match rt.to_lowercase().as_str() {
            "mlx" => llmfit_core::fit::InferenceRuntime::Mlx,
            "llamacpp" | "llama.cpp" | "llama_cpp" => llmfit_core::fit::InferenceRuntime::LlamaCpp,
            "vllm" => llmfit_core::fit::InferenceRuntime::Vllm,
            other => {
                eprintln!(
                    "Unknown runtime '{}'. Valid options: mlx, llamacpp, vllm",
                    other
                );
                std::process::exit(1);
            }
        });

    // Query installed models across local providers so that `fit.installed`
    // is populated for CLI output (same behavior as the TUI). This also causes
    // backends like Docker Model Runner to receive a probe request when
    // DOCKER_MODEL_RUNNER_HOST is set.
    use llmfit_core::providers::{
        self as provs, DockerModelRunnerProvider, LlamaCppProvider, LmStudioProvider, MlxProvider,
        ModelProvider, OllamaProvider,
    };
    let ollama_installed = OllamaProvider::new().installed_models();
    let mlx_installed = MlxProvider::new().installed_models();
    let llamacpp_installed = LlamaCppProvider::new().installed_models();
    let docker_mr_installed = DockerModelRunnerProvider::new().installed_models();
    let lmstudio_installed = LmStudioProvider::new().installed_models();

    let mut fits: Vec<ModelFit> = db
        .get_all_models()
        .iter()
        .filter(|m| backend_compatible(m, &specs))
        .map(|m| {
            let mut fit =
                ModelFit::analyze_with_forced_runtime(m, &specs, context_limit, forced_rt);
            fit.installed = provs::is_model_installed(&m.name, &ollama_installed)
                || provs::is_model_installed_mlx(&m.name, &mlx_installed)
                || provs::is_model_installed_llamacpp(&m.name, &llamacpp_installed)
                || provs::is_model_installed_docker_mr(&m.name, &docker_mr_installed)
                || provs::is_model_installed_lmstudio(&m.name, &lmstudio_installed);
            fit
        })
        .collect();

    // Filter by minimum fit level
    let min_level = match min_fit.to_lowercase().as_str() {
        "perfect" => llmfit_core::fit::FitLevel::Perfect,
        "good" => llmfit_core::fit::FitLevel::Good,
        "marginal" => llmfit_core::fit::FitLevel::Marginal,
        _ => llmfit_core::fit::FitLevel::Marginal,
    };
    fits.retain(|f| match (min_level, f.fit_level) {
        (llmfit_core::fit::FitLevel::Marginal, llmfit_core::fit::FitLevel::TooTight) => false,
        (
            llmfit_core::fit::FitLevel::Good,
            llmfit_core::fit::FitLevel::TooTight | llmfit_core::fit::FitLevel::Marginal,
        ) => false,
        (llmfit_core::fit::FitLevel::Perfect, llmfit_core::fit::FitLevel::Perfect) => true,
        (llmfit_core::fit::FitLevel::Perfect, _) => false,
        _ => true,
    });

    // Hide MLX-only models on non-Apple Silicon systems
    let is_apple_silicon =
        specs.backend == llmfit_core::hardware::GpuBackend::Metal && specs.unified_memory;
    if !is_apple_silicon {
        fits.retain(|f| !f.model.is_mlx_only());
    }

    // Filter by runtime
    match runtime_filter.to_lowercase().as_str() {
        "mlx" => fits.retain(|f| f.runtime == llmfit_core::fit::InferenceRuntime::Mlx),
        "llamacpp" | "llama.cpp" | "llama_cpp" => {
            fits.retain(|f| f.runtime == llmfit_core::fit::InferenceRuntime::LlamaCpp)
        }
        "vllm" => fits.retain(|f| f.runtime == llmfit_core::fit::InferenceRuntime::Vllm),
        _ => {} // "any" or unrecognized — keep all
    }

    // Filter by use case if specified
    if let Some(ref uc) = use_case {
        let target = match uc.to_lowercase().as_str() {
            "coding" | "code" => Some(llmfit_core::models::UseCase::Coding),
            "reasoning" | "reason" => Some(llmfit_core::models::UseCase::Reasoning),
            "chat" => Some(llmfit_core::models::UseCase::Chat),
            "multimodal" | "vision" => Some(llmfit_core::models::UseCase::Multimodal),
            "embedding" | "embed" => Some(llmfit_core::models::UseCase::Embedding),
            "general" => Some(llmfit_core::models::UseCase::General),
            _ => None,
        };
        if let Some(target_uc) = target {
            fits.retain(|f| f.use_case == target_uc);
        }
    }

    // Filter by capability if specified
    if let Some(ref caps_str) = capability {
        let requested: Vec<&str> = caps_str.split(',').map(|s| s.trim()).collect();
        fits.retain(|f| {
            requested
                .iter()
                .all(|req| match req.to_lowercase().as_str() {
                    "vision" => f
                        .model
                        .capabilities
                        .contains(&llmfit_core::models::Capability::Vision),
                    "tool_use" | "tools" | "tool-use" | "function_calling" => f
                        .model
                        .capabilities
                        .contains(&llmfit_core::models::Capability::ToolUse),
                    _ => true,
                })
        });
    }

    // Filter by license if specified
    if let Some(ref lic_str) = license {
        fits.retain(|f| llmfit_core::models::matches_license_filter(&f.model.license, lic_str));
    }

    fits = llmfit_core::fit::rank_models_by_fit(fits);
    fits.truncate(limit);

    if csv {
        display::display_csv_fits(&fits);
    } else if json {
        display::display_json_fits(&specs, &fits);
    } else {
        if !fits.is_empty() {
            specs.display();
        }
        display::display_model_fits(&fits);
    }
}

fn run_download(
    model: &str,
    quant: Option<&str>,
    budget: Option<f64>,
    list_only: bool,
    overrides: &HardwareOverrides,
) {
    use llmfit_core::providers::LlamaCppProvider;

    let provider = LlamaCppProvider::new();

    // Resolve repo ID: try known mapping, then treat as repo, then search
    let repo_id = if model.contains('/') {
        model.to_string()
    } else if let Some(repo) = llmfit_core::providers::gguf_pull_tag(model) {
        repo
    } else {
        // Search HuggingFace
        println!(
            "Searching HuggingFace for GGUF models matching '{}'...",
            model
        );
        let results = LlamaCppProvider::search_hf_gguf(model);
        if results.is_empty() {
            eprintln!(
                "No GGUF models found for '{}'. Try a different search term.",
                model
            );
            eprintln!("Tip: use 'llmfit hf-search <query>' to browse available models.");
            std::process::exit(1);
        }
        if results.len() > 1 && !list_only {
            println!("\nFound {} repositories:", results.len());
            for (i, (id, desc)) in results.iter().enumerate().take(10) {
                println!("  {}. {} ({})", i + 1, id, desc);
            }
            println!("\nUsing first result: {}", results[0].0);
        }
        results[0].0.clone()
    };

    // List available GGUF files
    println!("Fetching available files from {}...", repo_id);
    let files = LlamaCppProvider::list_repo_gguf_files(&repo_id);
    if files.is_empty() {
        eprintln!("No GGUF files found in repository '{}'.", repo_id);
        eprintln!("Make sure this is a valid GGUF repository on HuggingFace.");
        std::process::exit(1);
    }

    if list_only {
        println!("\nAvailable GGUF files in {}:", repo_id);
        println!("{:<60} {:>10}", "Filename", "Size");
        println!("{}", "-".repeat(72));
        for (filename, size) in &files {
            let size_str = if *size > 1_073_741_824 {
                format!("{:.1} GB", *size as f64 / 1_073_741_824.0)
            } else {
                format!("{:.0} MB", *size as f64 / 1_048_576.0)
            };
            println!("{:<60} {:>10}", filename, size_str);
        }
        return;
    }

    // Select the file to download
    let (filename, file_size) = if let Some(q) = quant {
        // User specified a quantization
        let q_lower = q.to_lowercase();
        if let Some((f, s)) = files
            .iter()
            .find(|(f, _)| f.to_lowercase().contains(&q_lower))
        {
            (f.clone(), *s)
        } else {
            eprintln!(
                "No GGUF file found matching quantization '{}' in {}.",
                q, repo_id
            );
            eprintln!("\nAvailable files:");
            for (f, s) in &files {
                let size_str = format!("{:.1} GB", *s as f64 / 1_073_741_824.0);
                eprintln!("  {} ({})", f, size_str);
            }
            std::process::exit(1);
        }
    } else {
        // Auto-select based on hardware budget
        let mem_budget = if let Some(b) = budget {
            b
        } else {
            let specs = detect_specs(overrides);
            specs
                .total_gpu_vram_gb
                .or(Some(specs.available_ram_gb))
                .unwrap_or(16.0)
        };
        if let Some(result) = LlamaCppProvider::select_best_gguf(&files, mem_budget) {
            println!(
                "Selected {} ({:.1} GB) for {:.0} GB memory budget",
                result.0,
                result.1 as f64 / 1_073_741_824.0,
                mem_budget
            );
            result
        } else {
            // Nothing fits — pick smallest
            let mut sorted = files.clone();
            sorted.sort_by_key(|(_, s)| *s);
            let (f, s) = sorted.first().expect("files list is not empty");
            println!(
                "Warning: No quantization fits within {:.0} GB. Downloading smallest: {} ({:.1} GB)",
                mem_budget,
                f,
                *s as f64 / 1_073_741_824.0
            );
            (f.clone(), *s)
        }
    };

    // If the selected file is one shard of a multi-part model, expand it
    // here so we can show the user the full size and part count up front.
    // The actual download is still driven by `download_gguf`, which performs
    // the same expansion internally.
    let shard_set = llmfit_core::providers::collect_shard_set(&files, &filename);
    let (display_name, display_size) = if let Some(ref shards) = shard_set {
        let total: u64 = shards.iter().map(|(_, s)| *s).sum();
        let first = shards[0].0.clone();
        println!(
            "\nDetected sharded model: {} parts (total {:.1} GB)",
            shards.len(),
            total as f64 / 1_073_741_824.0
        );
        for (i, (f, s)) in shards.iter().enumerate() {
            println!(
                "  [{}/{}] {} ({:.1} GB)",
                i + 1,
                shards.len(),
                f,
                *s as f64 / 1_073_741_824.0
            );
        }
        (first, total)
    } else {
        (filename.clone(), file_size)
    };

    println!(
        "\nDownloading {} ({:.1} GB) to {}",
        display_name,
        display_size as f64 / 1_073_741_824.0,
        provider.models_dir().display()
    );

    match provider.download_gguf(&repo_id, &filename) {
        Ok(handle) => {
            // Poll for progress
            loop {
                match handle.receiver.recv() {
                    Ok(llmfit_core::providers::PullEvent::Progress { status, percent }) => {
                        if let Some(p) = percent {
                            print!("\r\x1b[K  {:.1}% - {}", p, status);
                            use std::io::Write;
                            let _ = std::io::stdout().flush();
                        } else {
                            println!("  {}", status);
                        }
                    }
                    Ok(llmfit_core::providers::PullEvent::Done) => {
                        println!("\n\n✓ Download complete!");
                        // For sharded models, point at the first shard;
                        // llama.cpp auto-loads the rest from the same dir.
                        let primary = if let Some(ref shards) = shard_set {
                            shards[0].0.clone()
                        } else {
                            filename.clone()
                        };
                        let local_name = std::path::Path::new(&primary)
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or(&primary);
                        let dest = provider.models_dir().join(local_name);
                        if let Some(ref shards) = shard_set {
                            println!(
                                "  Saved {} shards to: {}",
                                shards.len(),
                                provider.models_dir().display()
                            );
                            for (f, _) in shards {
                                let n = std::path::Path::new(f)
                                    .file_name()
                                    .and_then(|n| n.to_str())
                                    .unwrap_or(f);
                                println!("    {}", n);
                            }
                        } else {
                            println!("  Saved to: {}", dest.display());
                        }
                        if provider.llama_cli_path().is_some() {
                            println!(
                                "\n  Run with: llmfit run {}",
                                local_name.trim_end_matches(".gguf")
                            );
                            println!("  Or directly: llama-cli -m {} -cnv", dest.display());
                        } else {
                            println!("\n  Install llama.cpp to run this model:");
                            println!("    brew install llama.cpp");
                            println!("    # or build from source:");
                            println!(
                                "    git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp"
                            );
                            println!("    cmake -B build && cmake --build build --config Release");
                            println!("\n  Then run: llama-cli -m {} -cnv", dest.display());
                        }
                        break;
                    }
                    Ok(llmfit_core::providers::PullEvent::Error(e)) => {
                        eprintln!("\n\n✗ Download failed: {}", e);
                        std::process::exit(1);
                    }
                    Err(_) => {
                        eprintln!("\n\n✗ Download channel closed unexpectedly");
                        std::process::exit(1);
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("Failed to start download: {}", e);
            std::process::exit(1);
        }
    }
}

fn run_update(trending: usize, downloads: usize, token: Option<String>, status: bool, clear: bool) {
    use llmfit_core::update;

    // ── --status ──────────────────────────────────────────────────────────
    if status {
        match update::cache_file() {
            Some(path) => {
                if path.exists() {
                    let models = update::load_cache();
                    let modified = std::fs::metadata(&path)
                        .and_then(|m| m.modified())
                        .ok()
                        .and_then(|t| std::time::SystemTime::now().duration_since(t).ok())
                        .map(|d| {
                            let days = d.as_secs() / 86_400;
                            if days == 0 {
                                "today".to_string()
                            } else if days == 1 {
                                "yesterday".to_string()
                            } else {
                                format!("{} days ago", days)
                            }
                        })
                        .unwrap_or_else(|| "unknown".to_string());
                    println!("Cache file : {}", path.display());
                    println!("Models     : {}", models.len());
                    println!("Last update: {}", modified);
                } else {
                    println!("No cache found at {}", path.display());
                    println!("Run 'llmfit update' to fetch the latest models.");
                }
            }
            None => eprintln!("Cannot determine cache directory."),
        }
        return;
    }

    // ── --clear ───────────────────────────────────────────────────────────
    if clear {
        match update::clear_cache() {
            Ok(0) => println!("Cache is already empty."),
            Ok(n) => println!("Cleared {} cached model(s).", n),
            Err(e) => eprintln!("Error: {}", e),
        }
        return;
    }

    // ── fetch ─────────────────────────────────────────────────────────────
    // Resolve HF token: CLI flag wins, then environment variables.
    let resolved_token = token
        .or_else(|| std::env::var("HF_TOKEN").ok())
        .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok());

    let opts = update::UpdateOptions {
        trending_limit: trending,
        downloads_limit: downloads,
        token: resolved_token,
    };

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  llmfit — Model Database Update");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    match update::update_model_cache(&opts, |msg| println!("{}", msg)) {
        Ok((new_count, total)) => {
            println!();
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            if new_count > 0 {
                println!("  Added {} new model(s) to the cache.", new_count);
            } else {
                println!("  No new models found — cache is up to date.");
            }
            println!("  Total cached: {}", total);
            if let Some(p) = update::cache_file() {
                println!("  Cache file  : {}", p.display());
            }
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!();
            println!("Run 'llmfit' or 'llmfit fit' to see results with the updated list.");
        }
        Err(e) => {
            eprintln!("Update failed: {}", e);
            std::process::exit(1);
        }
    }
}

fn run_hf_search(query: &str, limit: usize) {
    use llmfit_core::providers::LlamaCppProvider;

    println!(
        "Searching HuggingFace for GGUF models matching '{}'...\n",
        query
    );
    let results = LlamaCppProvider::search_hf_gguf(query);

    if results.is_empty() {
        println!("No GGUF models found. Try a different search term.");
        return;
    }

    println!("{:<50} Type", "Repository");
    println!("{}", "-".repeat(65));
    for (id, desc) in results.iter().take(limit) {
        println!("{:<50} {}", id, desc);
    }

    println!("\nTo download: llmfit download <repository>");
    println!("To list files: llmfit download <repository> --list");
}

fn run_model(model: &str, server: bool, port: u16, ngl: i32, ctx_size: u32) {
    use llmfit_core::providers::LlamaCppProvider;

    let provider = LlamaCppProvider::new();

    // Find the model file
    let model_path = if std::path::Path::new(model).exists() {
        std::path::PathBuf::from(model)
    } else {
        // Search in cache directory
        let gguf_files = provider.list_gguf_files();
        let search = model.to_lowercase();
        let found = gguf_files.into_iter().find(|p| {
            p.file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.to_lowercase().contains(&search))
                .unwrap_or(false)
        });
        match found {
            Some(p) => p,
            None => {
                eprintln!("Model '{}' not found.", model);
                eprintln!("\nAvailable models in {}:", provider.models_dir().display());
                for f in provider.list_gguf_files() {
                    eprintln!("  {}", f.file_name().unwrap_or_default().to_string_lossy());
                }
                eprintln!("\nUse 'llmfit download <model>' to download a model first.");
                std::process::exit(1);
            }
        }
    };

    if server {
        let Some(bin) = provider.llama_server_path() else {
            eprintln!("llama-server not found in PATH.");
            eprintln!("Install llama.cpp: brew install llama.cpp");
            eprintln!("Or build from source: https://github.com/ggml-org/llama.cpp");
            std::process::exit(1);
        };

        println!(
            "Starting llama-server on port {} with {}...",
            port,
            model_path.display()
        );
        let status = std::process::Command::new(bin)
            .args([
                "-m",
                model_path.to_str().unwrap_or(""),
                "--port",
                &port.to_string(),
                "-ngl",
                &ngl.to_string(),
                "-c",
                &ctx_size.to_string(),
            ])
            .status();

        match status {
            Ok(s) if !s.success() => {
                std::process::exit(s.code().unwrap_or(1));
            }
            Err(e) => {
                eprintln!("Failed to run llama-server: {}", e);
                std::process::exit(1);
            }
            _ => {}
        }
    } else {
        let Some(bin) = provider.llama_cli_path() else {
            eprintln!("llama-cli not found in PATH.");
            eprintln!("Install llama.cpp: brew install llama.cpp");
            eprintln!("Or build from source: https://github.com/ggml-org/llama.cpp");
            std::process::exit(1);
        };

        println!("Running {} with llama-cli...\n", model_path.display());
        let status = std::process::Command::new(bin)
            .args([
                "-m",
                model_path.to_str().unwrap_or(""),
                "-ngl",
                &ngl.to_string(),
                "-c",
                &ctx_size.to_string(),
                "-cnv",
            ])
            .status();

        match status {
            Ok(s) if !s.success() => {
                std::process::exit(s.code().unwrap_or(1));
            }
            Err(e) => {
                eprintln!("Failed to run llama-cli: {}", e);
                std::process::exit(1);
            }
            _ => {}
        }
    }
}

fn run_plan(
    model_selector: &str,
    context: u32,
    quant: Option<String>,
    kv_quant: Option<String>,
    target_tps: Option<f64>,
    json: bool,
    overrides: &HardwareOverrides,
) -> Result<(), String> {
    let db = ModelDatabase::new();
    let specs = detect_specs(overrides);
    let model = resolve_model_selector(db.get_all_models(), model_selector)?;

    let kv_quant = match kv_quant {
        Some(s) => Some(llmfit_core::models::KvQuant::parse(&s).ok_or_else(|| {
            format!(
                "Unsupported --kv-quant '{}'. Valid: fp16, fp8, q8_0, q4_0, tq",
                s
            )
        })?),
        None => None,
    };

    if kv_quant == Some(llmfit_core::models::KvQuant::TurboQuant) {
        eprintln!(
            "warning: TurboQuant is experimental, not in upstream vLLM yet. \
             See https://github.com/0xSero/turboquant for the research integration. \
             Numbers below assume the documented compression ratio applied only to \
             full attention layers."
        );
    }

    let request = PlanRequest {
        context,
        quant,
        target_tps,
        kv_quant,
    };
    let plan = estimate_model_plan(model, &request, &specs)?;

    if json {
        display::display_json_plan(&plan);
    } else {
        specs.display();
        display::display_model_plan(&plan);
    }

    Ok(())
}

fn main() {
    let cli = Cli::parse();
    let context_limit = resolve_context_limit(cli.max_context);
    let overrides = HardwareOverrides {
        memory: cli.memory,
        ram: cli.ram,
        cpu_cores: cli.cpu_cores,
    };
    let auto_dashboard = !cli.no_dashboard
        && !cli.json
        && !matches!(cli.command.as_ref(), Some(Commands::Serve { .. }));

    let _dashboard_guard = if auto_dashboard {
        ensure_dashboard_available(&overrides, context_limit)
    } else {
        None
    };

    // If a subcommand is given, use classic CLI mode
    if let Some(command) = cli.command {
        match command {
            Commands::System => {
                let specs = detect_specs(&overrides);
                if cli.json {
                    display::display_json_system(&specs);
                } else {
                    specs.display();
                }
            }

            Commands::List { sort } => {
                let db = ModelDatabase::new();
                if cli.json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(db.get_all_models())
                            .expect("JSON serialization failed")
                    );
                } else {
                    display::display_all_models(db.get_all_models(), sort.into());
                }
            }

            Commands::Fit {
                perfect,
                tool_use,
                limit,
                sort,
            } => {
                run_fit(
                    perfect,
                    tool_use,
                    limit,
                    sort.into(),
                    cli.json,
                    cli.csv,
                    &overrides,
                    context_limit,
                );
            }

            Commands::Search { query } => {
                let db = ModelDatabase::new();
                let results = db.find_model(&query);
                display::display_search_results(&results, &query);
            }

            Commands::Info { model } => {
                let db = ModelDatabase::new();
                let specs = detect_specs(&overrides);
                let models = db.get_all_models();

                let idx = match find_name_index_by_selector(models, &model, |m| m.name.as_str()) {
                    Ok(i) => i,
                    Err(err) => {
                        println!("\n{}", err);
                        return;
                    }
                };

                let fit = ModelFit::analyze_with_context_limit(&models[idx], &specs, context_limit);
                if cli.json {
                    display::display_json_fits(&specs, &[fit]);
                } else {
                    display::display_model_detail(&fit);
                }
            }

            Commands::Diff {
                model_a,
                model_b,
                sort,
                fit,
                limit,
            } => {
                run_diff(
                    model_a,
                    model_b,
                    fit,
                    sort.into(),
                    limit,
                    cli.json,
                    &overrides,
                    context_limit,
                );
            }

            Commands::Plan {
                model,
                context,
                quant,
                kv_quant,
                target_tps,
            } => {
                if let Err(err) = run_plan(
                    &model, context, quant, kv_quant, target_tps, cli.json, &overrides,
                ) {
                    eprintln!("Error: {}", err);
                    std::process::exit(1);
                }
            }

            Commands::Recommend {
                limit,
                use_case,
                min_fit,
                runtime,
                force_runtime,
                capability,
                license,
                json,
            } => {
                run_recommend(
                    limit,
                    use_case,
                    min_fit,
                    runtime,
                    force_runtime,
                    capability,
                    license,
                    json,
                    cli.csv,
                    &overrides,
                    context_limit,
                );
            }

            Commands::Download {
                model,
                quant,
                budget,
                list,
            } => {
                run_download(&model, quant.as_deref(), budget, list, &overrides);
            }

            Commands::HfSearch { query, limit } => {
                run_hf_search(&query, limit);
            }

            Commands::Update {
                trending,
                downloads,
                token,
                status,
                clear,
            } => {
                run_update(trending, downloads, token, status, clear);
            }

            Commands::Run {
                model,
                server,
                port,
                ngl,
                ctx_size,
            } => {
                run_model(&model, server, port, ngl, ctx_size);
            }

            Commands::Serve { host, port } => {
                if let Err(err) = serve_api::run_serve(&host, port, &overrides, context_limit) {
                    eprintln!("Error: {}", err);
                    std::process::exit(1);
                }
            }
        }
        return;
    }

    // If --cli, --json, or --csv flag, use classic fit output
    if cli.cli || cli.json || cli.csv {
        run_fit(
            cli.perfect,
            cli.tool_use,
            cli.limit,
            cli.sort.into(),
            cli.json,
            cli.csv,
            &overrides,
            context_limit,
        );
        return;
    }

    // Default: launch TUI
    if let Err(e) = run_tui(&overrides, context_limit) {
        eprintln!("Error running TUI: {}", e);
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmfit_core::fit::{FitLevel, InferenceRuntime, RunMode, ScoreComponents};
    use llmfit_core::models::LlmModel;

    fn mock_fit(name: &str, fit_level: FitLevel) -> ModelFit {
        ModelFit {
            model: LlmModel {
                name: name.to_string(),
                provider: "test".to_string(),
                parameter_count: "7B".to_string(),
                parameters_raw: None,
                min_ram_gb: 4.0,
                recommended_ram_gb: 8.0,
                min_vram_gb: Some(4.0),
                quantization: "Q4_K_M".to_string(),
                context_length: 8192,
                use_case: "general".to_string(),
                is_moe: false,
                num_experts: None,
                active_experts: None,
                active_parameters: None,
                release_date: Some("2025-01-01".to_string()),
                gguf_sources: vec![],
                capabilities: vec![],
                format: llmfit_core::models::ModelFormat::default(),
                num_attention_heads: None,
                num_key_value_heads: None,
                num_hidden_layers: None,
                head_dim: None,
                attention_layout: None,
                license: None,
                hidden_size: None,
                moe_intermediate_size: None,
                vocab_size: None,
                shared_expert_intermediate_size: None,
            },
            fit_level,
            run_mode: RunMode::Gpu,
            memory_required_gb: 4.0,
            memory_available_gb: 8.0,
            utilization_pct: 50.0,
            notes: vec![],
            moe_offloaded_gb: None,
            score: 80.0,
            score_components: ScoreComponents {
                quality: 80.0,
                speed: 80.0,
                fit: 80.0,
                context: 80.0,
            },
            estimated_tps: 30.0,
            best_quant: "Q4_K_M".to_string(),
            use_case: llmfit_core::models::UseCase::General,
            runtime: InferenceRuntime::LlamaCpp,
            installed: false,
            fits_with_turboquant: false,
        }
    }

    #[test]
    fn fit_filter_runnable_excludes_too_tight() {
        let runnable = mock_fit("alpha/model", FitLevel::Good);
        let tight = mock_fit("beta/model", FitLevel::TooTight);
        assert!(fit_matches_filter(&runnable, FitArg::Runnable));
        assert!(!fit_matches_filter(&tight, FitArg::Runnable));
    }

    #[test]
    fn selector_prefers_exact_match() {
        let fits = vec![
            mock_fit("org/model-a", FitLevel::Perfect),
            mock_fit("org/model-a-instruct", FitLevel::Perfect),
        ];
        let idx = find_fit_index_by_selector(&fits, "org/model-a").expect("should resolve");
        assert_eq!(idx, 0);
    }

    #[test]
    fn selector_errors_on_ambiguous_partial() {
        let fits = vec![
            mock_fit("org/model-a", FitLevel::Perfect),
            mock_fit("org/model-a-instruct", FitLevel::Perfect),
        ];
        let err = find_fit_index_by_selector(&fits, "model-a").expect_err("should be ambiguous");
        assert!(err.contains("Multiple models match"));
    }

    #[test]
    fn generic_selector_prefers_exact_match_for_models() {
        let models = vec![
            LlmModel {
                name: "Qwen/Qwen3-Coder-Next-FP8".to_string(),
                provider: "Qwen".to_string(),
                parameter_count: "7B".to_string(),
                parameters_raw: None,
                min_ram_gb: 4.0,
                recommended_ram_gb: 8.0,
                min_vram_gb: Some(4.0),
                quantization: "Q4_K_M".to_string(),
                context_length: 8192,
                use_case: "general".to_string(),
                is_moe: false,
                num_experts: None,
                active_experts: None,
                active_parameters: None,
                release_date: None,
                gguf_sources: vec![],
                capabilities: vec![],
                format: llmfit_core::models::ModelFormat::default(),
                num_attention_heads: None,
                num_key_value_heads: None,
                num_hidden_layers: None,
                head_dim: None,
                attention_layout: None,
                license: None,
                hidden_size: None,
                moe_intermediate_size: None,
                vocab_size: None,
                shared_expert_intermediate_size: None,
            },
            LlmModel {
                name: "Qwen/Qwen3-Coder-Next".to_string(),
                provider: "Qwen".to_string(),
                parameter_count: "7B".to_string(),
                parameters_raw: None,
                min_ram_gb: 4.0,
                recommended_ram_gb: 8.0,
                min_vram_gb: Some(4.0),
                quantization: "Q4_K_M".to_string(),
                context_length: 8192,
                use_case: "general".to_string(),
                is_moe: false,
                num_experts: None,
                active_experts: None,
                active_parameters: None,
                release_date: None,
                gguf_sources: vec![],
                capabilities: vec![],
                format: llmfit_core::models::ModelFormat::default(),
                num_attention_heads: None,
                num_key_value_heads: None,
                num_hidden_layers: None,
                head_dim: None,
                attention_layout: None,
                license: None,
                hidden_size: None,
                moe_intermediate_size: None,
                vocab_size: None,
                shared_expert_intermediate_size: None,
            },
        ];

        let idx =
            find_name_index_by_selector(&models, "Qwen/Qwen3-Coder-Next", |m| m.name.as_str())
                .expect("should resolve exact model");
        assert_eq!(idx, 1);
    }
}
