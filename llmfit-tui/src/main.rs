mod display;
mod serve_api;
mod theme;
mod tui_app;
mod actions;

use crate::display::{get_model_fits, resolve_model_selector, estimate_model_plan, PlanRequest};
use clap::{Parser, Subcommand, ValueEnum};
use llmfit_core::cluster::ClusterSpecs;
use llmfit_core::db::ModelDatabase;
use llmfit_core::fit::{ModelFit, SortColumn, backend_compatible, FitArg};
use llmfit_core::hardware::SystemSpecs;
use std::process::exit;

#[derive(Parser, Debug)]
#[command(name = "llmfit", author, version, about, long_about = Some(r#"
A command-line tool and TUI for finding the best-fit LLM for your hardware.

GLOBAL FLAGS:
  --json      Output in JSON format (where applicable)
  --limit N   Limit the number of results
  --sort COL  Sort results by a specific column
  --perfect   Only show models that fit perfectly on the GPU

EXIT CODES:
  0  Success
  1  Error (model not found, command failed, etc.)

ENVIRONMENT VARIABLES:
  OLLAMA_MODELS  Path to Ollama models directory
  HF_HOME        Path to HuggingFace cache directory
  LLMFIT_MODELS_PATH  Path to custom model database file
"#), after_long_help = "For detailed help on a subcommand, use 'llmfit <COMMAND> -h'.")]
struct Cli {
    /// Filter models by name (e.g., "llama", "7b", "instruct")
    filter: Option<String>,

    /// Limit the number of results
    #[arg(short, long, default_value = "25")]
    limit: Option<usize>,

    /// Sort results by a specific column
    #[arg(short, long, default_value = "tps")]
    sort: SortColumn,

    /// Only show models that fit perfectly on the GPU
    #[arg(short, long)]
    perfect: bool,

    /// Output in JSON format (where applicable)
    #[arg(long)]
    json: bool,

    /// Run in non-interactive CLI mode instead of TUI
    #[arg(long)]
    cli: bool,

    /// Cap context length used for memory estimation (tokens).
    /// Falls back to OLLAMA_CONTEXT_LENGTH if not set.
    #[arg(long, value_name = "TOKENS", value_parser = clap::value_parser!(u32).range(1..))]
    max_context: Option<u32>,

    /// Use cluster mode (auto-loads saved cluster config).
    /// Overrides local hardware detection with cluster resources.
    #[arg(long)]
    cluster: bool,

    /// Disable cluster mode even if a cluster config exists.
    #[arg(long)]
    no_cluster: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// View detailed system specifications
    System {
        #[arg(short, long, default_value = "summary")]
        verbosity: Verbosity,
    },

    /// List all available models in the database
    List {
        filter: Option<String>,
        #[arg(short = 'q', long)]
        show_quantizations: bool,
        #[arg(short, long, default_value = "tps")]
        sort: SortColumn,
    },

    /// Find the best-fit models for your hardware
    Fit {
        filter: Option<String>,
        #[arg(short, long)]
        show_all: bool,
        #[arg(short, long, default_value = "tps")]
        sort: SortColumn,
        #[arg(long)]
        fit: Option<FitArg>,
    },

    /// Search for a model by name
    Search { query: String },

    /// Get detailed information about a specific model
    Info { model_selector: String },

    /// Compare two or more models
    Diff { models: Vec<String> },

    /// Plan resource allocation for a model
    Plan {
        model_selector: String,
        quantization: Option<String>,
        #[arg(long)]
        ram_gb: Option<f32>,
        #[arg(long)]
        cpu_cores: Option<u32>,
    },

    /// Recommend models based on use case and budget
    Recommend {
        use_case: Option<String>,
        budget: Option<f32>,
        #[arg(short, long, default_value = "tps")]
        sort: SortColumn,
    },
    
    /// Download a GGUF model from HuggingFace for use with llama.cpp
    #[command(long_about = "\\
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
    #[command(long_about = "\\
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

    /// Run a downloaded GGUF model with llama-cli or llama-server
    #[command(long_about = "\\
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
    #[command(long_about = "\\
Start llmfit REST API server for cluster/node scheduling workflows.

Exposes llmfit's hardware detection and model fitting as a REST API. Useful
for multi-node clusters, CI pipelines, and orchestration systems that need
to query hardware capabilities and model recommendations programmatically.

PRECONDITIONS:
  The specified host:port must be available for binding.

SIDE EFFECTS:
  Binds an HTTP server on the specified host and port (default 0.0.0.0:8787).
  Runs until terminated.

EXIT CODES:
  0  Clean shutdown
  1  Port binding failure or runtime error

AGENT USAGE:
  llmfit serve --port 8787
  All endpoints return JSON. See API.md for the full endpoint reference.")]
    Serve {
        /// Host interface to bind
        #[arg(long, default_value = "0.0.0.0")]
        host: String,

        /// Port to listen on
        #[arg(long, default_value = "8787")]
        port: u16,
    },

    /// Manage DGX Spark cluster configuration
    Cluster {
        #[command(subcommand)]
        action: ClusterAction,
    },
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
enum Verbosity {
    Summary,
    Full,
}

#[derive(Subcommand, Debug)]
enum ClusterAction {
    /// Add a new node to the cluster
    Add {
        #[arg(help = "URL of the node to add (e.g., http://<host>:8787)")]
        node_url: String,
    },
    /// Remove a node from the cluster
    Remove {
        #[arg(help = "URL of the node to remove")]
        node_url: String,
    },
    /// List all nodes in the cluster
    List,
    /// Clear the cluster configuration
    Clear,
}

fn main() {
    let cli = Cli::parse();
    let db = ModelDatabase::from_default_paths().unwrap();

    let system_specs = if cli.cluster && !cli.no_cluster {
        match ClusterSpecs::from_saved() {
            Ok(cluster_specs) => {
                println!("Cluster mode enabled. Using specs from {} nodes.", cluster_specs.nodes.len());
                SystemSpecs::from(cluster_specs)
            }
            Err(e) => {
                eprintln!("Could not load cluster specs: {}", e);
                exit(1);
            }
        }
    } else {
        let mut specs = SystemSpecs::new();
        if let Some(max_context) = cli.max_context {
            specs.set_max_context_length(max_context);
        }
        specs
    };

    if let Some(command) = cli.command {
        match command {
            Commands::System { verbosity } => {
                display::print_system_specs(&system_specs, cli.json, verbosity);
            }
            Commands::List {
                filter,
                show_quantizations,
                sort,
            } => {
                display::print_model_list(
                    &db,
                    cli.json,
                    filter.as_deref(),
                    show_quantizations,
                    sort.into(),
                );
            }
            Commands::Fit {
                filter,
                show_all,
                sort,
                fit,
            } => {
                let models = get_model_fits(
                    &db,
                    &system_specs,
                    filter.as_deref(),
                    cli.limit,
                    sort.into(),
                    fit,
                );
                display::print_model_fits(&system_specs, &models, cli.json, show_all);
            }
            Commands::Search { query } => {
                let models = get_model_fits(
                    &db,
                    &system_specs,
                    Some(&query),
                    cli.limit,
                    cli.sort.into(),
                    None,
                );
                display::print_model_fits(&system_specs, &models, cli.json, false);
            }
            Commands::Info { model_selector } => {
                let Some(model) = resolve_model_selector(&db, &model_selector) else {
                    eprintln!("Model not found: {}", model_selector);
                    exit(1);
                };
                display::print_model_info(&system_specs, model, cli.json);
            }
            Commands::Diff { models } => {
                let resolved_models: Vec<_> = models
                    .iter()
                    .map(|selector| {
                        resolve_model_selector(&db, selector).unwrap_or_else(|| {
                            eprintln!("Model not found: {}", selector);
                            exit(1);
                        })
                    })
                    .collect();
                display::print_model_diff(&system_specs, &resolved_models, cli.json);
            }
            Commands::Plan {
                model_selector,
                quantization,
                ram_gb,
                cpu_cores,
            } => {
                let Some(model) = resolve_model_selector(&db, &model_selector) else {
                    eprintln!("Model not found: {}", model_selector);
                    exit(1);
                };
                let plan_request = PlanRequest {
                    model,
                    quantization: quantization.as_deref(),
                    system_ram_gb: ram_gb,
                    system_cpu_cores: cpu_cores,
                };
                let plan = estimate_model_plan(&system_specs, &plan_request, cli.json);
                display::print_plan(&plan, cli.json);
            }
            Commands::Recommend {
                use_case,
                budget,
                sort,
            } => {
                let models = get_model_fits(
                    &db,
                    &system_specs,
                    use_case.as_deref(),
                    cli.limit,
                    sort.into(),
                    None,
                );
                display::print_recommendations(&models, budget, cli.json);
            }
            Commands::Download {
                model,
                quant,
                budget,
                list,
            } => {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(actions::download(
                    &system_specs,
                    model,
                    quant,
                    budget,
                    list,
                ));
            }
            Commands::HfSearch { query, limit } => {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(actions::hf_search(query, limit));
            }
            Commands::Run {
                model,
                server,
                port,
                ngl,
                ctx_size,
            } => {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(actions::run(
                    &system_specs,
                    model,
                    server,
                    port,
                    ngl,
                    ctx_size,
                ));
            }
            Commands::Serve { host, port } => {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(serve_api::start_server(
                    host,
                    port,
                    db.clone(),
                    system_specs.clone(),
                ));
            }
            Commands::Cluster { action } => {
                match ClusterSpecs::from_saved() {
                    Ok(mut cluster_specs) => {
                        let config_path = ClusterSpecs::get_config_path().unwrap();
                        match action {
                            ClusterAction::Add { node_url } => {
                                println!("Adding node {}...", node_url);
                                match cluster_specs.add_node(&node_url) {
                                    Ok(_) => {
                                        cluster_specs.save().unwrap();
                                        println!("Node added. Current cluster size: {} nodes. See details at {}", cluster_specs.nodes.len(), config_path.display());
                                    }
                                    Err(e) => eprintln!("Error adding node: {}", e),
                                }
                            }
                            ClusterAction::Remove { node_url } => {
                                println!("Removing node {}...", node_url);
                                if cluster_specs.remove_node(&node_url) {
                                    cluster_specs.save().unwrap();
                                    println!("Node removed. Current cluster size: {} nodes. See details at {}", cluster_specs.nodes.len(), config_path.display());
                                } else {
                                    eprintln!("Node not found in cluster: {}", node_url);
                                }
                            }
                            ClusterAction::List => {
                                println!("{}", cluster_specs);
                            }
                            ClusterAction::Clear => {
                                if let Err(e) = std::fs::remove_file(&config_path) {
                                    eprintln!("Error clearing cluster configuration: {}", e);
                                } else {
                                    println!("Cluster configuration cleared from {}", config_path.display());
                                }
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Error loading cluster specs: {}", e);
                        exit(1);
                    }
                }
            }
        }
    } else {
        let models = get_model_fits(
            &db,
            &system_specs,
            None,
            cli.limit,
            cli.sort.into(),
            if cli.perfect { Some(FitArg::Perfect) } else { None },
        );
        if cli.cli {
            display::print_model_fits(&system_specs, &models, false, false);
        } else {
            let mut tui_app = tui_app::TuiApp::new(db, models, system_specs);
            tui_app.run().unwrap();
        }
    }
}
