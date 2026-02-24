mod display;
mod theme;
mod tui_app;
mod tui_events;
mod tui_ui;

use clap::{Parser, Subcommand};
use llmfit_core::fit::ModelFit;
use llmfit_core::hardware::SystemSpecs;
use llmfit_core::models::ModelDatabase;

#[derive(Parser)]
#[command(name = "llmfit")]
#[command(about = "Right-size LLM models to your system's hardware", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Show only models that perfectly match recommended specs
    #[arg(short, long)]
    perfect: bool,

    /// Limit number of results
    #[arg(short = 'n', long)]
    limit: Option<usize>,

    /// Use classic CLI table output instead of TUI
    #[arg(long)]
    cli: bool,

    /// Output results as JSON (for tool integration)
    #[arg(long)]
    json: bool,

    /// Override GPU VRAM size (e.g. "32G", "32000M", "1.5T").
    /// Useful when GPU memory autodetection fails.
    #[arg(long, value_name = "SIZE")]
    memory: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Show system hardware specifications
    System,

    /// List all available LLM models
    List,

    /// Find models that fit your system (classic table output)
    Fit {
        /// Show only models that perfectly match recommended specs
        #[arg(short, long)]
        perfect: bool,

        /// Limit number of results
        #[arg(short = 'n', long)]
        limit: Option<usize>,
    },

    /// Search for specific models
    Search {
        /// Search query (model name, provider, or size)
        query: String,
    },

    /// Show detailed information about a specific model
    Info {
        /// Model name or partial name to look up
        model: String,
    },

    /// Recommend top models for your hardware (JSON-friendly)
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

        /// Output as JSON (default for recommend)
        #[arg(long, default_value = "true")]
        json: bool,
    },
}

/// Detect system specs with optional GPU memory override.
fn detect_specs(memory_override: &Option<String>) -> SystemSpecs {
    let specs = SystemSpecs::detect();
    if let Some(mem_str) = memory_override {
        match llmfit_core::hardware::parse_memory_size(mem_str) {
            Some(gb) => specs.with_gpu_memory_override(gb),
            None => {
                eprintln!(
                    "Warning: could not parse --memory value '{}'. Expected format: 32G, 32000M, 1.5T",
                    mem_str
                );
                specs
            }
        }
    } else {
        specs
    }
}

fn run_fit(perfect: bool, limit: Option<usize>, json: bool, memory_override: &Option<String>) {
    let specs = detect_specs(memory_override);
    let db = ModelDatabase::new();

    if !json {
        specs.display();
    }

    let mut fits: Vec<ModelFit> = db
        .get_all_models()
        .iter()
        .map(|m| ModelFit::analyze(m, &specs))
        .collect();

    if perfect {
        fits.retain(|f| f.fit_level == llmfit_core::fit::FitLevel::Perfect);
    }

    fits = llmfit_core::fit::rank_models_by_fit(fits);

    if let Some(n) = limit {
        fits.truncate(n);
    }

    if json {
        display::display_json_fits(&specs, &fits);
    } else {
        display::display_model_fits(&fits);
    }
}

fn run_tui(memory_override: &Option<String>) -> std::io::Result<()> {
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

    // Create app state
    let specs = detect_specs(memory_override);
    let mut app = tui_app::App::with_specs(specs);

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

fn run_recommend(
    limit: usize,
    use_case: Option<String>,
    min_fit: String,
    runtime_filter: String,
    json: bool,
    memory_override: &Option<String>,
) {
    let specs = detect_specs(memory_override);
    let db = ModelDatabase::new();

    let mut fits: Vec<ModelFit> = db
        .get_all_models()
        .iter()
        .map(|m| ModelFit::analyze(m, &specs))
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

    // Filter by runtime
    match runtime_filter.to_lowercase().as_str() {
        "mlx" => fits.retain(|f| f.runtime == llmfit_core::fit::InferenceRuntime::Mlx),
        "llamacpp" | "llama.cpp" | "llama_cpp" => {
            fits.retain(|f| f.runtime == llmfit_core::fit::InferenceRuntime::LlamaCpp)
        }
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

    fits = llmfit_core::fit::rank_models_by_fit(fits);
    fits.truncate(limit);

    if json {
        display::display_json_fits(&specs, &fits);
    } else {
        if !fits.is_empty() {
            specs.display();
        }
        display::display_model_fits(&fits);
    }
}

fn main() {
    let cli = Cli::parse();

    // If a subcommand is given, use classic CLI mode
    if let Some(command) = cli.command {
        match command {
            Commands::System => {
                let specs = detect_specs(&cli.memory);
                if cli.json {
                    display::display_json_system(&specs);
                } else {
                    specs.display();
                }
            }

            Commands::List => {
                let db = ModelDatabase::new();
                display::display_all_models(db.get_all_models());
            }

            Commands::Fit { perfect, limit } => {
                run_fit(perfect, limit, cli.json, &cli.memory);
            }

            Commands::Search { query } => {
                let db = ModelDatabase::new();
                let results = db.find_model(&query);
                display::display_search_results(&results, &query);
            }

            Commands::Info { model } => {
                let db = ModelDatabase::new();
                let specs = detect_specs(&cli.memory);
                let results = db.find_model(&model);

                if results.is_empty() {
                    println!("\nNo model found matching '{}'", model);
                    return;
                }

                if results.len() > 1 {
                    println!("\nMultiple models found. Please be more specific:");
                    for m in results {
                        println!("  - {}", m.name);
                    }
                    return;
                }

                let fit = ModelFit::analyze(results[0], &specs);
                if cli.json {
                    display::display_json_fits(&specs, &[fit]);
                } else {
                    display::display_model_detail(&fit);
                }
            }

            Commands::Recommend {
                limit,
                use_case,
                min_fit,
                runtime,
                json,
            } => {
                run_recommend(limit, use_case, min_fit, runtime, json, &cli.memory);
            }
        }
        return;
    }

    // If --cli flag, use classic fit output
    if cli.cli {
        run_fit(cli.perfect, cli.limit, cli.json, &cli.memory);
        return;
    }

    // Default: launch TUI
    if let Err(e) = run_tui(&cli.memory) {
        eprintln!("Error running TUI: {}", e);
        std::process::exit(1);
    }
}
