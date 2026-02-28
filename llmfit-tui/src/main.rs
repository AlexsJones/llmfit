mod display;
mod theme;
mod tui_app;
mod tui_events;
mod tui_ui;

use clap::{Parser, Subcommand};
use llmfit_core::fit::{ModelFit, backend_compatible};
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

    /// Cap context length used for memory estimation (tokens).
    /// Falls back to OLLAMA_CONTEXT_LENGTH if not set.
    #[arg(long, value_name = "TOKENS", value_parser = clap::value_parser!(u32).range(1..))]
    max_context: Option<u32>,
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

    /// Download a GGUF model from HuggingFace for use with llama.cpp
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
    HfSearch {
        /// Search query (model name, architecture, etc.)
        query: String,

        /// Maximum number of results
        #[arg(short = 'n', long, default_value = "10")]
        limit: usize,
    },

    /// Run a downloaded GGUF model with llama-cli or llama-server
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

fn run_fit(
    perfect: bool,
    limit: Option<usize>,
    json: bool,
    memory_override: &Option<String>,
    context_limit: Option<u32>,
) {
    let specs = detect_specs(memory_override);
    let db = ModelDatabase::new();

    if !json {
        specs.display();
    }

    let hidden: usize = db
        .get_all_models()
        .iter()
        .filter(|m| !backend_compatible(m, &specs))
        .count();

    let mut fits: Vec<ModelFit> = db
        .get_all_models()
        .iter()
        .filter(|m| backend_compatible(m, &specs))
        .map(|m| ModelFit::analyze_with_context_limit(m, &specs, context_limit))
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

fn run_tui(memory_override: &Option<String>, context_limit: Option<u32>) -> std::io::Result<()> {
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
    let specs = detect_specs(memory_override);
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
    json: bool,
    memory_override: &Option<String>,
    context_limit: Option<u32>,
) {
    let specs = detect_specs(memory_override);
    let db = ModelDatabase::new();

    let mut fits: Vec<ModelFit> = db
        .get_all_models()
        .iter()
        .filter(|m| backend_compatible(m, &specs))
        .map(|m| ModelFit::analyze_with_context_limit(m, &specs, context_limit))
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

fn run_download(
    model: &str,
    quant: Option<&str>,
    budget: Option<f64>,
    list_only: bool,
    memory_override: &Option<String>,
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
            let specs = detect_specs(memory_override);
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

    println!(
        "\nDownloading {} ({:.1} GB) to {}",
        filename,
        file_size as f64 / 1_073_741_824.0,
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
                        let dest = provider.models_dir().join(&filename);
                        println!("  Saved to: {}", dest.display());
                        if provider.llama_cli_path().is_some() {
                            println!(
                                "\n  Run with: llmfit run {}",
                                filename.trim_end_matches(".gguf")
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

    println!("{:<50} {}", "Repository", "Type");
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

fn main() {
    let cli = Cli::parse();
    let context_limit = resolve_context_limit(cli.max_context);

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
                run_fit(perfect, limit, cli.json, &cli.memory, context_limit);
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

                let fit = ModelFit::analyze_with_context_limit(results[0], &specs, context_limit);
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
                run_recommend(
                    limit,
                    use_case,
                    min_fit,
                    runtime,
                    json,
                    &cli.memory,
                    context_limit,
                );
            }

            Commands::Download {
                model,
                quant,
                budget,
                list,
            } => {
                run_download(&model, quant.as_deref(), budget, list, &cli.memory);
            }

            Commands::HfSearch { query, limit } => {
                run_hf_search(&query, limit);
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
        }
        return;
    }

    // If --cli flag, use classic fit output
    if cli.cli {
        run_fit(cli.perfect, cli.limit, cli.json, &cli.memory, context_limit);
        return;
    }

    // Default: launch TUI
    if let Err(e) = run_tui(&cli.memory, context_limit) {
        eprintln!("Error running TUI: {}", e);
        std::process::exit(1);
    }
}
