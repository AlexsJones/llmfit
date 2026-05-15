use std::path::PathBuf;
use std::sync::OnceLock;

use colored::*;
use llmfit_core::fit::{FitLevel, ModelFit, RunMode, SortColumn};
use llmfit_core::hardware::SystemSpecs;
use llmfit_core::models::LlmModel;
use llmfit_core::plan::PlanEstimate;
use tabled::{Table, Tabled, settings::Style};

#[derive(Tabled)]
struct ModelRow {
    #[tabled(rename = "Status")]
    status: String,
    #[tabled(rename = "Model")]
    name: String,
    #[tabled(rename = "Provider")]
    provider: String,
    #[tabled(rename = "Size")]
    size: String,
    #[tabled(rename = "Score")]
    score: String,
    #[tabled(rename = "tok/s est.")]
    tps: String,
    #[tabled(rename = "Quant")]
    quant: String,
    #[tabled(rename = "Runtime")]
    runtime: String,
    #[tabled(rename = "Mode")]
    mode: String,
    #[tabled(rename = "Mem %")]
    mem_use: String,
    #[tabled(rename = "Context")]
    context: String,
    #[tabled(rename = "Added to HF")]
    release_date: String,
}

pub fn display_all_models(models: &[LlmModel], sort: SortColumn) {
    let mut models: Vec<&LlmModel> = models.iter().collect();
    match sort {
        SortColumn::ReleaseDate => {
            models.sort_by(|a, b| {
                b.release_date
                    .as_deref()
                    .unwrap_or("")
                    .cmp(a.release_date.as_deref().unwrap_or(""))
            });
        }
        SortColumn::Params => {
            models.sort_by(|a, b| {
                b.parameters_raw
                    .unwrap_or(0)
                    .cmp(&a.parameters_raw.unwrap_or(0))
            });
        }
        SortColumn::Ctx => {
            models.sort_by(|a, b| b.context_length.cmp(&a.context_length));
        }
        SortColumn::MemPct => {
            models.sort_by(|a, b| {
                let a_mem = a.min_vram_gb.unwrap_or(a.min_ram_gb);
                let b_mem = b.min_vram_gb.unwrap_or(b.min_ram_gb);
                b_mem
                    .partial_cmp(&a_mem)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        _ => {}
    }
    println!("\n{}", "=== Available LLM Models ===".bold().cyan());
    println!(
        "Total models: {} (sorted by: {})\n",
        models.len(),
        sort.label()
    );

    let rows: Vec<ModelRow> = models
        .iter()
        .map(|m| ModelRow {
            status: "--".to_string(),
            name: m.name.clone(),
            provider: m.provider.clone(),
            size: m.parameter_count.clone(),
            score: "-".to_string(),
            tps: "-".to_string(),
            quant: m.quantization.clone(),
            runtime: "-".to_string(),
            mode: "-".to_string(),
            mem_use: "-".to_string(),
            context: format!("{}k", m.context_length / 1000),
            release_date: m
                .release_date
                .clone()
                .unwrap_or_else(|| "\u{2014}".to_string()),
        })
        .collect();

    let table = Table::new(rows).with(Style::rounded()).to_string();
    println!("{}", table);
}

pub fn display_model_fits(fits: &[ModelFit]) {
    if fits.is_empty() {
        println!(
            "\n{}",
            "No compatible models found for your system.".yellow()
        );
        return;
    }

    println!("\n{}", "=== Model Compatibility Analysis ===".bold().cyan());
    println!("Found {} compatible model(s)\n", fits.len());

    let rows: Vec<ModelRow> = fits
        .iter()
        .map(|fit| {
            let status_prefix = if fit.installed { "✓ " } else { "" };
            let status_text = format!("{}{} {}", status_prefix, fit.fit_emoji(), fit.fit_text());

            ModelRow {
                status: status_text,
                name: fit.model.name.clone(),
                provider: fit.model.provider.clone(),
                size: fit.model.parameter_count.clone(),
                score: format!("{:.0}", fit.score),
                tps: format!("{:.1}", fit.estimated_tps),
                quant: fit.best_quant.clone(),
                runtime: fit.runtime_text().to_string(),
                mode: fit.run_mode_text().to_string(),
                mem_use: format!("{:.1}%", fit.utilization_pct),
                context: format!("{}k", fit.model.context_length / 1000),
                release_date: fit
                    .model
                    .release_date
                    .clone()
                    .unwrap_or_else(|| "\u{2014}".to_string()),
            }
        })
        .collect();

    let table = Table::new(rows).with(Style::rounded()).to_string();
    println!("{}", table);
    println!(
        "  Note: tok/s values are baseline estimates; real runtime depends on engine/runtime."
    );
}

pub fn display_model_detail(fit: &ModelFit) {
    println!("\n{}", format!("=== {} ===", fit.model.name).bold().cyan());
    println!();
    println!("{}: {}", "Provider".bold(), fit.model.provider);
    println!("{}: {}", "Parameters".bold(), fit.model.parameter_count);
    println!("{}: {}", "Quantization".bold(), fit.model.quantization);
    println!("{}: {}", "Best Quant".bold(), fit.best_quant);
    println!(
        "{}: {} tokens",
        "Context Length".bold(),
        fit.model.context_length
    );
    println!("{}: {}", "Use Case".bold(), fit.model.use_case);
    println!("{}: {}", "Category".bold(), fit.use_case.label());
    if let Some(ref date) = fit.model.release_date {
        println!("{}: {}", "Released".bold(), date);
    }
    println!(
        "{}: {}",
        "License".bold(),
        fit.model.license.as_deref().unwrap_or("Unknown")
    );
    println!(
        "{}: {} (baseline est. ~{:.1} tok/s)",
        "Runtime".bold(),
        fit.runtime_text(),
        fit.estimated_tps
    );
    println!();

    println!("{}", "Score Breakdown:".bold().underline());
    println!("  Overall Score: {:.1} / 100", fit.score);
    println!(
        "  Quality: {:.0}  Speed: {:.0}  Fit: {:.0}  Context: {:.0}",
        fit.score_components.quality,
        fit.score_components.speed,
        fit.score_components.fit,
        fit.score_components.context
    );
    println!("  Baseline Est. Speed: {:.1} tok/s", fit.estimated_tps);
    println!();

    println!("{}", "Resource Requirements:".bold().underline());
    if let Some(vram) = fit.model.min_vram_gb {
        println!("  Min VRAM: {:.1} GB", vram);
    }
    println!("  Min RAM: {:.1} GB (CPU inference)", fit.model.min_ram_gb);
    println!("  Recommended RAM: {:.1} GB", fit.model.recommended_ram_gb);
    println!(
        "  Disk (est): {:.1} GB (at {})",
        fit.model.estimate_disk_gb(&fit.best_quant),
        fit.best_quant
    );
    let quants: &[&str] = if fit.best_quant.starts_with("mlx") {
        &["mlx-8bit", "mlx-4bit"]
    } else {
        &["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
    };
    let breakdown: Vec<String> = quants
        .iter()
        .map(|q| format!("{}: {:.1}G", q, fit.model.estimate_disk_gb(q)))
        .collect();
    println!("  Disk/quant: {}", breakdown.join("  "));

    // MoE Architecture info
    if fit.model.is_moe {
        println!();
        println!("{}", "MoE Architecture:".bold().underline());
        if let (Some(num_experts), Some(active_experts)) =
            (fit.model.num_experts, fit.model.active_experts)
        {
            println!(
                "  Experts: {} active / {} total per token",
                active_experts, num_experts
            );
        }
        if let Some(active_vram) = fit.model.moe_active_vram_gb() {
            println!(
                "  Active VRAM: {:.1} GB (vs {:.1} GB full model)",
                active_vram,
                fit.model.min_vram_gb.unwrap_or(0.0)
            );
        }
        if let Some(offloaded) = fit.moe_offloaded_gb {
            println!("  Offloaded: {:.1} GB inactive experts in RAM", offloaded);
        }
    }
    println!();

    println!("{}", "Fit Analysis:".bold().underline());

    let fit_color = match fit.fit_level {
        FitLevel::Perfect => "green",
        FitLevel::Good => "yellow",
        FitLevel::Marginal => "orange",
        FitLevel::TooTight => "red",
    };

    println!(
        "  Status: {} {}",
        fit.fit_emoji(),
        fit.fit_text().color(fit_color)
    );
    println!("  Run Mode: {}", fit.run_mode_text());
    println!(
        "  Memory Utilization: {:.1}% ({:.1} / {:.1} GB)",
        fit.utilization_pct, fit.memory_required_gb, fit.memory_available_gb
    );
    println!();

    if !fit.model.gguf_sources.is_empty() {
        println!("{}", "GGUF Downloads:".bold().underline());
        for src in &fit.model.gguf_sources {
            println!("  {} → https://huggingface.co/{}", src.provider, src.repo);
        }
        println!(
            "  {}",
            format!(
                "Tip: llmfit download {} --quant {}",
                fit.model.gguf_sources[0].repo, fit.best_quant
            )
            .dimmed()
        );
        println!();
    }

    if !fit.notes.is_empty() {
        println!("{}", "Notes:".bold().underline());
        for note in &fit.notes {
            println!("  {}", note);
        }
        println!();
    }
}

pub fn display_model_diff(fits: &[ModelFit], sort_label: &str) {
    if fits.len() < 2 {
        println!("\n{}", "Need at least 2 models to compare.".yellow());
        return;
    }

    println!("\n{}", "=== Model Diff ===".bold().cyan());
    println!(
        "Comparing {} model(s) (sorted by {})\n",
        fits.len(),
        sort_label
    );

    let metric_width = 20usize;
    let col_width = 32usize;

    let model_headers: Vec<String> = fits
        .iter()
        .enumerate()
        .map(|(i, fit)| {
            let label = format!("M{}: {}", i + 1, fit.model.name);
            truncate_to_width(&label, col_width)
        })
        .collect();

    print!("{:<metric_width$}", "Metric".bold());
    for header in &model_headers {
        print!("  {:<col_width$}", header.bold());
    }
    println!();

    print!("{:-<metric_width$}", "");
    for _ in &model_headers {
        print!("  {:-<col_width$}", "");
    }
    println!();

    let base = &fits[0];

    print_metric_row(
        "Score",
        fits.iter()
            .map(|f| format_with_delta(format!("{:.1}", f.score), f.score - base.score))
            .collect(),
        metric_width,
        col_width,
    );
    print_metric_row(
        "Baseline tok/s",
        fits.iter()
            .map(|f| {
                format_with_delta(
                    format!("{:.1}", f.estimated_tps),
                    f.estimated_tps - base.estimated_tps,
                )
            })
            .collect(),
        metric_width,
        col_width,
    );
    print_metric_row(
        "Fit",
        fits.iter()
            .map(|f| format!("{} {}", f.fit_emoji(), f.fit_text()))
            .collect(),
        metric_width,
        col_width,
    );
    print_metric_row(
        "Run Mode",
        fits.iter().map(|f| f.run_mode_text().to_string()).collect(),
        metric_width,
        col_width,
    );
    print_metric_row(
        "Runtime",
        fits.iter().map(|f| f.runtime_text().to_string()).collect(),
        metric_width,
        col_width,
    );
    print_metric_row(
        "Memory %",
        fits.iter()
            .map(|f| {
                format_with_delta(
                    format!("{:.1}%", f.utilization_pct),
                    f.utilization_pct - base.utilization_pct,
                )
            })
            .collect(),
        metric_width,
        col_width,
    );
    print_metric_row(
        "Params",
        fits.iter()
            .map(|f| f.model.parameter_count.clone())
            .collect(),
        metric_width,
        col_width,
    );
    print_metric_row(
        "Context",
        fits.iter()
            .map(|f| format!("{} tokens", f.model.context_length))
            .collect(),
        metric_width,
        col_width,
    );
    print_metric_row(
        "Best Quant",
        fits.iter().map(|f| f.best_quant.clone()).collect(),
        metric_width,
        col_width,
    );
    print_metric_row(
        "Provider",
        fits.iter().map(|f| f.model.provider.clone()).collect(),
        metric_width,
        col_width,
    );
}

fn print_metric_row(metric: &str, values: Vec<String>, metric_width: usize, col_width: usize) {
    print!("{:<metric_width$}", metric);
    for value in values {
        print!("  {:<col_width$}", truncate_to_width(&value, col_width));
    }
    println!();
}

fn format_with_delta(value: String, delta: f64) -> String {
    if delta.abs() < 0.05 {
        return value;
    }
    format!("{} ({:+.1})", value, delta)
}

fn truncate_to_width(input: &str, width: usize) -> String {
    if input.chars().count() <= width {
        return input.to_string();
    }
    let mut out = input
        .chars()
        .take(width.saturating_sub(3))
        .collect::<String>();
    out.push_str("...");
    out
}

pub fn display_search_results(models: &[&LlmModel], query: &str) {
    if models.is_empty() {
        println!(
            "\n{}",
            format!("No models found matching '{}'", query).yellow()
        );
        return;
    }

    println!(
        "\n{}",
        format!("=== Search Results for '{}' ===", query)
            .bold()
            .cyan()
    );
    println!("Found {} model(s)\n", models.len());

    let rows: Vec<ModelRow> = models
        .iter()
        .map(|m| ModelRow {
            status: "--".to_string(),
            name: m.name.clone(),
            provider: m.provider.clone(),
            size: m.parameter_count.clone(),
            score: "-".to_string(),
            tps: "-".to_string(),
            quant: m.quantization.clone(),
            runtime: "-".to_string(),
            mode: "-".to_string(),
            mem_use: "-".to_string(),
            context: format!("{}k", m.context_length / 1000),
            release_date: m
                .release_date
                .clone()
                .unwrap_or_else(|| "\u{2014}".to_string()),
        })
        .collect();

    let table = Table::new(rows).with(Style::rounded()).to_string();
    println!("{}", table);
}

// ────────────────────────────────────────────────────────────────────
// JSON output for machine consumption (OpenClaw skills, scripts, etc.)
// ────────────────────────────────────────────────────────────────────

/// Serialize system specs to JSON and print to stdout.
pub fn display_json_system(specs: &SystemSpecs) {
    let output = serde_json::json!({
        "system": system_json(specs),
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&output).expect("JSON serialization failed")
    );
}

/// Serialize system specs + model fits to JSON and print to stdout.
pub fn display_json_fits(specs: &SystemSpecs, fits: &[ModelFit]) {
    let models: Vec<serde_json::Value> = fits.iter().map(fit_to_json).collect();
    let output = serde_json::json!({
        "system": system_json(specs),
        "models": models,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&output).expect("JSON serialization failed")
    );
}

/// Serialize system specs + model fits to JSON with llama.cpp commands and print to stdout.
pub fn display_json_fits_with_llamacpp(specs: &SystemSpecs, fits: &[ModelFit]) {
    use llmfit_core::fit::InferenceRuntime;

    let models: Vec<serde_json::Value> = fits
        .iter()
        .map(|fit| {
            let mut json = fit_to_json(fit);

            // Add suggested llama.cpp command for llama.cpp-compatible models
            if fit.runtime == InferenceRuntime::LlamaCpp
                && let Some(cmd) = generate_llamacpp_command(fit)
            {
                json.as_object_mut().unwrap().insert(
                    "llamacpp_command".to_string(),
                    serde_json::Value::String(cmd),
                );
            }

            json
        })
        .collect();

    let output = serde_json::json!({
        "system": system_json(specs),
        "models": models,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&output).expect("JSON serialization failed")
    );
}

/// Generate a llama.cpp command string for a model fit.
fn generate_llamacpp_command(fit: &ModelFit) -> Option<String> {
    if fit.run_mode == RunMode::TensorParallel {
        return None;
    }

    // Get the GGUF source repo if available
    let repo = fit.model.gguf_sources.first().map(|s| &s.repo);

    let quant = &fit.best_quant;
    let context = fit.effective_context_length;
    let ngl_args = llamacpp_ngl_args(fit.run_mode)?;
    let conversation_arg = if should_use_llamacpp_conversation_mode(fit) {
        " -cnv"
    } else {
        ""
    };

    // Use the -hf option with HuggingFace repo to let llama-cli handle model
    // downloading/caching. This avoids path guessing issues and works for both
    // installed and non-installed models. llama-cli automatically downloads
    // to its own cache if the model isn't present locally.
    repo.map(|repo| {
        format!(
            "llama-cli -hf {}:{} {} -c {}{}",
            repo, quant, ngl_args, context, conversation_arg
        )
    })
}

fn llamacpp_ngl_args(run_mode: RunMode) -> Option<&'static str> {
    match run_mode {
        RunMode::CpuOffload | RunMode::MoeOffload => {
            llamacpp_ngl_args_for_support(run_mode, llamacpp_supports_fit_arg())
        }
        _ => llamacpp_ngl_args_for_support(run_mode, false),
    }
}

fn llamacpp_ngl_args_for_support(
    run_mode: RunMode,
    supports_fit_arg: bool,
) -> Option<&'static str> {
    match run_mode {
        RunMode::Gpu => Some("-ngl all"),
        RunMode::CpuOnly => Some("-ngl 0"),
        RunMode::CpuOffload => Some(if supports_fit_arg {
            "-ngl auto --fit on"
        } else {
            "-ngl auto"
        }),
        // llmfit's MoE estimate is not an exact llama.cpp layer/MoE split, so
        // prefer llama.cpp's fit-aware auto offload rather than forcing --cpu-moe.
        RunMode::MoeOffload => Some(if supports_fit_arg {
            "-ngl auto --fit on"
        } else {
            "-ngl auto"
        }),
        RunMode::TensorParallel => None,
    }
}

fn llamacpp_supports_fit_arg() -> bool {
    static SUPPORTS_FIT_ARG: OnceLock<bool> = OnceLock::new();

    *SUPPORTS_FIT_ARG.get_or_init(|| {
        let candidate = llamacpp_binary_arg();
        let Ok(output) = std::process::Command::new(&candidate)
            .arg("--help")
            .output()
        else {
            return false;
        };

        String::from_utf8_lossy(&output.stdout).contains("--fit")
            || String::from_utf8_lossy(&output.stderr).contains("--fit")
    })
}

fn llamacpp_binary_arg() -> PathBuf {
    let name = format!("llama-cli{}", std::env::consts::EXE_SUFFIX);
    if let Ok(dir) = std::env::var("LLAMA_CPP_PATH") {
        let candidate = PathBuf::from(dir).join(&name);
        if candidate.is_file() {
            return candidate;
        }
    }
    PathBuf::from(name)
}

/// Best-effort heuristic: checks use_case and name substrings to guess whether
/// a model is conversational. May misfire on edge cases (e.g. an embedding model
/// named "multichat", or a general-purpose model named "functionary").
fn should_use_llamacpp_conversation_mode(fit: &ModelFit) -> bool {
    use llmfit_core::models::{Capability, UseCase};

    let name = fit.model.name.to_lowercase();
    let use_case = fit.model.use_case.to_lowercase();

    if fit.use_case == UseCase::Embedding
        || use_case.contains("embedding")
        || name.contains("embed")
        || name.contains("bge")
    {
        return false;
    }

    fit.use_case == UseCase::Chat
        || fit.model.capabilities.contains(&Capability::ToolUse)
        || use_case.contains("chat")
        || use_case.contains("instruct")
        || use_case.contains("instruction")
        || use_case.contains("tool")
        || use_case.contains("function")
        || name.contains("chat")
        || name.contains("instruct")
        || name.contains("-it")
}

/// Serialize diff output via serde derives (new diff-only path).
pub fn display_json_diff_fits(specs: &SystemSpecs, fits: &[ModelFit]) {
    #[derive(serde::Serialize)]
    struct FitsOutput<'a> {
        system: &'a SystemSpecs,
        models: &'a [ModelFit],
    }
    let output = FitsOutput {
        system: specs,
        models: fits,
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&output).expect("JSON serialization failed")
    );
}

fn system_json(specs: &SystemSpecs) -> serde_json::Value {
    let gpus_json: Vec<serde_json::Value> = specs
        .gpus
        .iter()
        .map(|g| {
            serde_json::json!({
                "name": g.name,
                "vram_gb": g.vram_gb.map(round2),
                "backend": g.backend.label(),
                "count": g.count,
                "unified_memory": g.unified_memory,
            })
        })
        .collect();

    serde_json::json!({
        "total_ram_gb": round2(specs.total_ram_gb),
        "available_ram_gb": round2(specs.available_ram_gb),
        "cpu_cores": specs.total_cpu_cores,
        "cpu_name": specs.cpu_name,
        "has_gpu": specs.has_gpu,
        "gpu_vram_gb": specs.gpu_vram_gb.map(round2),
        "gpu_name": specs.gpu_name,
        "gpu_count": specs.gpu_count,
        "unified_memory": specs.unified_memory,
        "backend": specs.backend.label(),
        "gpus": gpus_json,
    })
}

fn fit_to_json(fit: &ModelFit) -> serde_json::Value {
    serde_json::json!({
        "name": fit.model.name,
        "provider": fit.model.provider,
        "parameter_count": fit.model.parameter_count,
        "params_b": round2(fit.model.params_b()),
        "context_length": fit.model.context_length,
        "effective_context_length": fit.effective_context_length,
        "use_case": fit.model.use_case,
        "category": fit.use_case.label(),
        "release_date": fit.model.release_date,
        "license": fit.model.license,
        "is_moe": fit.model.is_moe,
        "fit_level": fit.fit_text(),
        "run_mode": fit.run_mode_text(),
        "score": round1(fit.score),
        "score_components": {
            "quality": round1(fit.score_components.quality),
            "speed": round1(fit.score_components.speed),
            "fit": round1(fit.score_components.fit),
            "context": round1(fit.score_components.context),
        },
        "estimated_tps": round1(fit.estimated_tps),
        "runtime": fit.runtime_text(),
        "runtime_label": fit.runtime.label(),
        "best_quant": fit.best_quant,
        "disk_size_gb": round2(fit.model.estimate_disk_gb(&fit.best_quant)),
        "memory_required_gb": round2(fit.memory_required_gb),
        "memory_available_gb": round2(fit.memory_available_gb),
        "moe_offloaded_gb": fit.moe_offloaded_gb.map(round2),
        "total_memory_gb": round2(fit.memory_required_gb + fit.moe_offloaded_gb.unwrap_or(0.0)),
        "utilization_pct": round1(fit.utilization_pct),
        "notes": fit.notes,
        "gguf_sources": fit.model.gguf_sources,
        "installed": fit.installed,
        "capabilities": fit.model.capabilities.iter().map(|c| c.label()).collect::<Vec<_>>(),
        "capability_ids": serde_json::to_value(&fit.model.capabilities).unwrap(),
    })
}

fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

pub fn display_model_plan(plan: &PlanEstimate) {
    println!("\n{}", "=== Hardware Planning Estimate ===".bold().cyan());
    println!("{} {}", "Model:".bold(), plan.model_name);
    println!("{} {}", "Provider:".bold(), plan.provider);
    println!("{} {}", "Context:".bold(), plan.context);
    println!("{} {}", "Quantization:".bold(), plan.quantization);
    println!("{} {}", "KV cache:".bold(), plan.kv_quant.label());
    if let Some(tps) = plan.target_tps {
        println!("{} {:.1} tok/s", "Target TPS:".bold(), tps);
    }
    println!("{} {}", "Note:".bold(), plan.estimate_notice);
    println!();

    println!("{}", "Minimum Hardware:".bold().underline());
    println!(
        "  VRAM: {}",
        plan.minimum
            .vram_gb
            .map(|v| format!("{v:.1} GB"))
            .unwrap_or_else(|| "Not required".to_string())
    );
    println!("  RAM: {:.1} GB", plan.minimum.ram_gb);
    println!("  CPU Cores: {}", plan.minimum.cpu_cores);
    println!();

    println!("{}", "Recommended Hardware:".bold().underline());
    println!(
        "  VRAM: {}",
        plan.recommended
            .vram_gb
            .map(|v| format!("{v:.1} GB"))
            .unwrap_or_else(|| "Not required".to_string())
    );
    println!("  RAM: {:.1} GB", plan.recommended.ram_gb);
    println!("  CPU Cores: {}", plan.recommended.cpu_cores);
    println!();

    println!("{}", "Feasible Run Paths:".bold().underline());
    for path in &plan.run_paths {
        println!(
            "  {}: {}",
            path.path.label(),
            if path.feasible { "Yes" } else { "No" }
        );
        if let Some(min) = &path.minimum {
            println!(
                "    min: VRAM={} RAM={:.1} GB cores={}",
                min.vram_gb
                    .map(|v| format!("{v:.1} GB"))
                    .unwrap_or_else(|| "n/a".to_string()),
                min.ram_gb,
                min.cpu_cores
            );
        }
        if let Some(tps) = path.estimated_tps {
            println!("    est speed: {:.1} tok/s", tps);
        }
    }
    println!();

    println!("{}", "Upgrade Deltas:".bold().underline());
    if plan.upgrade_deltas.is_empty() {
        println!("  None required for the selected target.");
    } else {
        for delta in &plan.upgrade_deltas {
            println!("  {}", delta.description);
        }
    }
    println!();

    if !plan.kv_alternatives.is_empty() {
        println!("{}", "KV Cache Alternatives:".bold().underline());
        println!(
            "  {:<8} {:>10} {:>10} {:>10}  notes",
            "kv", "kv (GB)", "total", "savings"
        );
        for alt in &plan.kv_alternatives {
            let label = if alt.supported {
                alt.kv_quant.label().to_string()
            } else {
                format!("{} (n/a)", alt.kv_quant.label())
            };
            let savings_str = if alt.savings_fraction > 0.0 {
                format!("-{:.0}%", alt.savings_fraction * 100.0)
            } else {
                "-".to_string()
            };
            let note = alt.note.as_deref().unwrap_or("");
            println!(
                "  {:<8} {:>10.2} {:>10.2} {:>10}  {}",
                label, alt.kv_cache_gb, alt.memory_required_gb, savings_str, note
            );
        }
        println!();
    }
}

pub fn display_json_plan(plan: &PlanEstimate) {
    println!(
        "{}",
        serde_json::to_string_pretty(plan).expect("JSON serialization failed")
    );
}

// ────────────────────────────────────────────────────────────────────
// CSV export for spreadsheet / data analysis
// ────────────────────────────────────────────────────────────────────

/// Flat row struct for CSV serialization. Numerical fields are raw f64
/// values (no units/percent signs) for easy import into spreadsheets.
#[derive(serde::Serialize)]
struct CsvFitRow {
    name: String,
    provider: String,
    parameter_count: String,
    params_billion: f64,
    context_length: u32,
    fit_level: String,
    run_mode: String,
    score: f64,
    score_quality: f64,
    score_speed: f64,
    score_fit: f64,
    score_context: f64,
    estimated_tps: f64,
    memory_required_gb: f64,
    memory_available_gb: f64,
    utilization_pct: f64,
    disk_size_gb: f64,
    best_quant: String,
    runtime: String,
    use_case: String,
    release_date: Option<String>,
    license: Option<String>,
    is_moe: bool,
    installed: bool,
}

/// Serialize model fits as CSV to stdout.
pub fn display_csv_fits(fits: &[ModelFit]) {
    let mut writer = csv::Writer::from_writer(std::io::stdout());

    for fit in fits {
        writer
            .serialize(CsvFitRow {
                name: fit.model.name.clone(),
                provider: fit.model.provider.clone(),
                parameter_count: fit.model.parameter_count.clone(),
                params_billion: round2(fit.model.params_b()),
                context_length: fit.model.context_length,
                fit_level: fit.fit_text().to_lowercase(),
                run_mode: fit.run_mode_text().to_lowercase(),
                score: round1(fit.score),
                score_quality: round1(fit.score_components.quality),
                score_speed: round1(fit.score_components.speed),
                score_fit: round1(fit.score_components.fit),
                score_context: round1(fit.score_components.context),
                estimated_tps: round1(fit.estimated_tps),
                memory_required_gb: round2(fit.memory_required_gb),
                memory_available_gb: round2(fit.memory_available_gb),
                utilization_pct: round1(fit.utilization_pct),
                disk_size_gb: round2(fit.model.estimate_disk_gb(&fit.best_quant)),
                best_quant: fit.best_quant.clone(),
                runtime: fit.runtime.label().to_string(),
                use_case: fit.use_case.label().to_string(),
                release_date: fit.model.release_date.clone(),
                license: fit.model.license.clone(),
                is_moe: fit.model.is_moe,
                installed: fit.installed,
            })
            .expect("CSV serialization failed");
    }

    writer.flush().expect("CSV flush failed");
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmfit_core::fit::{FitLevel, InferenceRuntime, ScoreComponents};
    use llmfit_core::models::{Capability, GgufSource, ModelFormat, UseCase};

    fn mock_fit(run_mode: RunMode, use_case: UseCase, model_use_case: &str) -> ModelFit {
        ModelFit {
            model: LlmModel {
                name: "test/model-7b".to_string(),
                provider: "test".to_string(),
                parameter_count: "7B".to_string(),
                parameters_raw: None,
                min_ram_gb: 4.0,
                recommended_ram_gb: 8.0,
                min_vram_gb: Some(4.0),
                quantization: "Q4_K_M".to_string(),
                context_length: 131_072,
                use_case: model_use_case.to_string(),
                is_moe: false,
                num_experts: None,
                active_experts: None,
                active_parameters: None,
                release_date: None,
                gguf_sources: vec![GgufSource {
                    repo: "test/model-7b-GGUF".to_string(),
                    provider: "test".to_string(),
                }],
                capabilities: vec![],
                format: ModelFormat::Gguf,
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
            fit_level: FitLevel::Good,
            run_mode,
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
            use_case,
            runtime: InferenceRuntime::LlamaCpp,
            installed: false,
            fits_with_turboquant: false,
            effective_context_length: 8_192,
        }
    }

    #[test]
    fn llamacpp_command_uses_effective_context() {
        let fit = mock_fit(RunMode::Gpu, UseCase::Chat, "chat");

        let command = generate_llamacpp_command(&fit).expect("expected command");

        assert!(command.contains("-c 8192"));
        assert!(!command.contains("-c 131072"));
    }

    #[test]
    fn llamacpp_command_uses_cpu_only_ngl_zero() {
        let fit = mock_fit(RunMode::CpuOnly, UseCase::General, "general");

        let command = generate_llamacpp_command(&fit).expect("expected command");

        assert!(command.contains("-ngl 0"));
        assert!(!command.contains("-ngl all"));
    }

    #[test]
    fn llamacpp_offload_ngl_args_use_fit_only_when_supported() {
        assert_eq!(
            llamacpp_ngl_args_for_support(RunMode::CpuOffload, true),
            Some("-ngl auto --fit on")
        );
        assert_eq!(
            llamacpp_ngl_args_for_support(RunMode::CpuOffload, false),
            Some("-ngl auto")
        );
        assert_eq!(
            llamacpp_ngl_args_for_support(RunMode::MoeOffload, true),
            Some("-ngl auto --fit on")
        );
        assert_eq!(
            llamacpp_ngl_args_for_support(RunMode::MoeOffload, false),
            Some("-ngl auto")
        );
    }

    #[test]
    fn llamacpp_command_omits_conversation_for_embeddings() {
        let fit = mock_fit(RunMode::Gpu, UseCase::Embedding, "embedding");

        let command = generate_llamacpp_command(&fit).expect("expected command");

        assert!(!command.contains("-cnv"));
    }

    #[test]
    fn llamacpp_command_includes_conversation_for_chat_and_instruct() {
        let chat = mock_fit(RunMode::Gpu, UseCase::Chat, "chat");
        let instruct = mock_fit(RunMode::Gpu, UseCase::General, "instruction tuned");

        let chat_command = generate_llamacpp_command(&chat).expect("expected command");
        let instruct_command = generate_llamacpp_command(&instruct).expect("expected command");

        assert!(chat_command.contains("-cnv"));
        assert!(instruct_command.contains("-cnv"));
    }

    #[test]
    fn llamacpp_command_includes_conversation_for_tool_use() {
        let mut fit = mock_fit(RunMode::Gpu, UseCase::General, "general");
        fit.model.capabilities.push(Capability::ToolUse);

        let command = generate_llamacpp_command(&fit).expect("expected command");

        assert!(command.contains("-cnv"));
    }

    #[test]
    fn llamacpp_command_omits_tensor_parallel_suggestion() {
        let fit = mock_fit(RunMode::TensorParallel, UseCase::Chat, "chat");

        assert!(generate_llamacpp_command(&fit).is_none());
    }

    #[test]
    fn fit_json_includes_effective_context_length() {
        let fit = mock_fit(RunMode::Gpu, UseCase::Chat, "chat");

        let json = fit_to_json(&fit);

        assert_eq!(json["context_length"], 131_072);
        assert_eq!(json["effective_context_length"], 8_192);
    }
}
