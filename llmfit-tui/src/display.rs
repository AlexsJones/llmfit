use colored::*;
use llmfit_core::fit::{FitLevel, ModelFit};
use llmfit_core::hardware::SystemSpecs;
use llmfit_core::models::LlmModel;
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
    #[tabled(rename = "tok/s")]
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
}

pub fn display_all_models(models: &[LlmModel]) {
    println!("\n{}", "=== Available LLM Models ===".bold().cyan());
    println!("Total models: {}\n", models.len());

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
            let status_text = format!("{} {}", fit.fit_emoji(), fit.fit_text());

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
            }
        })
        .collect();

    let table = Table::new(rows).with(Style::rounded()).to_string();
    println!("{}", table);
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
        "{}: {} (est. ~{:.1} tok/s)",
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
    println!("  Estimated Speed: {:.1} tok/s", fit.estimated_tps);
    println!();

    println!("{}", "Resource Requirements:".bold().underline());
    if let Some(vram) = fit.model.min_vram_gb {
        println!("  Min VRAM: {:.1} GB", vram);
    }
    println!("  Min RAM: {:.1} GB (CPU inference)", fit.model.min_ram_gb);
    println!("  Recommended RAM: {:.1} GB", fit.model.recommended_ram_gb);

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

    if !fit.notes.is_empty() {
        println!("{}", "Notes:".bold().underline());
        for note in &fit.notes {
            println!("  {}", note);
        }
        println!();
    }
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
        "use_case": fit.model.use_case,
        "category": fit.use_case.label(),
        "release_date": fit.model.release_date,
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
        "memory_required_gb": round2(fit.memory_required_gb),
        "memory_available_gb": round2(fit.memory_available_gb),
        "utilization_pct": round1(fit.utilization_pct),
        "notes": fit.notes,
    })
}

fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}
