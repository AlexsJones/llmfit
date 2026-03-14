use colored::*;
use llmfit_core::fit::{FitLevel, ModelFit, SortColumn};
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
        "moe_offloaded_gb": fit.moe_offloaded_gb.map(round2),
        "total_memory_gb": round2(fit.memory_required_gb + fit.moe_offloaded_gb.unwrap_or(0.0)),
        "utilization_pct": round1(fit.utilization_pct),
        "notes": fit.notes,
        "gguf_sources": fit.model.gguf_sources,
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
}

pub fn display_json_plan(plan: &PlanEstimate) {
    println!(
        "{}",
        serde_json::to_string_pretty(plan).expect("JSON serialization failed")
    );
}
