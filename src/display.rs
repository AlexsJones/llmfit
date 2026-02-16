use crate::fit::{FitLevel, ModelFit};
use crate::models::LlmModel;
use colored::*;
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
        println!("\n{}", "No compatible models found for your system.".yellow());
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
    println!("{}: {}", "Context Length".bold(), format!("{} tokens", fit.model.context_length));
    println!("{}: {}", "Use Case".bold(), fit.model.use_case);
    println!("{}: {}", "Category".bold(), fit.use_case.label());
    println!();

    println!("{}", "Score Breakdown:".bold().underline());
    println!("  Overall Score: {:.1} / 100", fit.score);
    println!("  Quality: {:.0}  Speed: {:.0}  Fit: {:.0}  Context: {:.0}",
        fit.score_components.quality, fit.score_components.speed,
        fit.score_components.fit, fit.score_components.context);
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
            println!("  Experts: {} active / {} total per token", active_experts, num_experts);
        }
        if let Some(active_vram) = fit.model.moe_active_vram_gb() {
            println!("  Active VRAM: {:.1} GB (vs {:.1} GB full model)",
                active_vram, fit.model.min_vram_gb.unwrap_or(0.0));
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
    
    println!("  Status: {} {}", 
        fit.fit_emoji(), 
        fit.fit_text().color(fit_color)
    );
    println!("  Run Mode: {}", fit.run_mode_text());
    println!("  Memory Utilization: {:.1}% ({:.1} / {:.1} GB)",
        fit.utilization_pct, fit.memory_required_gb, fit.memory_available_gb);
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
        println!("\n{}", format!("No models found matching '{}'", query).yellow());
        return;
    }

    println!("\n{}", format!("=== Search Results for '{}' ===", query).bold().cyan());
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
            mode: "-".to_string(),
            mem_use: "-".to_string(),
            context: format!("{}k", m.context_length / 1000),
        })
        .collect();

    let table = Table::new(rows).with(Style::rounded()).to_string();
    println!("{}", table);
}
