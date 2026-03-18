use colored::*;
use llmfit_core::fit::{FitLevel, ModelFit};
use llmfit_core::hardware::SystemSpecs;
use llmfit_core::models::LlmModel;
use llmfit_core::plan::PlanEstimate;
use tabled::{Table, Tabled, settings::Style};

use super::DisplayMode;

pub struct TableDisplay;

// ────────────────────────────────────────────────────────────────────
// Internal row type used by tabled
// ────────────────────────────────────────────────────────────────────

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
}

impl DisplayMode for TableDisplay {
    fn display_models(&self, models: &[LlmModel]) {
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

    fn display_model_fits(&self, _specs: &SystemSpecs, fits: &[ModelFit]) {
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
        println!(
            "  Note: tok/s values are baseline estimates; real runtime depends on engine/runtime."
        );
    }

    fn display_model_detail(&self, _specs: &SystemSpecs, fit: &ModelFit) {
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

    fn display_system(&self, specs: &SystemSpecs) {
        specs.display();
    }

    fn display_search_results(&self, models: &[&LlmModel], query: &str) {
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

    fn display_plan(&self, specs: &SystemSpecs, plan: &PlanEstimate) {
        specs.display();
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

    fn display_diff(&self, specs: &SystemSpecs, fits: &[ModelFit], sort_label: &str) {
        specs.display();
        if fits.is_empty() {
            return;
        }
        println!(
            "\n{} (sorted by {})\n",
            "=== Model comparison ===".bold().cyan(),
            sort_label
        );
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
}
