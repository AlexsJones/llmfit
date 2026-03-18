use llmfit_core::fit::ModelFit;
use llmfit_core::hardware::SystemSpecs;
use llmfit_core::models::LlmModel;
use llmfit_core::plan::PlanEstimate;

use super::{DisplayMode, round1, round2};

pub struct JsonDisplay;

impl DisplayMode for JsonDisplay {
    fn display_models(&self, models: &[LlmModel]) {
        let models_json: Vec<serde_json::Value> = models
            .iter()
            .map(|m| {
                serde_json::json!({
                    "name": m.name,
                    "provider": m.provider,
                    "parameter_count": m.parameter_count,
                    "quantization": m.quantization,
                    "context_length": m.context_length,
                    "use_case": m.use_case,
                    "release_date": m.release_date,
                    "is_moe": m.is_moe,
                    "min_vram_gb": m.min_vram_gb,
                    "min_ram_gb": m.min_ram_gb,
                    "recommended_ram_gb": m.recommended_ram_gb,
                    "gguf_sources": m.gguf_sources,
                })
            })
            .collect();

        let output = serde_json::json!({
            "total": models.len(),
            "models": models_json,
        });

        println!(
            "{}",
            serde_json::to_string_pretty(&output).expect("JSON serialization failed")
        );
    }

    fn display_model_fits(&self, specs: &SystemSpecs, fits: &[ModelFit]) {
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

    fn display_model_detail(&self, specs: &SystemSpecs, fit: &ModelFit) {
        let output = serde_json::json!({
            "system": system_json(specs),
            "model": fit_detail_to_json(fit),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&output).expect("JSON serialization failed")
        );
    }

    fn display_system(&self, specs: &SystemSpecs) {
        let output = serde_json::json!({
            "system": system_json(specs),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&output).expect("JSON serialization failed")
        );
    }

    fn display_search_results(&self, models: &[&LlmModel], query: &str) {
        let models_json: Vec<serde_json::Value> = models
            .iter()
            .map(|m| {
                serde_json::json!({
                    "name": m.name,
                    "provider": m.provider,
                    "parameter_count": m.parameter_count,
                    "quantization": m.quantization,
                    "context_length": m.context_length,
                    "use_case": m.use_case,
                })
            })
            .collect();

        let output = serde_json::json!({
            "query": query,
            "total": models.len(),
            "models": models_json,
        });

        println!(
            "{}",
            serde_json::to_string_pretty(&output).expect("JSON serialization failed")
        );
    }

    fn display_plan(&self, _specs: &SystemSpecs, plan: &PlanEstimate) {
        println!(
            "{}",
            serde_json::to_string_pretty(plan).expect("JSON serialization failed")
        );
    }

    fn display_diff(&self, specs: &SystemSpecs, fits: &[ModelFit], _sort_label: &str) {
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
}

// ────────────────────────────────────────────────────────────────────
// Private helpers
// ────────────────────────────────────────────────────────────────────

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
        "gguf_sources": fit.model.gguf_sources,
    })
}

/// Detailed serialisation — mirrors every section shown by `display_model_detail`
/// in the table backend: identity, scores, resources, MoE, fit analysis, sources, notes.
fn fit_detail_to_json(fit: &ModelFit) -> serde_json::Value {
    // MoE block (only present when the model uses the architecture)
    let moe = if fit.model.is_moe {
        let experts = match (fit.model.num_experts, fit.model.active_experts) {
            (Some(total), Some(active)) => serde_json::json!({
                "total": total,
                "active_per_token": active,
            }),
            _ => serde_json::Value::Null,
        };
        serde_json::json!({
            "experts": experts,
            "active_vram_gb": fit.model.moe_active_vram_gb().map(round2),
            "offloaded_gb": fit.moe_offloaded_gb.map(round2),
        })
    } else {
        serde_json::Value::Null
    };

    // GGUF download entries + convenience tip
    let gguf_sources: Vec<serde_json::Value> = fit
        .model
        .gguf_sources
        .iter()
        .map(|src| {
            serde_json::json!({
                "provider": src.provider,
                "repo": src.repo,
                "url": format!("https://huggingface.co/{}", src.repo),
            })
        })
        .collect();

    let download_tip = fit
        .model
        .gguf_sources
        .first()
        .map(|src| format!("llmfit download {} --quant {}", src.repo, fit.best_quant));

    serde_json::json!({
        // ── Identity ──────────────────────────────────────────────
        "name":             fit.model.name,
        "provider":         fit.model.provider,
        "parameter_count":  fit.model.parameter_count,
        "params_b":         round2(fit.model.params_b()),
        "quantization":     fit.model.quantization,
        "best_quant":       fit.best_quant,
        "context_length":   fit.model.context_length,
        "use_case":         fit.model.use_case,
        "category":         fit.use_case.label(),
        "release_date":     fit.model.release_date,
        "is_moe":           fit.model.is_moe,

        // ── Runtime ───────────────────────────────────────────────
        "runtime":          fit.runtime_text(),
        "runtime_label":    fit.runtime.label(),
        "estimated_tps":    round1(fit.estimated_tps),

        // ── Score breakdown ───────────────────────────────────────
        "score": {
            "overall": round1(fit.score),
            "quality": round1(fit.score_components.quality),
            "speed":   round1(fit.score_components.speed),
            "fit":     round1(fit.score_components.fit),
            "context": round1(fit.score_components.context),
        },

        // ── Resource requirements ──────────────────────────────────
        "resources": {
            "min_vram_gb":         fit.model.min_vram_gb.map(round2),
            "min_ram_gb":          round2(fit.model.min_ram_gb),
            "recommended_ram_gb":  round2(fit.model.recommended_ram_gb),
        },

        // ── MoE architecture (null when not applicable) ────────────
        "moe": moe,

        // ── Fit analysis ──────────────────────────────────────────
        "fit": {
            "level":            fit.fit_text(),
            "emoji":            fit.fit_emoji(),
            "run_mode":         fit.run_mode_text(),
            "memory_required_gb":  round2(fit.memory_required_gb),
            "memory_available_gb": round2(fit.memory_available_gb),
            "utilization_pct":     round1(fit.utilization_pct),
        },

        // ── GGUF sources ──────────────────────────────────────────
        "gguf_sources":  gguf_sources,
        "download_tip":  download_tip,

        // ── Notes ────────────────────────────────────────────────
        "notes": fit.notes,
    })
}
