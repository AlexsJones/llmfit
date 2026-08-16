use llmfit_core::fit::{FitLevel, InferenceRuntime, ModelFit, RunMode};
use llmfit_core::hardware::SystemSpecs;

pub fn system_json(specs: &SystemSpecs) -> serde_json::Value {
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
                "memory_bandwidth_gbps": llmfit_core::hardware::gpu_memory_bandwidth_gbps(&g.name),
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
        "gpu_available_gb": specs.gpu_available_gb.map(round2),
        "gpu_name": specs.gpu_name,
        "gpu_count": specs.gpu_count,
        "unified_memory": specs.unified_memory,
        "backend": specs.backend.label(),
        "gpus": gpus_json,
    })
}

pub fn fit_to_json(fit: &ModelFit) -> serde_json::Value {
    serde_json::json!({
        "name": fit.model.name,
        "provider": fit.model.provider,
        "parameter_count": fit.model.parameter_count,
        "params_b": round2(fit.model.params_b()),
        "context_length": fit.model.context_length,
        "usable_context": fit.usable_context,
        "effective_context_length": fit.effective_context_length,
        "use_case": fit.model.use_case,
        "category": fit.use_case.label(),
        "release_date": fit.model.release_date,
        "is_moe": fit.model.is_moe,
        "fit_level": fit_level_code(fit.fit_level),
        "fit_label": fit.fit_text(),
        "run_mode": run_mode_code(fit.run_mode),
        "run_mode_label": fit.run_mode_text(),
        "score": round1(fit.score),
        "score_components": {
            "quality": round1(fit.score_components.quality),
            "speed": round1(fit.score_components.speed),
            "fit": round1(fit.score_components.fit),
            "context": round1(fit.score_components.context),
        },
        "estimated_tps": round1(fit.estimated_tps),
        "runtime": runtime_code(fit.runtime),
        "runtime_label": fit.runtime_text(),
        "best_quant": fit.best_quant,
        "memory_required_gb": round2(fit.memory_required_gb),
        "memory_available_gb": round2(fit.memory_available_gb),
        "moe_offloaded_gb": fit.moe_offloaded_gb.map(round2),
        "total_memory_gb": round2(fit.memory_required_gb + fit.moe_offloaded_gb.unwrap_or(0.0)),
        "utilization_pct": round1(fit.utilization_pct),
        "notes": fit.notes,
        "gguf_sources": fit.model.gguf_sources,
        "capabilities": fit.model.capabilities,
        "capability_ids": fit.model.capabilities,
        "license": fit.model.license,
        "supports_tp": fit.model.valid_tp_sizes(),
        "installed": fit.installed,
        "disk_size_gb": round2(fit.model.estimate_disk_gb(&fit.best_quant)),
        "ollama_name": llmfit_core::providers::ollama_pull_tag(&fit.model.name),
        "estimate_basis": fit.estimate_basis,
        "verify_command": generate_llamabench_command(fit),
        "measured_tps": fit.measured_tps,
    })
}

pub fn fit_level_code(fit_level: FitLevel) -> &'static str {
    match fit_level {
        FitLevel::Perfect => "perfect",
        FitLevel::Good => "good",
        FitLevel::Marginal => "marginal",
        FitLevel::TooTight => "too_tight",
    }
}

pub fn run_mode_code(run_mode: RunMode) -> &'static str {
    match run_mode {
        RunMode::Gpu => "gpu",
        RunMode::TensorParallel => "tensor_parallel",
        RunMode::MoeOffload => "moe_offload",
        RunMode::CpuOffload => "cpu_offload",
        RunMode::CpuOnly => "cpu_only",
    }
}

pub fn runtime_code(runtime: InferenceRuntime) -> &'static str {
    match runtime {
        InferenceRuntime::Mlx => "mlx",
        InferenceRuntime::LlamaCpp => "llamacpp",
        InferenceRuntime::Vllm => "vllm",
        InferenceRuntime::BitNet => "bitnetcpp",
        InferenceRuntime::Unsupported => "unsupported",
    }
}

/// llama-bench invocation that measures the same quantity `estimated_tps`
/// models: single-request generation throughput (the `tg128` row). Prompt
/// processing (`pp512`) is deliberately not what llmfit estimates.
///
/// Only emitted for pure-GPU and CPU-only runs — offload splits depend on
/// llama.cpp's layer placement, which llama-bench can't express with a fixed
/// `-ngl`, so a benchmark there wouldn't be comparable to the estimate.
pub(crate) fn generate_llamabench_command(fit: &ModelFit) -> Option<String> {
    if fit.runtime != InferenceRuntime::LlamaCpp {
        return None;
    }
    let ngl = match fit.run_mode {
        RunMode::Gpu => "99",
        RunMode::CpuOnly => "0",
        _ => return None,
    };
    // llama-bench needs a local GGUF path (no -hf support); point users at
    // `llmfit download`, which prints the destination path.
    Some(format!(
        "llama-bench -m <path-to-{}-gguf> -ngl {} -p 512 -n 128",
        fit.best_quant, ngl
    ))
}

pub fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

pub fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmfit_core::hardware::{GpuBackend, GpuInfo};

    fn specs_with_gpu(name: &str) -> SystemSpecs {
        SystemSpecs {
            total_ram_gb: 32.0,
            available_ram_gb: 24.0,
            total_cpu_cores: 8,
            cpu_name: "Test CPU".to_string(),
            has_gpu: true,
            gpu_vram_gb: Some(16.0),
            total_gpu_vram_gb: Some(16.0),
            gpu_available_gb: None,
            gpu_name: Some(name.to_string()),
            gpu_count: 1,
            unified_memory: false,
            backend: GpuBackend::Cuda,
            gpus: vec![GpuInfo {
                name: name.to_string(),
                vram_gb: Some(16.0),
                backend: GpuBackend::Cuda,
                count: 1,
                unified_memory: false,
            }],
            cluster_mode: false,
            cluster_node_count: 0,
        }
    }

    #[test]
    fn system_json_includes_per_gpu_memory_bandwidth() {
        let json = system_json(&specs_with_gpu("Tesla T4"));
        assert_eq!(json["gpus"][0]["memory_bandwidth_gbps"], 320.0);
    }

    #[test]
    fn system_json_bandwidth_is_null_for_unknown_gpu() {
        let json = system_json(&specs_with_gpu("Some Unknown GPU"));
        let gpu = &json["gpus"][0];
        assert!(gpu.get("memory_bandwidth_gbps").is_some());
        assert!(gpu["memory_bandwidth_gbps"].is_null());
    }

    #[test]
    fn fit_json_exposes_context_fields() {
        let db = llmfit_core::models::ModelDatabase::new();
        let model = db
            .get_all_models()
            .iter()
            .find(|m| m.context_length > llmfit_core::fit::DEFAULT_ESTIMATION_CTX)
            .expect("catalog has a model with a large context window");
        let fit = ModelFit::analyze(model, &specs_with_gpu("Tesla T4"));

        let json = fit_to_json(&fit);

        assert_eq!(json["usable_context"], fit.usable_context);
        assert_eq!(
            json["effective_context_length"],
            llmfit_core::fit::DEFAULT_ESTIMATION_CTX
        );
        assert!(fit.usable_context <= model.context_length);
    }

    #[test]
    fn fit_json_carries_formerly_cli_only_fields() {
        let db = llmfit_core::models::ModelDatabase::new();
        let model = db
            .get_all_models()
            .iter()
            .next()
            .expect("catalog is non-empty");
        let fit = ModelFit::analyze(model, &specs_with_gpu("Tesla T4"));

        let json = fit_to_json(&fit);

        // Fields that used to live only in the CLI serializer now reach REST/MCP
        // consumers through the shared envelope (issue #759).
        for key in [
            "installed",
            "disk_size_gb",
            "capability_ids",
            "ollama_name",
            "estimate_basis",
            "verify_command",
            "measured_tps",
        ] {
            assert!(
                json.get(key).is_some(),
                "shared envelope is missing `{key}`"
            );
        }
    }
}
