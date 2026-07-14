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
        "license": fit.model.license,
        "supports_tp": fit.model.valid_tp_sizes(),
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
        InferenceRuntime::Unsupported => "unsupported",
    }
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
}
