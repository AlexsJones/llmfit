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
    let mut notes = fit.notes.clone();
    if let Some(note) = best_quant_mismatch_note(fit) {
        notes.push(note);
    }

    let mut value = serde_json::json!({
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
        "best_quant": sanitized_best_quant(fit),
        "memory_required_gb": round2(fit.memory_required_gb),
        "memory_available_gb": round2(fit.memory_available_gb),
        "moe_offloaded_gb": fit.moe_offloaded_gb.map(round2),
        "total_memory_gb": round2(fit.memory_required_gb + fit.moe_offloaded_gb.unwrap_or(0.0)),
        "utilization_pct": round1(fit.utilization_pct),
        "notes": notes,
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
    });

    // Inserted separately rather than folded into the json! call above: that
    // macro hits rustc's default recursion limit once the object grows this
    // wide (issue #969 added these three keys on top of an already-large
    // envelope).
    let obj = value
        .as_object_mut()
        .expect("fit_to_json returns an object");
    // Derived here rather than read off the fit: `measured_tps` is attached
    // after analysis, and a producer that skipped
    // `refresh_estimate_confidence` would otherwise ship "estimated" in the
    // same object as a measured tok/s figure (issue #969).
    let confidence = fit.effective_estimate_confidence();
    obj.insert(
        "estimate_confidence".to_string(),
        serde_json::json!(confidence.code()),
    );
    obj.insert(
        "estimate_confidence_label".to_string(),
        serde_json::json!(confidence.label()),
    );
    obj.insert(
        "prefill_tps".to_string(),
        serde_json::json!(fit.prefill_tps.map(round1)),
    );
    obj.insert(
        "ttft_ms".to_string(),
        serde_json::json!(fit.ttft_ms.map(round1)),
    );

    value
}

/// `fit.best_quant`, unless it's a GGUF-style label (`Q8_0`, `Q4_K_M`, ...)
/// attached to a model whose own repo name declares a native low-precision
/// format (NVFP4/MXFP4) that GGUF quant strings don't apply to. `None`
/// means "not meaningful for this model", not "no quant chosen" (issue
/// #969, problem 3).
pub fn sanitized_best_quant(fit: &ModelFit) -> Option<&str> {
    if fit.model.is_native_low_precision_named()
        && llmfit_core::models::is_gguf_quant_label(&fit.best_quant)
    {
        None
    } else {
        Some(fit.best_quant.as_str())
    }
}

/// Explanatory note for a cleared `best_quant`, or `None` when nothing was
/// cleared. Appended to the displayed notes list rather than the
/// fit-analysis `fit.notes` vec, since this is a presentation-layer
/// clarification shared by the JSON envelope and the CLI detail pane.
pub(crate) fn best_quant_mismatch_note(fit: &ModelFit) -> Option<String> {
    if sanitized_best_quant(fit).is_some() {
        return None;
    }
    Some(format!(
        "best_quant cleared: '{}' is a native NVFP4/MXFP4 repo, so the GGUF quant label '{}' \
         does not apply",
        fit.model.name, fit.best_quant
    ))
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
    use llmfit_core::EstimateConfidence;
    use llmfit_core::fit::{EstimateBasis, ScoreComponents};
    use llmfit_core::hardware::{GpuBackend, GpuInfo};
    use llmfit_core::models::{LlmModel, ModelFormat, UseCase};

    /// Minimal `ModelFit` for tests that only care about a few fields —
    /// `name`, `best_quant`, `estimate_confidence`, `prefill_tps`, `ttft_ms`.
    fn mock_fit(name: &str, best_quant: &str) -> ModelFit {
        ModelFit {
            model: LlmModel {
                name: name.to_string(),
                provider: "test".to_string(),
                parameter_count: "7B".to_string(),
                parameters_raw: Some(7_000_000_000),
                min_ram_gb: 4.5,
                recommended_ram_gb: 8.0,
                min_vram_gb: Some(4.5),
                quantization: "Q4_K_M".to_string(),
                context_length: 8192,
                use_case: "General".to_string(),
                is_moe: false,
                num_experts: None,
                active_experts: None,
                active_parameters: None,
                release_date: None,
                gguf_sources: vec![],
                capabilities: vec![],
                languages: vec![],
                format: ModelFormat::default(),
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
                architecture: None,
            },
            fit_level: FitLevel::Good,
            run_mode: RunMode::Gpu,
            memory_required_gb: 4.5,
            memory_available_gb: 16.0,
            utilization_pct: 28.1,
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
            best_quant: best_quant.to_string(),
            use_case: UseCase::General,
            runtime: InferenceRuntime::LlamaCpp,
            installed: false,
            fits_with_turboquant: false,
            effective_context_length: 8_192,
            usable_context: 8_192,
            estimate_basis: EstimateBasis::default(),
            measured_tps: None,
            estimate_confidence: EstimateConfidence::Estimated,
            prefill_tps: None,
            ttft_ms: None,
        }
    }

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
            // issue #969: confidence label + prefill/TTFT, additive on top
            // of the #759 envelope.
            "estimate_confidence",
            "estimate_confidence_label",
            "prefill_tps",
            "ttft_ms",
        ] {
            assert!(
                json.get(key).is_some(),
                "shared envelope is missing `{key}`"
            );
        }
    }

    /// Drives the confidence through the fields it is derived from, rather
    /// than by assigning the enum: the envelope re-derives at serialize time,
    /// so those fields are what an API consumer's label actually depends on.
    #[test]
    fn fit_json_serializes_estimate_confidence_code_and_label() {
        use llmfit_core::benchmarks::{MeasuredSource, MeasuredTps};

        let measured = |source| {
            Some(MeasuredTps {
                tok_s: 42.0,
                sample_count: 3,
                hardware_label: "test".to_string(),
                source,
            })
        };

        for (measured_tps, basis, code, label) in [
            (
                measured(MeasuredSource::LocalBench),
                EstimateBasis::default(),
                "measured_local",
                "measured (this machine)",
            ),
            (
                measured(MeasuredSource::Community),
                EstimateBasis::default(),
                "measured_community",
                "measured (community)",
            ),
            (
                None,
                EstimateBasis {
                    local_calibration: Some(1.2),
                    ..EstimateBasis::default()
                },
                "calibrated",
                "calibrated",
            ),
            (
                None,
                EstimateBasis {
                    method: "gpu_bandwidth_roofline".to_string(),
                    ..EstimateBasis::default()
                },
                "estimated",
                "estimated",
            ),
            (
                None,
                EstimateBasis {
                    method: llmfit_core::fit::UNSUPPORTED_METHOD.to_string(),
                    ..EstimateBasis::default()
                },
                "unsupported",
                "unsupported — no basis",
            ),
        ] {
            let mut fit = mock_fit("test/model-7b", "Q4_K_M");
            fit.measured_tps = measured_tps;
            fit.estimate_basis = basis;

            let json = fit_to_json(&fit);

            assert_eq!(json["estimate_confidence"], code);
            assert_eq!(json["estimate_confidence_label"], label);
        }
    }

    /// The envelope must not trust `fit.estimate_confidence`: it is stale for
    /// any producer that attached `measured_tps` without calling
    /// `refresh_estimate_confidence`, and "estimated" alongside a measured
    /// tok/s figure is exactly the contradiction API and MCP consumers can't
    /// see through (issue #969).
    #[test]
    fn fit_json_derives_confidence_instead_of_trusting_a_stale_field() {
        let mut fit = mock_fit("test/model-7b", "Q4_K_M");
        fit.measured_tps = Some(llmfit_core::benchmarks::MeasuredTps {
            tok_s: 42.0,
            sample_count: 3,
            hardware_label: "this machine".to_string(),
            source: llmfit_core::benchmarks::MeasuredSource::LocalBench,
        });
        // Deliberately left at the pre-measurement value.
        assert_eq!(fit.estimate_confidence, EstimateConfidence::Estimated);

        let json = fit_to_json(&fit);

        assert_eq!(json["estimate_confidence"], "measured_local");
        assert_eq!(json["estimate_confidence_label"], "measured (this machine)");
    }

    #[test]
    fn fit_json_prefill_and_ttft_are_null_when_not_estimated() {
        let fit = mock_fit("test/model-7b", "Q4_K_M");
        assert_eq!(fit.prefill_tps, None);
        assert_eq!(fit.ttft_ms, None);

        let json = fit_to_json(&fit);

        assert!(json["prefill_tps"].is_null(), "expected null, not 0.0");
        assert!(json["ttft_ms"].is_null(), "expected null, not 0.0");
    }

    #[test]
    fn fit_json_prefill_and_ttft_carry_values_when_estimated() {
        let mut fit = mock_fit("test/model-7b", "Q4_K_M");
        fit.prefill_tps = Some(1234.56);
        fit.ttft_ms = Some(78.9);

        let json = fit_to_json(&fit);

        assert_eq!(json["prefill_tps"], round1(1234.56));
        assert_eq!(json["ttft_ms"], round1(78.9));
    }

    #[test]
    fn best_quant_clears_gguf_label_on_native_low_precision_named_model() {
        let fit = mock_fit("nvidia/Qwen3-8B-NVFP4", "Q8_0");

        assert_eq!(sanitized_best_quant(&fit), None);

        let json = fit_to_json(&fit);
        assert!(json["best_quant"].is_null());
        let notes: Vec<String> = json["notes"]
            .as_array()
            .expect("notes is an array")
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert!(
            notes.iter().any(|n| n.contains("best_quant cleared")),
            "expected a mismatch note, got: {notes:?}"
        );
    }

    #[test]
    fn best_quant_keeps_gguf_label_for_plain_gguf_model() {
        let fit = mock_fit("acme/plain-7b-GGUF", "Q4_K_M");

        assert_eq!(sanitized_best_quant(&fit), Some("Q4_K_M"));

        let json = fit_to_json(&fit);
        assert_eq!(json["best_quant"], "Q4_K_M");
    }

    #[test]
    fn best_quant_keeps_native_label_for_low_precision_named_model() {
        // MXFP4/NVFP4-named model whose best_quant is already a native
        // label (e.g. picked via the MLX hierarchy) — nothing to clear.
        let fit = mock_fit("amd/MiniMax-M2.1-MXFP4", "mlx-4bit");

        assert_eq!(sanitized_best_quant(&fit), Some("mlx-4bit"));
    }
}
