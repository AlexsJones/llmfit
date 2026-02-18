use crate::hardware::{GpuBackend, SystemSpecs};
use crate::models::{self, LlmModel, UseCase};

/// Memory fit -- does the model fit in the available memory pool?
/// Perfect requires GPU acceleration. CPU paths cap at Good.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum FitLevel {
    Perfect,  // Recommended memory met on GPU
    Good,     // Fits with headroom (GPU tight, or CPU comfortable)
    Marginal, // Minimum memory met but tight
    TooTight, // Does not fit in available memory
}

/// Execution path -- how will inference run?
/// This is the "optimization" dimension, independent of memory fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum RunMode {
    Gpu,        // Fully loaded into VRAM -- fast
    MoeOffload, // MoE: active experts in VRAM, inactive offloaded to RAM
    CpuOffload, // Partial GPU offload, spills to system RAM -- mixed
    CpuOnly,    // Entirely in system RAM, no GPU -- slow
}

/// Multi-dimensional score components (0-100 each).
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct ScoreComponents {
    /// Quality: model family reputation + param count + quant penalty + task alignment.
    pub quality: f64,
    /// Speed: estimated tokens/sec normalized to 0-100.
    pub speed: f64,
    /// Fit: memory utilization efficiency (closer to filling without exceeding = higher).
    pub fit: f64,
    /// Context: context window capability vs reasonable target.
    pub context: f64,
}

#[derive(Clone)]
pub struct ModelFit {
    pub model: LlmModel,
    pub fit_level: FitLevel,
    pub run_mode: RunMode,
    pub memory_required_gb: f64, // the memory that matters for this run mode
    pub memory_available_gb: f64, // the memory pool being used
    pub utilization_pct: f64,    // memory_required / memory_available * 100
    pub notes: Vec<String>,
    pub moe_offloaded_gb: Option<f64>, // GB of inactive experts offloaded to RAM
    pub score: f64,                    // weighted composite score 0-100
    pub score_components: ScoreComponents,
    pub estimated_tps: f64, // estimated tokens per second
    pub best_quant: String, // best quantization for this hardware
    pub use_case: UseCase,  // inferred use case category
}

impl ModelFit {
    pub fn analyze(model: &LlmModel, system: &SystemSpecs) -> Self {
        let mut notes = Vec::new();

        let min_vram = model.min_vram_gb.unwrap_or(model.min_ram_gb);
        let use_case = UseCase::from_model(model);

        // Step 1: pick the best available execution path
        // Step 2: score memory fit purely on headroom in that path's memory pool
        let (run_mode, mem_required, mem_available) = if system.has_gpu {
            if system.unified_memory {
                // Apple Silicon: GPU and CPU share the same memory pool.
                // No CpuOffload -- there's no separate pool to spill to.
                if let Some(pool) = system.gpu_vram_gb {
                    notes.push("Unified memory: GPU and CPU share the same pool".to_string());
                    if model.is_moe {
                        notes.push(format!(
                            "MoE: {}/{} experts active (all share unified memory pool)",
                            model.active_experts.unwrap_or(0),
                            model.num_experts.unwrap_or(0)
                        ));
                    }
                    (RunMode::Gpu, min_vram, pool)
                } else {
                    cpu_path(model, system, &mut notes)
                }
            } else if let Some(system_vram) = system.gpu_vram_gb {
                if min_vram <= system_vram {
                    // Fits in VRAM -- GPU path
                    notes.push("GPU: model loaded into VRAM".to_string());
                    if model.is_moe {
                        notes.push(format!(
                            "MoE: all {} experts loaded in VRAM (optimal)",
                            model.num_experts.unwrap_or(0)
                        ));
                    }
                    (RunMode::Gpu, min_vram, system_vram)
                } else if model.is_moe {
                    // MoE model: try expert offloading before CPU fallback
                    moe_offload_path(model, system, system_vram, min_vram, &mut notes)
                } else if model.min_ram_gb <= system.available_ram_gb {
                    // Doesn't fit in VRAM, spill to system RAM
                    notes.push("GPU: insufficient VRAM, spilling to system RAM".to_string());
                    notes.push("Performance will be significantly reduced".to_string());
                    (
                        RunMode::CpuOffload,
                        model.min_ram_gb,
                        system.available_ram_gb,
                    )
                } else {
                    // Doesn't fit anywhere -- report against VRAM since GPU is preferred
                    notes.push("Insufficient VRAM and system RAM".to_string());
                    notes.push(format!(
                        "Need {:.1} GB VRAM or {:.1} GB system RAM",
                        min_vram, model.min_ram_gb
                    ));
                    (RunMode::Gpu, min_vram, system_vram)
                }
            } else {
                // GPU detected but VRAM unknown -- fall through to CPU
                notes.push("GPU detected but VRAM unknown".to_string());
                cpu_path(model, system, &mut notes)
            }
        } else {
            cpu_path(model, system, &mut notes)
        };

        // Score fit purely on memory headroom (Perfect requires GPU)
        let fit_level = score_fit(
            mem_required,
            mem_available,
            model.recommended_ram_gb,
            run_mode,
        );

        let utilization_pct = if mem_available > 0.0 {
            (mem_required / mem_available) * 100.0
        } else {
            f64::INFINITY
        };

        // Supplementary notes
        if run_mode == RunMode::CpuOnly {
            notes.push("No GPU -- inference will be slow".to_string());
        }
        if matches!(run_mode, RunMode::CpuOffload | RunMode::CpuOnly) && system.total_cpu_cores < 4
        {
            notes.push("Low CPU core count may bottleneck inference".to_string());
        }

        // Compute MoE offloaded amount if applicable
        let moe_offloaded_gb = if run_mode == RunMode::MoeOffload {
            model.moe_offloaded_ram_gb()
        } else {
            None
        };

        // Dynamic quantization: find best quant that fits
        let budget = mem_available;
        let (best_quant, _best_quant_mem) = model
            .best_quant_for_budget(budget, model.context_length)
            .unwrap_or((model.quantization.as_str(), mem_required));
        let best_quant_str = if best_quant != model.quantization {
            notes.push(format!(
                "Best quantization for hardware: {} (model default: {})",
                best_quant, model.quantization
            ));
            best_quant.to_string()
        } else {
            model.quantization.clone()
        };

        // Speed estimation
        let estimated_tps = estimate_tps(model, &best_quant_str, system, run_mode);

        // Multi-dimensional scoring
        let score_components = compute_scores(
            model,
            &best_quant_str,
            use_case,
            estimated_tps,
            mem_required,
            mem_available,
        );
        let score = weighted_score(score_components, use_case);

        if estimated_tps > 0.0 {
            notes.push(format!("Estimated speed: {:.1} tok/s", estimated_tps));
        }

        ModelFit {
            model: model.clone(),
            fit_level,
            run_mode,
            memory_required_gb: mem_required,
            memory_available_gb: mem_available,
            utilization_pct,
            notes,
            moe_offloaded_gb,
            score,
            score_components,
            estimated_tps,
            best_quant: best_quant_str,
            use_case,
        }
    }

    pub fn fit_emoji(&self) -> &str {
        match self.fit_level {
            FitLevel::Perfect => "🟢",
            FitLevel::Good => "🟡",
            FitLevel::Marginal => "🟠",
            FitLevel::TooTight => "🔴",
        }
    }

    pub fn fit_text(&self) -> &str {
        match self.fit_level {
            FitLevel::Perfect => "Perfect",
            FitLevel::Good => "Good",
            FitLevel::Marginal => "Marginal",
            FitLevel::TooTight => "Too Tight",
        }
    }

    pub fn run_mode_text(&self) -> &str {
        match self.run_mode {
            RunMode::Gpu => "GPU",
            RunMode::MoeOffload => "MoE",
            RunMode::CpuOffload => "CPU+GPU",
            RunMode::CpuOnly => "CPU",
        }
    }
}

/// Pure memory headroom scoring.
/// - GPU (including Apple Silicon unified memory): can reach Perfect.
/// - CpuOffload: caps at Good.
/// - CpuOnly: caps at Marginal -- CPU-only inference is always a compromise.
fn score_fit(
    mem_required: f64,
    mem_available: f64,
    recommended: f64,
    run_mode: RunMode,
) -> FitLevel {
    if mem_required > mem_available {
        return FitLevel::TooTight;
    }

    match run_mode {
        RunMode::Gpu => {
            if recommended <= mem_available {
                FitLevel::Perfect
            } else if mem_available >= mem_required * 1.2 {
                FitLevel::Good
            } else {
                FitLevel::Marginal
            }
        }
        RunMode::MoeOffload => {
            // MoE expert offloading -- GPU handles inference, inactive experts in RAM
            // Good performance with some latency on expert switching
            if mem_available >= mem_required * 1.2 {
                FitLevel::Good
            } else {
                FitLevel::Marginal
            }
        }
        RunMode::CpuOffload => {
            // Mixed GPU/CPU -- decent but not ideal
            if mem_available >= mem_required * 1.2 {
                FitLevel::Good
            } else {
                FitLevel::Marginal
            }
        }
        RunMode::CpuOnly => {
            // CPU-only is always a compromise -- cap at Marginal
            FitLevel::Marginal
        }
    }
}

/// Determine memory pool for CPU-only inference.
fn cpu_path(
    model: &LlmModel,
    system: &SystemSpecs,
    notes: &mut Vec<String>,
) -> (RunMode, f64, f64) {
    notes.push("CPU-only: model loaded into system RAM".to_string());
    if model.is_moe {
        notes.push("MoE architecture, but expert offloading requires a GPU".to_string());
    }
    (RunMode::CpuOnly, model.min_ram_gb, system.available_ram_gb)
}

/// Try MoE expert offloading: active experts in VRAM, inactive in RAM.
/// Falls back to CPU paths if offloading isn't viable.
fn moe_offload_path(
    model: &LlmModel,
    system: &SystemSpecs,
    system_vram: f64,
    total_vram: f64,
    notes: &mut Vec<String>,
) -> (RunMode, f64, f64) {
    if let Some(moe_vram) = model.moe_active_vram_gb() {
        let offloaded_gb = model.moe_offloaded_ram_gb().unwrap_or(0.0);
        if moe_vram <= system_vram && offloaded_gb <= system.available_ram_gb {
            notes.push(format!(
                "MoE: {}/{} experts active in VRAM ({:.1} GB)",
                model.active_experts.unwrap_or(0),
                model.num_experts.unwrap_or(0),
                moe_vram,
            ));
            notes.push(format!(
                "Inactive experts offloaded to system RAM ({:.1} GB)",
                offloaded_gb,
            ));
            return (RunMode::MoeOffload, moe_vram, system_vram);
        }
    }

    // MoE offloading not viable, fall back to generic paths
    if model.min_ram_gb <= system.available_ram_gb {
        notes.push("MoE: insufficient VRAM for expert offloading".to_string());
        notes.push("Spilling entire model to system RAM".to_string());
        notes.push("Performance will be significantly reduced".to_string());
        (
            RunMode::CpuOffload,
            model.min_ram_gb,
            system.available_ram_gb,
        )
    } else {
        notes.push("Insufficient VRAM and system RAM".to_string());
        notes.push(format!(
            "Need {:.1} GB VRAM (full) or {:.1} GB (MoE offload) + RAM",
            total_vram,
            model.moe_active_vram_gb().unwrap_or(total_vram),
        ));
        (RunMode::Gpu, total_vram, system_vram)
    }
}

pub fn rank_models_by_fit(models: Vec<ModelFit>) -> Vec<ModelFit> {
    let mut ranked = models;
    ranked.sort_by(|a, b| {
        // Primary: sort by composite score (higher is better)
        // But TooTight always sorts last
        let a_runnable = a.fit_level != FitLevel::TooTight;
        let b_runnable = b.fit_level != FitLevel::TooTight;

        match (a_runnable, b_runnable) {
            (true, false) => return std::cmp::Ordering::Less,
            (false, true) => return std::cmp::Ordering::Greater,
            _ => {}
        }

        // Within same runnability, sort by score descending
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked
}

// ────────────────────────────────────────────────────────────────────
// Speed estimation
// ────────────────────────────────────────────────────────────────────

/// Estimate tokens per second for a model on given hardware.
/// Based on backend speed constants / model params * quant multiplier.
fn estimate_tps(model: &LlmModel, quant: &str, system: &SystemSpecs, run_mode: RunMode) -> f64 {
    // Backend speed constant K (higher = faster)
    let k: f64 = match system.backend {
        GpuBackend::Cuda => 220.0,
        GpuBackend::Metal => 160.0,
        GpuBackend::Rocm => 180.0,
        GpuBackend::Vulkan => 150.0,
        GpuBackend::Sycl => 100.0,
        GpuBackend::CpuArm => 90.0,
        GpuBackend::CpuX86 => 70.0,
    };

    let params = model.params_b().max(0.1);
    let mut base = k / params;

    // Quantization speed multiplier
    base *= models::quant_speed_multiplier(quant);

    // Threading bonus for many cores
    if system.total_cpu_cores >= 8 {
        base *= 1.1;
    }

    // Run mode penalties
    match run_mode {
        RunMode::Gpu => {}                  // full speed
        RunMode::MoeOffload => base *= 0.8, // expert switching latency
        RunMode::CpuOffload => base *= 0.5, // significant penalty
        RunMode::CpuOnly => base *= 0.3,    // worst case—override K to CPU
    }

    // CPU-only should use CPU K regardless of detected GPU
    if run_mode == RunMode::CpuOnly {
        let cpu_k = if cfg!(target_arch = "aarch64") {
            90.0
        } else {
            70.0
        };
        base = (cpu_k / params) * models::quant_speed_multiplier(quant);
        if system.total_cpu_cores >= 8 {
            base *= 1.1;
        }
    }

    base.max(0.1)
}

// ────────────────────────────────────────────────────────────────────
// Multi-dimensional scoring (Quality, Speed, Fit, Context)
// ────────────────────────────────────────────────────────────────────

fn compute_scores(
    model: &LlmModel,
    quant: &str,
    use_case: UseCase,
    estimated_tps: f64,
    mem_required: f64,
    mem_available: f64,
) -> ScoreComponents {
    ScoreComponents {
        quality: quality_score(model, quant, use_case),
        speed: speed_score(estimated_tps, use_case),
        fit: fit_score(mem_required, mem_available),
        context: context_score(model, use_case),
    }
}

/// Quality score: base quality from param count + family bump + quant penalty + task alignment.
fn quality_score(model: &LlmModel, quant: &str, use_case: UseCase) -> f64 {
    let params = model.params_b();

    // Base quality by parameter count
    let base = if params < 1.0 {
        30.0
    } else if params < 3.0 {
        45.0
    } else if params < 7.0 {
        60.0
    } else if params < 10.0 {
        75.0
    } else if params < 20.0 {
        82.0
    } else if params < 40.0 {
        89.0
    } else {
        95.0
    };

    // Family/provider reputation bumps
    let name_lower = model.name.to_lowercase();
    let family_bump = if name_lower.contains("qwen") {
        2.0
    } else if name_lower.contains("deepseek") {
        3.0
    } else if name_lower.contains("llama") {
        2.0
    } else if name_lower.contains("mistral") || name_lower.contains("mixtral") {
        1.0
    } else if name_lower.contains("gemma") {
        1.0
    } else if name_lower.contains("phi") {
        0.0
    } else if name_lower.contains("starcoder") {
        1.0
    } else {
        0.0
    };

    // Quantization penalty
    let q_penalty = models::quant_quality_penalty(quant);

    // Task alignment bump
    let task_bump = match use_case {
        UseCase::Coding => {
            if name_lower.contains("code")
                || name_lower.contains("starcoder")
                || name_lower.contains("wizard")
            {
                6.0
            } else {
                0.0
            }
        }
        UseCase::Reasoning => {
            if params >= 13.0 {
                5.0
            } else {
                0.0
            }
        }
        UseCase::Multimodal => {
            if name_lower.contains("vision") || model.use_case.to_lowercase().contains("vision") {
                6.0
            } else {
                0.0
            }
        }
        _ => 0.0,
    };

    (base + family_bump + q_penalty + task_bump).clamp(0.0, 100.0)
}

/// Speed score: normalize estimated TPS against target for the use case.
fn speed_score(tps: f64, use_case: UseCase) -> f64 {
    let target = match use_case {
        UseCase::General | UseCase::Coding | UseCase::Multimodal | UseCase::Chat => 40.0,
        UseCase::Reasoning => 25.0,
        UseCase::Embedding => 200.0,
    };
    ((tps / target) * 100.0).clamp(0.0, 100.0)
}

/// Fit score: how well the model fills available memory without exceeding.
fn fit_score(required: f64, available: f64) -> f64 {
    if available <= 0.0 || required > available {
        return 0.0;
    }
    let ratio = required / available;
    // Sweet spot: 50-80% utilization scores highest
    if ratio <= 0.5 {
        // Under-utilizing: still good but not optimal
        60.0 + (ratio / 0.5) * 40.0
    } else if ratio <= 0.8 {
        100.0
    } else if ratio <= 0.9 {
        // Getting tight
        70.0
    } else {
        // Very tight
        50.0
    }
}

/// Context score: context window capability vs target for the use case.
fn context_score(model: &LlmModel, use_case: UseCase) -> f64 {
    let target: u32 = match use_case {
        UseCase::General | UseCase::Chat => 4096,
        UseCase::Coding | UseCase::Reasoning => 8192,
        UseCase::Multimodal => 4096,
        UseCase::Embedding => 512,
    };
    if model.context_length >= target {
        100.0
    } else if model.context_length >= target / 2 {
        70.0
    } else {
        30.0
    }
}

/// Weighted composite score based on use-case category.
/// Weights: [Quality, Speed, Fit, Context]
fn weighted_score(sc: ScoreComponents, use_case: UseCase) -> f64 {
    let (wq, ws, wf, wc) = match use_case {
        UseCase::General => (0.45, 0.30, 0.15, 0.10),
        UseCase::Coding => (0.50, 0.20, 0.15, 0.15),
        UseCase::Reasoning => (0.55, 0.15, 0.15, 0.15),
        UseCase::Chat => (0.40, 0.35, 0.15, 0.10),
        UseCase::Multimodal => (0.50, 0.20, 0.15, 0.15),
        UseCase::Embedding => (0.30, 0.40, 0.20, 0.10),
    };
    let raw = sc.quality * wq + sc.speed * ws + sc.fit * wf + sc.context * wc;
    (raw * 10.0).round() / 10.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{GpuBackend, SystemSpecs};

    // ────────────────────────────────────────────────────────────────────
    // Helper to create test model
    // ────────────────────────────────────────────────────────────────────

    fn test_model(param_count: &str, min_ram: f64, min_vram: Option<f64>) -> LlmModel {
        LlmModel {
            name: "Test Model".to_string(),
            provider: "Test".to_string(),
            parameter_count: param_count.to_string(),
            parameters_raw: None,
            min_ram_gb: min_ram,
            recommended_ram_gb: min_ram * 2.0,
            min_vram_gb: min_vram,
            quantization: "Q4_K_M".to_string(),
            context_length: 4096,
            use_case: "General".to_string(),
            is_moe: false,
            num_experts: None,
            active_experts: None,
            active_parameters: None,
        }
    }

    fn test_system(ram: f64, has_gpu: bool, vram: Option<f64>) -> SystemSpecs {
        SystemSpecs {
            total_ram_gb: ram,
            available_ram_gb: ram * 0.8, // simulate some usage
            total_cpu_cores: 8,
            cpu_name: "Test CPU".to_string(),
            has_gpu,
            gpu_vram_gb: vram,
            gpu_name: if has_gpu {
                Some("Test GPU".to_string())
            } else {
                None
            },
            gpu_count: if has_gpu { 1 } else { 0 },
            unified_memory: false,
            backend: if has_gpu {
                GpuBackend::Cuda
            } else {
                GpuBackend::CpuX86
            },
            gpus: vec![],
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // score_fit tests
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn test_score_fit_too_tight() {
        // Model doesn't fit
        let fit = score_fit(10.0, 8.0, 16.0, RunMode::Gpu);
        assert_eq!(fit, FitLevel::TooTight);
    }

    #[test]
    fn test_score_fit_gpu_perfect() {
        // GPU with recommended memory met
        let fit = score_fit(8.0, 16.0, 12.0, RunMode::Gpu);
        assert_eq!(fit, FitLevel::Perfect);
    }

    #[test]
    fn test_score_fit_gpu_good() {
        // GPU with good headroom but not recommended
        let fit = score_fit(8.0, 10.0, 16.0, RunMode::Gpu);
        assert_eq!(fit, FitLevel::Good);
    }

    #[test]
    fn test_score_fit_gpu_marginal() {
        // GPU with minimal headroom
        let fit = score_fit(8.0, 8.5, 16.0, RunMode::Gpu);
        assert_eq!(fit, FitLevel::Marginal);
    }

    #[test]
    fn test_score_fit_cpu_caps_at_marginal() {
        // CPU-only never reaches Perfect
        let fit = score_fit(4.0, 32.0, 8.0, RunMode::CpuOnly);
        assert_eq!(fit, FitLevel::Marginal);
    }

    #[test]
    fn test_score_fit_cpu_offload_caps_at_good() {
        // CpuOffload with plenty of headroom caps at Good
        let fit = score_fit(8.0, 16.0, 12.0, RunMode::CpuOffload);
        assert_eq!(fit, FitLevel::Good);
    }

    #[test]
    fn test_score_fit_moe_offload() {
        // MoE offload with good headroom
        let fit = score_fit(6.0, 8.0, 12.0, RunMode::MoeOffload);
        assert_eq!(fit, FitLevel::Good);

        // MoE offload with tight fit
        let fit_tight = score_fit(7.0, 7.5, 14.0, RunMode::MoeOffload);
        assert_eq!(fit_tight, FitLevel::Marginal);
    }

    // ────────────────────────────────────────────────────────────────────
    // ModelFit::analyze tests
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn test_model_fit_gpu_path() {
        let model = test_model("7B", 4.0, Some(4.0));
        let system = test_system(16.0, true, Some(8.0));

        let fit = ModelFit::analyze(&model, &system);

        // Should use GPU path
        assert_eq!(fit.run_mode, RunMode::Gpu);
        assert!(matches!(fit.fit_level, FitLevel::Good | FitLevel::Perfect));
        assert_eq!(fit.memory_available_gb, 8.0);
    }

    #[test]
    fn test_model_fit_cpu_only() {
        let model = test_model("7B", 4.0, Some(4.0));
        let system = test_system(16.0, false, None);

        let fit = ModelFit::analyze(&model, &system);

        // Should use CPU path
        assert_eq!(fit.run_mode, RunMode::CpuOnly);
        // CPU-only caps at Marginal
        assert_eq!(fit.fit_level, FitLevel::Marginal);
    }

    #[test]
    fn test_model_fit_cpu_offload() {
        let model = test_model("13B", 8.0, Some(8.0));
        let system = test_system(32.0, true, Some(4.0));

        let fit = ModelFit::analyze(&model, &system);

        // Model doesn't fit in VRAM but fits in RAM
        assert_eq!(fit.run_mode, RunMode::CpuOffload);
        assert!(
            fit.notes
                .iter()
                .any(|n| n.contains("spilling to system RAM"))
        );
    }

    #[test]
    fn test_model_fit_unified_memory() {
        let model = test_model("7B", 4.0, Some(4.0));
        let mut system = test_system(16.0, true, Some(16.0));
        system.unified_memory = true;

        let fit = ModelFit::analyze(&model, &system);

        // Should use GPU path on unified memory
        assert_eq!(fit.run_mode, RunMode::Gpu);
        assert!(fit.notes.iter().any(|n| n.contains("Unified memory")));
    }

    #[test]
    fn test_model_fit_too_tight() {
        let model = test_model("70B", 40.0, Some(40.0));
        let system = test_system(16.0, true, Some(8.0));

        let fit = ModelFit::analyze(&model, &system);

        // Model doesn't fit anywhere
        assert_eq!(fit.fit_level, FitLevel::TooTight);
    }

    #[test]
    fn test_model_fit_utilization() {
        let model = test_model("7B", 4.0, Some(4.0));
        let system = test_system(16.0, true, Some(8.0));

        let fit = ModelFit::analyze(&model, &system);

        // Utilization should be reasonable
        assert!(fit.utilization_pct > 0.0);
        assert!(fit.utilization_pct <= 100.0);
        assert_eq!(
            fit.utilization_pct,
            (fit.memory_required_gb / fit.memory_available_gb) * 100.0
        );
    }

    // ────────────────────────────────────────────────────────────────────
    // rank_models_by_fit tests
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn test_rank_models_by_fit() {
        let model1 = test_model("7B", 4.0, Some(4.0));
        let model2 = test_model("13B", 8.0, Some(8.0));
        let model3 = test_model("70B", 40.0, Some(40.0));

        let system = test_system(16.0, true, Some(10.0));

        let fit1 = ModelFit::analyze(&model1, &system);
        let fit2 = ModelFit::analyze(&model2, &system);
        let fit3 = ModelFit::analyze(&model3, &system);

        let ranked = rank_models_by_fit(vec![fit3.clone(), fit1.clone(), fit2.clone()]);

        // TooTight models should be at the end
        assert_eq!(ranked.last().unwrap().fit_level, FitLevel::TooTight);

        // Runnable models should be sorted by score
        let runnable: Vec<_> = ranked
            .iter()
            .filter(|f| f.fit_level != FitLevel::TooTight)
            .collect();

        // Should be sorted by score descending
        for i in 0..runnable.len() - 1 {
            assert!(runnable[i].score >= runnable[i + 1].score);
        }
    }

    #[test]
    fn test_rank_models_separates_runnable_from_too_tight() {
        let model1 = test_model("7B", 4.0, Some(4.0));
        let model2 = test_model("70B", 40.0, Some(40.0));
        let model3 = test_model("13B", 8.0, Some(8.0));

        let system = test_system(16.0, true, Some(10.0));

        let fit1 = ModelFit::analyze(&model1, &system);
        let fit2 = ModelFit::analyze(&model2, &system); // TooTight
        let fit3 = ModelFit::analyze(&model3, &system);

        let ranked = rank_models_by_fit(vec![fit2, fit1, fit3]);

        // All TooTight should be at the end
        let first_too_tight = ranked
            .iter()
            .position(|f| f.fit_level == FitLevel::TooTight);
        if let Some(pos) = first_too_tight {
            for f in &ranked[pos..] {
                assert_eq!(f.fit_level, FitLevel::TooTight);
            }
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // Scoring function tests
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn test_fit_score_sweet_spot() {
        // Sweet spot: 50-80% utilization
        let score = fit_score(6.0, 10.0);
        assert!(score >= 95.0); // Should be near perfect

        let score2 = fit_score(8.0, 10.0);
        assert_eq!(score2, 100.0);
    }

    #[test]
    fn test_fit_score_under_utilized() {
        // Under-utilizing: still good but not optimal
        let score = fit_score(2.0, 10.0);
        assert!(score >= 60.0);
        assert!(score < 100.0);
    }

    #[test]
    fn test_fit_score_tight() {
        // Very tight fit
        let score = fit_score(9.5, 10.0);
        assert!(score >= 50.0);
        assert!(score < 80.0);
    }

    #[test]
    fn test_fit_score_exceeds_available() {
        // Exceeds available memory
        let score = fit_score(11.0, 10.0);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_speed_score_normalized() {
        // At target TPS
        let score = speed_score(40.0, UseCase::General);
        assert_eq!(score, 100.0);

        // Below target
        let score2 = speed_score(20.0, UseCase::General);
        assert_eq!(score2, 50.0);

        // Above target (capped at 100)
        let score3 = speed_score(80.0, UseCase::General);
        assert_eq!(score3, 100.0);
    }

    #[test]
    fn test_context_score() {
        let model = test_model("7B", 4.0, Some(4.0));

        // Context meets target
        let score = context_score(&model, UseCase::General); // target: 4096
        assert_eq!(score, 100.0);

        // Context below target
        let score2 = context_score(&model, UseCase::Coding); // target: 8192
        assert!(score2 < 100.0);
    }

    #[test]
    fn test_quality_score_by_params() {
        let small = test_model("1B", 1.0, Some(1.0));
        let medium = test_model("7B", 4.0, Some(4.0));
        let large = test_model("70B", 40.0, Some(40.0));

        let score_small = quality_score(&small, "Q4_K_M", UseCase::General);
        let score_medium = quality_score(&medium, "Q4_K_M", UseCase::General);
        let score_large = quality_score(&large, "Q4_K_M", UseCase::General);

        // Larger models should score higher
        assert!(score_medium > score_small);
        assert!(score_large > score_medium);
    }

    #[test]
    fn test_quality_score_quant_penalty() {
        let model = test_model("7B", 4.0, Some(4.0));

        let score_q8 = quality_score(&model, "Q8_0", UseCase::General);
        let score_q4 = quality_score(&model, "Q4_K_M", UseCase::General);
        let score_q2 = quality_score(&model, "Q2_K", UseCase::General);

        // Higher quant should have better quality
        assert!(score_q8 > score_q4);
        assert!(score_q4 > score_q2);
    }

    #[test]
    fn test_weighted_score_composition() {
        let components = ScoreComponents {
            quality: 80.0,
            speed: 70.0,
            fit: 90.0,
            context: 100.0,
        };

        // Different use cases should produce different scores
        let general_score = weighted_score(components, UseCase::General);
        let coding_score = weighted_score(components, UseCase::Coding);
        let embedding_score = weighted_score(components, UseCase::Embedding);

        // All should be valid scores
        assert!(general_score > 0.0 && general_score <= 100.0);
        assert!(coding_score > 0.0 && coding_score <= 100.0);
        assert!(embedding_score > 0.0 && embedding_score <= 100.0);

        // Scores should differ based on different weights
        assert_ne!(general_score, embedding_score);
    }

    #[test]
    fn test_estimate_tps_run_mode_penalties() {
        let model = test_model("7B", 4.0, Some(4.0));
        let system = test_system(16.0, true, Some(10.0));

        let tps_gpu = estimate_tps(&model, "Q4_K_M", &system, RunMode::Gpu);
        let tps_moe = estimate_tps(&model, "Q4_K_M", &system, RunMode::MoeOffload);
        let tps_offload = estimate_tps(&model, "Q4_K_M", &system, RunMode::CpuOffload);
        let tps_cpu = estimate_tps(&model, "Q4_K_M", &system, RunMode::CpuOnly);

        // GPU should be fastest
        assert!(tps_gpu > tps_moe);
        assert!(tps_moe > tps_offload);
        assert!(tps_offload > tps_cpu);

        // All should be positive
        assert!(tps_gpu > 0.0);
        assert!(tps_cpu > 0.0);
    }
}
