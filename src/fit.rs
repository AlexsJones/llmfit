use crate::hardware::SystemSpecs;
use crate::models::LlmModel;

/// Memory fit -- does the model fit in the available memory pool?
/// Perfect requires GPU acceleration. CPU paths cap at Good.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitLevel {
    Perfect,      // Recommended memory met on GPU
    Good,         // Fits with headroom (GPU tight, or CPU comfortable)
    Marginal,     // Minimum memory met but tight
    TooTight,     // Does not fit in available memory
}

/// Execution path -- how will inference run?
/// This is the "optimization" dimension, independent of memory fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunMode {
    Gpu,         // Fully loaded into VRAM -- fast
    MoeOffload,  // MoE: active experts in VRAM, inactive offloaded to RAM
    CpuOffload,  // Partial GPU offload, spills to system RAM -- mixed
    CpuOnly,     // Entirely in system RAM, no GPU -- slow
}

pub struct ModelFit {
    pub model: LlmModel,
    pub fit_level: FitLevel,
    pub run_mode: RunMode,
    pub memory_required_gb: f64,   // the memory that matters for this run mode
    pub memory_available_gb: f64,  // the memory pool being used
    pub utilization_pct: f64,      // memory_required / memory_available * 100
    pub notes: Vec<String>,
    pub moe_offloaded_gb: Option<f64>, // GB of inactive experts offloaded to RAM
}

impl ModelFit {
    pub fn analyze(model: &LlmModel, system: &SystemSpecs) -> Self {
        let mut notes = Vec::new();

        let min_vram = model.min_vram_gb.unwrap_or(model.min_ram_gb);

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
                    (RunMode::CpuOffload, model.min_ram_gb, system.available_ram_gb)
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
        let fit_level = score_fit(mem_required, mem_available, model.recommended_ram_gb, run_mode);

        let utilization_pct = if mem_available > 0.0 {
            (mem_required / mem_available) * 100.0
        } else {
            f64::INFINITY
        };

        // Supplementary notes
        if run_mode == RunMode::CpuOnly {
            notes.push("No GPU -- inference will be slow".to_string());
        }
        if matches!(run_mode, RunMode::CpuOffload | RunMode::CpuOnly) && system.total_cpu_cores < 4 {
            notes.push("Low CPU core count may bottleneck inference".to_string());
        }

        // Compute MoE offloaded amount if applicable
        let moe_offloaded_gb = if run_mode == RunMode::MoeOffload {
            model.moe_offloaded_ram_gb()
        } else {
            None
        };

        ModelFit {
            model: model.clone(),
            fit_level,
            run_mode,
            memory_required_gb: mem_required,
            memory_available_gb: mem_available,
            utilization_pct,
            notes,
            moe_offloaded_gb,
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
fn score_fit(mem_required: f64, mem_available: f64, recommended: f64, run_mode: RunMode) -> FitLevel {
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
        (RunMode::CpuOffload, model.min_ram_gb, system.available_ram_gb)
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
        // First sort by fit level
        let fit_cmp = match (a.fit_level, b.fit_level) {
            (FitLevel::Perfect, FitLevel::Perfect) => std::cmp::Ordering::Equal,
            (FitLevel::Perfect, _) => std::cmp::Ordering::Less,
            (_, FitLevel::Perfect) => std::cmp::Ordering::Greater,
            (FitLevel::Good, FitLevel::Good) => std::cmp::Ordering::Equal,
            (FitLevel::Good, _) => std::cmp::Ordering::Less,
            (_, FitLevel::Good) => std::cmp::Ordering::Greater,
            (FitLevel::Marginal, FitLevel::Marginal) => std::cmp::Ordering::Equal,
            (FitLevel::Marginal, _) => std::cmp::Ordering::Less,
            (_, FitLevel::Marginal) => std::cmp::Ordering::Greater,
            _ => std::cmp::Ordering::Equal,
        };

        if fit_cmp != std::cmp::Ordering::Equal {
            return fit_cmp;
        }

        // Within same fit level, prefer GPU over CPU
        let mode_cmp = match (a.run_mode, b.run_mode) {
            (RunMode::Gpu, RunMode::Gpu) => std::cmp::Ordering::Equal,
            (RunMode::Gpu, _) => std::cmp::Ordering::Less,
            (_, RunMode::Gpu) => std::cmp::Ordering::Greater,
            (RunMode::MoeOffload, RunMode::MoeOffload) => std::cmp::Ordering::Equal,
            (RunMode::MoeOffload, _) => std::cmp::Ordering::Less,
            (_, RunMode::MoeOffload) => std::cmp::Ordering::Greater,
            (RunMode::CpuOffload, RunMode::CpuOffload) => std::cmp::Ordering::Equal,
            (RunMode::CpuOffload, _) => std::cmp::Ordering::Less,
            (_, RunMode::CpuOffload) => std::cmp::Ordering::Greater,
            _ => std::cmp::Ordering::Equal,
        };

        if mode_cmp != std::cmp::Ordering::Equal {
            return mode_cmp;
        }

        // Then by utilization (lower is better)
        a.utilization_pct.partial_cmp(&b.utilization_pct).unwrap()
    });
    ranked
}
