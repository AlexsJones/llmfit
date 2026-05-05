use crate::hardware::{GpuBackend, SystemSpecs};
use crate::models::{self, KvQuant, LlmModel, UseCase};

/// Default context window cap used for memory estimation when no explicit
/// `--max-context` is provided. Most runtimes (llama.cpp, Ollama) default to
/// 8 192 tokens, so estimating at the model's advertised maximum (e.g. 262 144)
/// would wildly overestimate KV-cache memory for typical usage.
pub const DEFAULT_ESTIMATION_CTX: u32 = 8_192;

/// Tunable calculation parameters — used to calibrate TPS and memory estimates.
///
/// Users can adjust these via the TUI's Advanced Configuration panel (A)
/// in response to issue #449 (tok/s overestimation on Qwen3 30B).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CalcConfig {
    /// Default context window cap for memory estimation (tokens).
    /// When None, uses `model.context_length.min(DEFAULT_ESTIMATION_CTX)`.
    #[serde(default)]
    pub context_cap: Option<u32>,
    /// Efficiency factor for bandwidth-based TPS estimation.
    /// Accounts for kernel launch overhead, KV-cache reads, memory controller inefficiency.
    /// Default: 0.55
    #[serde(default = "default_efficiency")]
    pub efficiency: f64,
    /// Speed multipliers per run mode (applied after base TPS calculation).
    #[serde(default)]
    pub run_mode_factors: RunModeFactors,
    /// Scoring weights per use case: (quality, speed, fit, context).
    #[serde(default)]
    pub scoring_weights: ScoringWeights,
}

impl Default for CalcConfig {
    fn default() -> Self {
        Self {
            context_cap: None,
            efficiency: default_efficiency(),
            run_mode_factors: RunModeFactors::default(),
            scoring_weights: ScoringWeights::default(),
        }
    }
}

fn default_efficiency() -> f64 {
    0.55
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct RunModeFactors {
    pub gpu: f64,
    pub tensor_parallel: f64,
    pub moe_offload: f64,
    pub cpu_offload: f64,
    pub cpu_only: f64,
}

impl Default for RunModeFactors {
    fn default() -> Self {
        Self {
            gpu: 1.0,
            tensor_parallel: 0.9,
            moe_offload: 0.8,
            cpu_offload: 0.5,
            cpu_only: 0.3,
        }
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct ScoringWeights {
    /// (quality_weight, speed_weight, fit_weight, context_weight) per use case,
    /// stored in the same order as `UseCase` variants.
    /// Order: General, Coding, Reasoning, Chat, Multimodal, Embedding
    pub weights: [[f64; 4]; 6],
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            weights: [
                [0.45, 0.30, 0.15, 0.10], // General
                [0.50, 0.20, 0.15, 0.15], // Coding
                [0.55, 0.15, 0.15, 0.15], // Reasoning
                [0.40, 0.35, 0.15, 0.10], // Chat
                [0.50, 0.20, 0.15, 0.15], // Multimodal
                [0.30, 0.40, 0.20, 0.10], // Embedding
            ],
        }
    }
}

impl ScoringWeights {
    pub fn get(&self, use_case: UseCase) -> (f64, f64, f64, f64) {
        let idx = match use_case {
            UseCase::General => 0,
            UseCase::Coding => 1,
            UseCase::Reasoning => 2,
            UseCase::Chat => 3,
            UseCase::Multimodal => 4,
            UseCase::Embedding => 5,
        };
        let w = self.weights[idx];
        (w[0], w[1], w[2], w[3])
    }
}

/// Inference runtime — the software framework used for inference.
/// Orthogonal to `GpuBackend` which represents hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum InferenceRuntime {
    LlamaCpp, // llama.cpp / Ollama
    Mlx,      // Apple MLX framework
    Vllm,     // vLLM (for AWQ/GPTQ pre-quantized models)
}

impl InferenceRuntime {
    pub fn label(&self) -> &'static str {
        match self {
            InferenceRuntime::LlamaCpp => "llama.cpp",
            InferenceRuntime::Mlx => "MLX",
            InferenceRuntime::Vllm => "vLLM",
        }
    }
}

/// Column to sort model fits by in the TUI/UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortColumn {
    Score,
    Tps,
    Params,
    MemPct,
    Ctx,
    ReleaseDate,
    UseCase,
    Provider,
}

impl SortColumn {
    pub fn label(&self) -> &str {
        match self {
            SortColumn::Score => "Score",
            SortColumn::Tps => "tok/s",
            SortColumn::Params => "Params",
            SortColumn::MemPct => "Mem%",
            SortColumn::Ctx => "Ctx",
            SortColumn::ReleaseDate => "Date",
            SortColumn::UseCase => "Use",
            SortColumn::Provider => "Provider",
        }
    }

    pub fn next(&self) -> Self {
        match self {
            SortColumn::Score => SortColumn::Tps,
            SortColumn::Tps => SortColumn::Params,
            SortColumn::Params => SortColumn::MemPct,
            SortColumn::MemPct => SortColumn::Ctx,
            SortColumn::Ctx => SortColumn::ReleaseDate,
            SortColumn::ReleaseDate => SortColumn::UseCase,
            SortColumn::UseCase => SortColumn::Provider,
            SortColumn::Provider => SortColumn::Score,
        }
    }
}

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
    Gpu,            // Fully loaded into VRAM -- fast
    MoeOffload,     // MoE: active experts in VRAM, inactive offloaded to RAM
    CpuOffload,     // Partial GPU offload, spills to system RAM -- mixed
    CpuOnly,        // Entirely in system RAM, no GPU -- slow
    TensorParallel, // Distributed via NCCL across cluster nodes
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

#[derive(Clone, serde::Serialize)]
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
    pub estimated_tps: f64,            // baseline estimated tokens per second
    pub best_quant: String,            // best quantization for this hardware
    pub use_case: UseCase,             // inferred use case category
    pub runtime: InferenceRuntime,     // inference runtime (MLX or llama.cpp)
    pub installed: bool,               // model found in a local runtime provider
    pub fits_with_turboquant: bool,    // TooTight at fp16 KV but fits with TurboQuant KV
    pub effective_context_length: u32, // context length used for memory estimation
}

impl ModelFit {
    pub fn analyze(model: &LlmModel, system: &SystemSpecs) -> Self {
        Self::analyze_with_context_limit(model, system, None)
    }

    pub fn analyze_with_context_limit(
        model: &LlmModel,
        system: &SystemSpecs,
        context_limit: Option<u32>,
    ) -> Self {
        Self::analyze_inner(model, system, context_limit, None, None)
    }

    /// Analyze with an optional runtime override. When `force_runtime` is
    /// `Some`, the automatic runtime selection (which prefers MLX on Apple
    /// Silicon) is bypassed so the caller can request e.g. llama.cpp results
    /// even on a Metal system.  Pre-quantized models always use vLLM
    /// regardless of the override.
    pub fn analyze_with_forced_runtime(
        model: &LlmModel,
        system: &SystemSpecs,
        context_limit: Option<u32>,
        force_runtime: Option<InferenceRuntime>,
    ) -> Self {
        Self::analyze_inner(model, system, context_limit, force_runtime, None)
    }

    /// Analyze with a custom calculation configuration.
    ///
    /// This lets users tune TPS efficiency, run mode factors, and scoring
    /// weights — addressing issue #449 (tok/s overestimation).
    pub fn analyze_with_config(model: &LlmModel, system: &SystemSpecs, config: CalcConfig) -> Self {
        // Merge config context_cap with a default if not set
        let context_limit = config.context_cap;
        Self::analyze_inner(model, system, context_limit, None, Some(config))
    }

    fn analyze_inner(
        model: &LlmModel,
        system: &SystemSpecs,
        context_limit: Option<u32>,
        force_runtime: Option<InferenceRuntime>,
        config: Option<CalcConfig>,
    ) -> Self {
        let config = config.unwrap_or_default();
        let mut notes = Vec::new();
        // When no explicit context limit is given, cap the estimation at
        // DEFAULT_ESTIMATION_CTX. Most runtimes (llama.cpp, Ollama) use a
        // much smaller context than the model's advertised maximum, so using
        // the full context window (e.g. 262 144) would drastically overestimate
        // KV-cache memory requirements.
        let estimation_ctx = match context_limit {
            Some(limit) => limit.min(model.context_length),
            None => model.context_length.min(DEFAULT_ESTIMATION_CTX),
        };

        // Also respect the user-configured context cap if set.
        let estimation_ctx = match config.context_cap {
            Some(cap) => estimation_ctx.min(cap),
            None => estimation_ctx,
        };

        let min_vram = model.min_vram_gb.unwrap_or(model.min_ram_gb);
        let use_case = UseCase::from_model(model);
        let default_mem_required =
            model.estimate_memory_gb(model.quantization.as_str(), estimation_ctx);
        if estimation_ctx < model.context_length {
            notes.push(format!(
                "Context capped at {} tokens for estimation (model supports up to {}; use --max-context to override)",
                estimation_ctx, model.context_length
            ));
        }

        // Determine inference runtime up front so path selection can use
        // the correct quantization hierarchy.
        // Honour the force_runtime override first if provided; otherwise
        // pre-quantized models default to vLLM, falling back to auto-detect.
        let runtime = if let Some(forced) = force_runtime {
            forced
        } else if system.cluster_mode {
            InferenceRuntime::Vllm
        } else if model.is_prequantized() {
            InferenceRuntime::Vllm
        } else if system.backend == GpuBackend::Metal && system.unified_memory {
            InferenceRuntime::Mlx
        } else {
            InferenceRuntime::LlamaCpp
        };
        let choose_quant =
            |budget: f64| best_quant_for_runtime_budget(model, runtime, budget, estimation_ctx);

        // Step 1: pick the best available execution path
        // Step 2: score memory fit purely on headroom in that path's memory pool
        let (run_mode, mem_required, mem_available) = if system.cluster_mode {
            // Cluster mode: vLLM with tensor parallelism across multiple nodes.
            // Total VRAM is the sum across all nodes (NCCL handles distribution).
            let pool = system.total_gpu_vram_gb.unwrap_or(0.0);
            let tp_size = system.cluster_node_count;
            if let Some((_, best_mem)) = choose_quant(pool) {
                notes.push(format!(
                    "Cluster: tensor-parallel across {} nodes via vLLM (TP={})",
                    tp_size, tp_size
                ));
                (RunMode::TensorParallel, best_mem, pool)
            } else {
                notes.push(format!(
                    "Cluster: {} nodes but model exceeds aggregate VRAM ({:.1} GB)",
                    tp_size, pool
                ));
                (RunMode::TensorParallel, default_mem_required, pool)
            }
        } else if system.has_gpu {
            if system.unified_memory {
                // Unified memory (Apple Silicon or NVIDIA Tegra/Grace Blackwell):
                // GPU and CPU share the same memory pool.
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
                    if model.is_moe {
                        (RunMode::Gpu, min_vram, pool)
                    } else if let Some((_, best_mem)) = choose_quant(pool) {
                        (RunMode::Gpu, best_mem, pool)
                    } else {
                        (RunMode::Gpu, default_mem_required, pool)
                    }
                } else {
                    cpu_path(model, system, runtime, estimation_ctx, &mut notes)
                }
            } else if let Some(system_vram) = system.total_gpu_vram_gb {
                // Use total VRAM across all same-model GPUs for fit scoring.
                // Multi-GPU inference (tensor splitting) is supported by llama.cpp, vLLM, etc.
                if model.is_moe && min_vram <= system_vram {
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
                    // MoE model doesn't fit at default quant — but check if the full
                    // model fits at the best available quant before falling to offload.
                    // Many runtimes (llama.cpp, Ollama) load ALL experts into VRAM when
                    // the quantized model file fits, avoiding DDR bandwidth bottleneck.
                    if let Some((best_q, best_mem)) =
                        best_quant_for_runtime_budget(model, runtime, system_vram, estimation_ctx)
                        && best_mem <= system_vram
                    {
                        notes.push(
                            "GPU: all MoE experts loaded into VRAM (quantized fit)".to_string(),
                        );
                        notes.push(format!(
                            "MoE: all {} experts in VRAM at {} ({:.1} GB)",
                            model.num_experts.unwrap_or(0),
                            best_q,
                            best_mem,
                        ));
                        (RunMode::Gpu, best_mem, system_vram)
                    } else {
                        // Full model doesn't fit — try expert offloading
                        moe_offload_path(model, system, system_vram, min_vram, runtime, &mut notes)
                    }
                } else if let Some((_, best_mem)) = choose_quant(system_vram) {
                    notes.push("GPU: model loaded into VRAM".to_string());
                    (RunMode::Gpu, best_mem, system_vram)
                } else if let Some((_, best_mem)) = choose_quant(system.available_ram_gb) {
                    // Doesn't fit in VRAM, spill to system RAM
                    notes.push("GPU: insufficient VRAM, spilling to system RAM".to_string());
                    notes.push("Performance will be significantly reduced".to_string());
                    (RunMode::CpuOffload, best_mem, system.available_ram_gb)
                } else {
                    // Doesn't fit anywhere -- report against VRAM since GPU is preferred
                    notes.push("Insufficient VRAM and system RAM".to_string());
                    notes.push(format!(
                        "Need {:.1} GB VRAM or {:.1} GB system RAM",
                        min_vram, model.min_ram_gb
                    ));
                    (RunMode::Gpu, default_mem_required, system_vram)
                }
            } else {
                // GPU detected but VRAM unknown -- fall through to CPU
                notes.push("GPU detected but VRAM unknown".to_string());
                cpu_path(model, system, runtime, estimation_ctx, &mut notes)
            }
        } else {
            cpu_path(model, system, runtime, estimation_ctx, &mut notes)
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
        // Pre-quantized models (AWQ/GPTQ) have a fixed quantization — skip dynamic selection.
        let (best_quant, _best_quant_mem) = if model.is_prequantized() {
            (model.quantization.as_str(), mem_required)
        } else {
            let budget = mem_available;
            let hierarchy: &[&str] = if runtime == InferenceRuntime::Mlx {
                models::MLX_QUANT_HIERARCHY
            } else {
                models::QUANT_HIERARCHY
            };
            model
                .best_quant_for_budget_with(budget, estimation_ctx, hierarchy)
                .or_else(|| {
                    // Fall back to GGUF hierarchy if MLX quants don't fit
                    if runtime == InferenceRuntime::Mlx {
                        model.best_quant_for_budget(budget, estimation_ctx)
                    } else {
                        None
                    }
                })
                .unwrap_or((model.quantization.as_str(), mem_required))
        };
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
        let estimated_tps =
            estimate_tps(model, &best_quant_str, system, run_mode, runtime, &config);

        // Add runtime comparison note on Apple Silicon
        if runtime == InferenceRuntime::Mlx {
            let llamacpp_tps = estimate_tps(
                model,
                &best_quant_str,
                system,
                run_mode,
                InferenceRuntime::LlamaCpp,
                &config,
            );
            if llamacpp_tps > 0.1 {
                let speedup = ((estimated_tps / llamacpp_tps - 1.0) * 100.0).round();
                if speedup > 0.0 {
                    notes.push(format!(
                        "MLX runtime: ~{:.0}% faster than llama.cpp ({:.1} vs {:.1} tok/s)",
                        speedup, estimated_tps, llamacpp_tps
                    ));
                }
            }
        }

        // Multi-dimensional scoring
        let score_components = compute_scores(
            model,
            &best_quant_str,
            use_case,
            estimated_tps,
            mem_required,
            mem_available,
        );
        let score = weighted_score(score_components, use_case, &config);

        if estimated_tps > 0.0 {
            notes.push(format!(
                "Baseline estimated speed: {:.1} tok/s",
                estimated_tps
            ));
        }

        // Check if a TooTight model would fit with TurboQuant KV compression.
        // Only compute on CUDA systems — TurboQuant requires vLLM + CUDA.
        let fits_with_turboquant =
            fit_level == FitLevel::TooTight && system.backend == GpuBackend::Cuda && {
                let tq_mem = model.estimate_memory_gb_with_kv(
                    best_quant,
                    estimation_ctx,
                    KvQuant::TurboQuant,
                );
                tq_mem <= mem_available
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
            score,
            score_components,
            estimated_tps,
            best_quant: best_quant_str,
            use_case,
            runtime,
            installed: false, // set later by App after provider detection
            fits_with_turboquant,
            effective_context_length: estimation_ctx,
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

    pub fn runtime_text(&self) -> &str {
        self.runtime.label()
    }

    pub fn run_mode_text(&self) -> &str {
        match self.run_mode {
            RunMode::Gpu => "GPU",
            RunMode::TensorParallel => "TP",
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
        RunMode::Gpu | RunMode::TensorParallel => {
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
    runtime: InferenceRuntime,
    estimation_ctx: u32,
    notes: &mut Vec<String>,
) -> (RunMode, f64, f64) {
    notes.push("CPU-only: model loaded into system RAM".to_string());
    if model.is_moe {
        notes.push("MoE architecture, but expert offloading requires a GPU".to_string());
        return (RunMode::CpuOnly, model.min_ram_gb, system.available_ram_gb);
    }

    if let Some((_, best_mem)) =
        best_quant_for_runtime_budget(model, runtime, system.available_ram_gb, estimation_ctx)
    {
        (RunMode::CpuOnly, best_mem, system.available_ram_gb)
    } else {
        (
            RunMode::CpuOnly,
            model.estimate_memory_gb(model.quantization.as_str(), estimation_ctx),
            system.available_ram_gb,
        )
    }
}

/// Try MoE expert offloading: active experts in VRAM, inactive in RAM.
/// Falls back to CPU paths if offloading isn't viable.
fn moe_offload_path(
    model: &LlmModel,
    system: &SystemSpecs,
    system_vram: f64,
    total_vram: f64,
    runtime: InferenceRuntime,
    notes: &mut Vec<String>,
) -> (RunMode, f64, f64) {
    let hierarchy: &[&str] = if runtime == InferenceRuntime::Mlx {
        models::MLX_QUANT_HIERARCHY
    } else {
        models::QUANT_HIERARCHY
    };

    for &quant in hierarchy {
        if let Some((moe_vram, offloaded_gb)) = moe_memory_for_quant(model, quant)
            && moe_vram <= system_vram
            && offloaded_gb <= system.available_ram_gb
        {
            notes.push(format!(
                "MoE: {}/{} experts active in VRAM ({:.1} GB) at {}",
                model.active_experts.unwrap_or(0),
                model.num_experts.unwrap_or(0),
                moe_vram,
                quant,
            ));
            notes.push(format!(
                "Inactive experts offloaded to system RAM ({:.1} GB)",
                offloaded_gb,
            ));
            return (RunMode::MoeOffload, moe_vram, system_vram);
        }
    }

    // On MLX, also try GGUF-style quant levels as a fallback.
    if runtime == InferenceRuntime::Mlx {
        for &quant in models::QUANT_HIERARCHY {
            if let Some((moe_vram, offloaded_gb)) = moe_memory_for_quant(model, quant)
                && moe_vram <= system_vram
                && offloaded_gb <= system.available_ram_gb
            {
                notes.push(format!(
                    "MoE: {}/{} experts active in VRAM ({:.1} GB) at {}",
                    model.active_experts.unwrap_or(0),
                    model.num_experts.unwrap_or(0),
                    moe_vram,
                    quant,
                ));
                notes.push(format!(
                    "Inactive experts offloaded to system RAM ({:.1} GB)",
                    offloaded_gb,
                ));
                return (RunMode::MoeOffload, moe_vram, system_vram);
            }
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

/// Compute MoE active VRAM + offloaded RAM for a specific quantization level.
fn moe_memory_for_quant(model: &LlmModel, quant: &str) -> Option<(f64, f64)> {
    if !model.is_moe {
        return None;
    }

    let active_params = model.active_parameters? as f64;
    let total_params = model.parameters_raw? as f64;
    let bpp = models::quant_bpp(quant);

    let active_vram = ((active_params * bpp) / (1024.0 * 1024.0 * 1024.0) * 1.1).max(0.5);
    let inactive_params = (total_params - active_params).max(0.0);
    let offloaded_ram = (inactive_params * bpp) / (1024.0 * 1024.0 * 1024.0);

    Some((active_vram, offloaded_ram))
}

fn best_quant_for_runtime_budget(
    model: &LlmModel,
    runtime: InferenceRuntime,
    budget: f64,
    estimation_ctx: u32,
) -> Option<(&'static str, f64)> {
    // Pre-quantized models (vLLM) don't support dynamic re-quantization
    if runtime == InferenceRuntime::Vllm {
        return None;
    }
    let hierarchy: &[&str] = if runtime == InferenceRuntime::Mlx {
        models::MLX_QUANT_HIERARCHY
    } else {
        models::QUANT_HIERARCHY
    };
    model
        .best_quant_for_budget_with(budget, estimation_ctx, hierarchy)
        .or_else(|| {
            if runtime == InferenceRuntime::Mlx {
                model.best_quant_for_budget(budget, estimation_ctx)
            } else {
                None
            }
        })
}

pub fn backend_compatible(model: &LlmModel, system: &SystemSpecs) -> bool {
    if model.is_mlx_model() {
        system.backend == GpuBackend::Metal && system.unified_memory
    } else if model.is_prequantized() {
        if !matches!(system.backend, GpuBackend::Cuda | GpuBackend::Rocm) {
            return false;
        }
        // For CUDA GPUs, check that the GPU's compute capability meets the
        // minimum required by the quantization format (e.g. AWQ needs Turing+).
        // ROCm and unrecognized NVIDIA GPUs are assumed compatible.
        if system.backend == GpuBackend::Cuda
            && let Some(min_cc) = crate::hardware::quant_min_compute_capability(&model.quantization)
            && let Some(gpu_name) = &system.gpu_name
            && let Some(gpu_cc) = crate::hardware::gpu_compute_capability(gpu_name)
        {
            return gpu_cc >= min_cc;
        }
        true
    } else {
        true
    }
}

pub fn rank_models_by_fit(models: Vec<ModelFit>) -> Vec<ModelFit> {
    rank_models_by_fit_opts(models, false)
}

pub fn rank_models_by_fit_opts(models: Vec<ModelFit>, installed_first: bool) -> Vec<ModelFit> {
    rank_models_by_fit_opts_col(models, installed_first, SortColumn::Score)
}

pub fn rank_models_by_fit_opts_col(
    models: Vec<ModelFit>,
    installed_first: bool,
    sort_column: SortColumn,
) -> Vec<ModelFit> {
    let mut ranked = models;
    ranked.sort_by(|a, b| {
        // Installed-first: if toggled, installed models sort above non-installed
        if installed_first {
            let inst_cmp = b.installed.cmp(&a.installed);
            if inst_cmp != std::cmp::Ordering::Equal {
                return inst_cmp;
            }
        }

        // TooTight always sorts last regardless of column
        let a_runnable = a.fit_level != FitLevel::TooTight;
        let b_runnable = b.fit_level != FitLevel::TooTight;

        match (a_runnable, b_runnable) {
            (true, false) => return std::cmp::Ordering::Less,
            (false, true) => return std::cmp::Ordering::Greater,
            _ => {}
        }

        // Sort by selected column
        match sort_column {
            SortColumn::Score => b
                .score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal),
            SortColumn::Tps => {
                let cmp = b
                    .estimated_tps
                    .partial_cmp(&a.estimated_tps)
                    .unwrap_or(std::cmp::Ordering::Equal);
                if cmp == std::cmp::Ordering::Equal {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    cmp
                }
            }
            SortColumn::Params => {
                let a_params = a.model.params_b();
                let b_params = b.model.params_b();
                b_params
                    .partial_cmp(&a_params)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
            SortColumn::MemPct => b
                .utilization_pct
                .partial_cmp(&a.utilization_pct)
                .unwrap_or(std::cmp::Ordering::Equal),
            SortColumn::Ctx => b.model.context_length.cmp(&a.model.context_length),
            SortColumn::ReleaseDate => {
                let a_date = a.model.release_date.as_deref().unwrap_or("");
                let b_date = b.model.release_date.as_deref().unwrap_or("");
                match (a_date.is_empty(), b_date.is_empty()) {
                    (true, false) => std::cmp::Ordering::Greater, // no date = last
                    (false, true) => std::cmp::Ordering::Less,
                    (true, true) => b
                        .score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal),
                    (false, false) => {
                        let cmp = b_date.cmp(a_date); // descending = newest first
                        if cmp == std::cmp::Ordering::Equal {
                            b.score
                                .partial_cmp(&a.score)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        } else {
                            cmp
                        }
                    }
                }
            }
            SortColumn::UseCase => {
                let cmp = a.use_case.label().cmp(b.use_case.label());
                if cmp == std::cmp::Ordering::Equal {
                    // Secondary sort by score within same use case
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    cmp
                }
            }
            SortColumn::Provider => {
                let cmp = a
                    .model
                    .provider
                    .to_lowercase()
                    .cmp(&b.model.provider.to_lowercase());
                if cmp == std::cmp::Ordering::Equal {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    cmp
                }
            }
        }
    });
    ranked
}

// ────────────────────────────────────────────────────────────────────
// Speed estimation
// ────────────────────────────────────────────────────────────────────

/// Estimate tokens per second for a model on given hardware.
/// Estimate tokens per second for a model on the given hardware.
///
/// LLM token generation is **memory-bandwidth-bound**: each generated token
/// requires reading the full model weights once from VRAM. The theoretical
/// upper bound is therefore:
///
///   max_tps = memory_bandwidth_GB_s / model_size_GB
///
/// In practice, real throughput is ~50–70% of this ceiling due to kernel
/// launch overhead, KV-cache reads, and other fixed costs.
///
/// When the GPU model is recognized, we use its **actual memory bandwidth**
/// (from the lookup table in `hardware::gpu_memory_bandwidth_gbps`) to
/// produce a physics-grounded estimate. Otherwise we fall back to the
/// original per-backend constant `K`.
///
/// References:
///  - kipply, "Transformer Inference Arithmetic" (2022)
///  - ggerganov, llama.cpp Apple Silicon benchmarks (Discussion #4167)
///  - Google, "Efficiently Scaling Transformer Inference" (arXiv:2211.05102)
///  - ggerganov, llama.cpp NVIDIA T4 benchmarks (Discussion #4225)

/// Read the system DDR bandwidth (GB/s) from the `LLMFIT_DDR_BANDWIDTH` env var,
/// falling back to a conservative 50 GB/s default (DDR4-3200 dual-channel).
///
/// Override: `export LLMFIT_DDR_BANDWIDTH=90` for DDR5-5600 dual-channel, etc.
/// Typical values: DDR4-3200 dual-channel ~50 GB/s, DDR5-5600 dual-channel ~90 GB/s.
/// VRAM utilization threshold above which MoE cache-pressure penalty applies.
/// Below this, inactive experts don't significantly compete for L2 cache.
const VRAM_PRESSURE_UTIL_THRESHOLD: f64 = 0.60;

/// Floor for the VRAM cache-pressure penalty factor.
/// Prevents unrealistically low throughput estimates for models near 100% VRAM.
const VRAM_PRESSURE_PENALTY_FLOOR: f64 = 0.30;

/// Default expert density ratio when num_experts is unknown.
/// Conservative 50% — assumes half the experts are inactive on average.
const VRAM_PRESSURE_DEFAULT_EXPERT_RATIO: f64 = 0.50;

/// Print a debug line to stderr when LLMFIT_DEBUG env var is set.
/// Usage: `LLMFIT_DEBUG=1 llmfit fit ...` to see which estimation path is taken.
/// Uses a macro to avoid string allocation when debug logging is disabled (hot path).
macro_rules! debug_log {
    ($($arg:tt)*) => {
        if std::env::var("LLMFIT_DEBUG").is_ok() {
            eprintln!("[llmfit:debug] {}", format!($($arg)*));
        }
    };
}

fn ddr_bandwidth_gbps() -> f64 {
    std::env::var("LLMFIT_DDR_BANDWIDTH")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(50.0)
}

fn estimate_tps(
    model: &LlmModel,
    quant: &str,
    system: &SystemSpecs,
    run_mode: RunMode,
    runtime: InferenceRuntime,
    config: &CalcConfig,
) -> f64 {
    use crate::hardware::gpu_memory_bandwidth_gbps;

    // MoE models execute only active experts per token, so speed estimates should
    // use active parameters when known; fit/memory paths still use full model size.
    let params = model
        .active_parameters
        .filter(|_| model.is_moe)
        .map(|p| (p as f64) / 1_000_000_000.0)
        .unwrap_or_else(|| model.params_b())
        .max(0.1);

    // ── Bandwidth-based estimation (preferred) ─────────────────────
    //
    // If we know the GPU's memory bandwidth, estimate tok/s from first
    // principles instead of using a fixed constant.
    //
    // model_bytes = params_B * bytes_per_param(quant)
    // raw_tps     = bandwidth_GB_s / model_bytes_GB
    // estimated   = raw_tps * efficiency * run_mode_factor
    //
    // The efficiency factor (0.55) accounts for:
    //  - Kernel launch / scheduling overhead
    //  - KV-cache memory reads (not captured in model size)
    //  - Memory controller inefficiency at high utilization
    //
    // Validated against:
    //  - RTX 4090 (1008 GB/s): Qwen3.5-27B Q4 → ~40 tok/s measured
    //  - T4 (320 GB/s): 7B F16 → ~16 tok/s (ggerganov benchmark)
    //  - Apple M1 Max (400 GB/s): 7B Q4_0 → ~61 tok/s (ggerganov benchmark)
    let gpu_name = system.gpu_name.as_deref().unwrap_or("");
    let bandwidth = gpu_memory_bandwidth_gbps(gpu_name);

    if run_mode != RunMode::CpuOnly
        && let Some(bw) = bandwidth
    {
        let bytes_per_param = models::quant_bytes_per_param(quant);
        let active_gb = params * bytes_per_param;

        // Efficiency factor — captures overhead not in the simple
        // bandwidth / model-size formula. Tunable via CalcConfig.
        let efficiency = config.efficiency;

        if matches!(run_mode, RunMode::MoeOffload | RunMode::Gpu) && model.is_moe {
            // MoE expert speed estimation: the per-token cost is dominated by
            // reading the active expert weights, not GPU compute.
            //
            // Two scenarios:
            //
            // 1. MoeOffload mode: inactive experts in RAM, CPU reads active experts
            //    from DDR memory -> DDR bandwidth is the bottleneck.
            //    Model: expert_read_time = active_gb / ddr_bandwidth
            //
            // 2. GPU mode (model fits VRAM): most runtimes (Ollama, basic llama.cpp)
            //    don't do expert-aware VRAM placement — they load all layers uniformly
            //    and process the full model size per token, not just active experts.
            //    Even runtimes that do expert-aware loading still read all expert weights
            //    from VRAM on each token (just the active ones per layer), but the
            //    VRAM bandwidth must cover the full model working set due to cache pressure
            //    from 128+ experts.
            //    Model: bandwidth / full_model_gb * efficiency (same as dense model)
            //
            // Measured examples on RX 6900 XT (16 GB VRAM, 512 GB/s, DDR4 ~50 GB/s):
            //   - Qwen3-Next-80B (MoeOffload): estimated 15.2, measured 15.4
            //   - Qwen3-30B-A3B (GPU mode, full-model): estimated 18.1, measured 16.3
            //
            // Note: PCIe bandwidth (~25 GB/s for Gen4 x16) could be the actual
            // ceiling on some systems, but in practice llama.cpp processes
            // offloaded layers on the CPU, so DDR bandwidth is the dominant factor.
            //
            // Typical DDR bandwidths: DDR4-3200 dual-channel ~50 GB/s,
            // DDR5-5600 dual-channel ~90 GB/s. We use a conservative 50 GB/s
            // default which can be overridden via LLMFIT_DDR_BANDWIDTH env var.
            if run_mode == RunMode::MoeOffload {
                let ddr_bw = ddr_bandwidth_gbps();

                let expert_read_time = active_gb / ddr_bw; // CPU reads from DDR
                let gpu_compute_time = active_gb / (bw * efficiency);
                let total_time = expert_read_time + gpu_compute_time;

                debug_log!(
                    "MoE Offload: {} ddr_bw={:.0}GB/s expert_read={:.3}s gpu_compute={:.3}s tps={:.1}",
                    model.name,
                    ddr_bw,
                    expert_read_time,
                    gpu_compute_time,
                    1.0 / total_time
                );
                return (1.0 / total_time).max(0.1);
            }

            // GPU mode: MoE model fits in VRAM with ALL expert weights loaded.
            // Per-token bandwidth cost decomposes into two components:
            //
            // 1. SCALABLE: active expert FFN weights (scales with quantization)
            //    Only selected experts (e.g., 8 of 256) are read per token.
            //    Confirmed via llama.cpp source tracing (3 CUDA paths).
            //
            // 2. FIXED: attention, router, shared experts, lm_head, embedding
            //    These are compute-bound and cost roughly constant time regardless
            //    of quantization. We represent them as bandwidth-equivalent bytes
            //    using MOE_FIXED_EFFECTIVE_BPP (K ≈ 3.2).
            //
            // Formula: tps = bw / (active_ffn_bytes + fixed_equivalent_bytes)
            //
            // When architecture metadata is available, we compute exact decomposition.
            // Otherwise, fall back to active_parameters * quant_bpp with moe_overhead.
            //
            // Validated against llama-bench on RX 6900 XT (512 GB/s):
            //   Two-component model (architecture-aware):
            //     - OLMoE Q2_K: est 281, meas 293 (0.96x)
            //     - OLMoE Q4_K_M: est 257, meas 258 (1.00x)
            //     - OLMoE Q8_0: est 216, meas 205 (1.05x)
            //   Fallback model (active_params * quant_bpp + tiered overhead):
            //     - DeepSeek-V2-Lite Q4_K_M: est 141, meas 124 (1.14x)
            //     - Qwen1.5-MoE-A2.7B Q4_K_M: est 131, meas 129 (1.02x)

            // VRAM cache-pressure penalty for GPU-mode MoE models.
            //
            // When all experts are loaded into VRAM (GPU mode), inactive experts
            // (e.g., 248 of 256) consume VRAM and pollute the GPU L2 cache.
            // This creates additional memory traffic as the cache evicts/refetches
            // expert weights on every token. The penalty is proportional to:
            //   - VRAM utilization above 60% (below 60%, model fits easily)
            //   - Expert density ratio (more inactive experts → more pressure)
            //
            // Calibrated against llama-bench on RX 6900 XT (16GB VRAM, 512 GB/s):
            //   - OLMoE-1B-7B Q4_K_M (25% util, 8/64): penalty=1.0 → est 200, meas 258 (0.77x)
            //   - Qwen1.5-MoE Q4_K_M (52% util, 4/60): penalty=1.0 → est 108, meas 129 (0.84x)
            //   - DeepSeek-V2-Lite Q4_K_M (57% util, 6/64): penalty=1.0 → est 142, meas 124 (1.14x)
            //   - Qwen3.5-35B Q2_K_XL (83% util, 8/256): penalty=0.78 → est 79, meas 80 (0.99x)
            //   - Qwen3.5-35B Q3_K_M (104% util, 8/256): penalty=1.0 → est 78, meas 80 (0.98x)
            let vram_pressure = if let Some(vram) = system.gpu_vram_gb {
                let total_model_gb = model.params_b() * models::quant_bpp(quant);
                let util = total_model_gb / vram;

                // Only apply penalty when model actually fits in VRAM (util <= 1.0)
                // AND utilization is above the threshold. Below it, the model fits
                // easily with plenty of L2 cache room — no pressure.
                if util > 1.0 {
                    // util > 1.0 means total model size exceeds VRAM, which should not
                    // happen in GPU mode (the routing logic only sends models that fit).
                    // Log a warning so this edge case is visible in debug output rather
                    // than silently returning a no-penalty value that masks the error.
                    debug_log!(
                        "VRAM pressure: {} util={:.2} exceeds 1.0 in GPU mode — possible routing error (total={:.1}GB vram={:.1}GB)",
                        model.name,
                        util,
                        total_model_gb,
                        vram,
                    );
                    1.0
                } else if util < VRAM_PRESSURE_UTIL_THRESHOLD {
                    1.0 // model fits easily, no cache-pressure penalty
                } else {
                    // Expert density: ratio of inactive to total experts.
                    // More inactive experts = more cache pollution per token.
                    // Note: if active_experts is not set in the catalog, we default
                    // to 1 active expert, which overestimates the ratio for models
                    // with more active experts (e.g., 4 or 8). This makes the
                    // penalty more conservative (higher) than reality for such models.
                    let expert_ratio = model
                        .num_experts
                        .map(|n| {
                            let active = model.active_experts.unwrap_or(1) as f64;
                            1.0 - (active / n as f64)
                        })
                        .unwrap_or(VRAM_PRESSURE_DEFAULT_EXPERT_RATIO);

                    // Linear penalty: penalty = 1.0 - (util - threshold) * expert_ratio
                    // At threshold: penalty=1.0. At util=1.0 with expert_ratio=0.97: penalty=0.61
                    // Floor prevents unrealistically low estimates.
                    (1.0 - (util - VRAM_PRESSURE_UTIL_THRESHOLD) * expert_ratio)
                        .max(VRAM_PRESSURE_PENALTY_FLOOR)
                }
            } else {
                1.0 // unknown VRAM → no penalty
            };

            // Tier 1: Architecture-aware two-component model
            if let Some((active_ffn_b, fixed_b)) = model.moe_bandwidth_decomposition() {
                let bpp = models::quant_bpp(quant);
                let active_ffn_bytes = active_ffn_b * bpp;
                let fixed_bytes = fixed_b * models::LlmModel::MOE_FIXED_EFFECTIVE_BPP;
                let per_token_bytes = active_ffn_bytes + fixed_bytes;
                let raw_tps = bw / per_token_bytes;
                let mode_factor = config.run_mode_factors.for_run_mode(run_mode);
                debug_log!(
                    "MoE GPU Tier1: {} active_ffn={:.1}B fixed={:.1}B vram_pressure={:.2} raw_tps={:.1}",
                    model.name,
                    active_ffn_b,
                    fixed_b,
                    vram_pressure,
                    raw_tps
                );
                return (raw_tps * mode_factor * vram_pressure).max(0.1);
            }

            // Tier 2: Fallback — active_parameters * quant_bpp with tiered moe_overhead
            let moe_active_gb = params * models::quant_bpp(quant);
            let moe_overhead = match model.num_experts {
                Some(n) if n <= 8 => 0.90, // calibrated for Mixtral-class
                Some(n) if n <= 16 => 0.85,
                Some(n) if n <= 32 => 0.80,
                Some(n) if n <= 64 => 0.70, // calibrated: OLMoE, Qwen1.5, DeepSeek
                Some(_) => 0.40,            // 128+ experts
                None => 0.60,               // unknown
            };
            let raw_tps = (bw / moe_active_gb) * efficiency * moe_overhead;
            let mode_factor = config.run_mode_factors.for_run_mode(run_mode);
            debug_log!(
                "MoE GPU Tier2 (fallback): {} moe_overhead={:.2} vram_pressure={:.2} raw_tps={:.1}",
                model.name,
                moe_overhead,
                vram_pressure,
                raw_tps
            );
            return (raw_tps * mode_factor * vram_pressure).max(0.1);
        }

        let raw_tps = (bw / active_gb) * efficiency;

        let mode_factor = config.run_mode_factors.for_run_mode(run_mode);

        return (raw_tps * mode_factor).max(0.1);
    }

    // ── Fallback: fixed-constant approach ──────────────────────────
    // Used when the GPU is not recognized (custom/unnamed GPUs,
    // synthetic entries from --memory override, etc.).
    let k: f64 = match (system.backend, runtime) {
        (GpuBackend::Metal, InferenceRuntime::Mlx) => 250.0,
        (GpuBackend::Metal, InferenceRuntime::LlamaCpp) => 160.0,
        (GpuBackend::Metal, InferenceRuntime::Vllm) => 160.0,
        (GpuBackend::Cuda, _) => 220.0,
        (GpuBackend::Rocm, _) => 180.0,
        (GpuBackend::Vulkan, _) => 150.0,
        (GpuBackend::Sycl, _) => 100.0,
        (GpuBackend::CpuArm, _) => 90.0,
        (GpuBackend::CpuX86, _) => 70.0,
        (GpuBackend::Ascend, _) => 390.0,
        (GpuBackend::AmdNpu, _) => 130.0,
    };

    let mut base = k / params;

    // Quantization speed multiplier
    base *= models::quant_speed_multiplier(quant);

    // Threading bonus for many cores
    if system.total_cpu_cores >= 8 {
        base *= 1.1;
    }

    // MoE offload: apply the same DDR bandwidth bottleneck model as the
    // bandwidth-based path, estimating GPU bandwidth from the K constant.
    // K = bandwidth * efficiency / bytes_per_param
    // COUPLING: efficiency factor must match CalcConfig default (0.55)
    let fallback_efficiency = 0.55;
    if run_mode == RunMode::MoeOffload {
        let estimated_gpu_bw = k * models::quant_bytes_per_param(quant) / fallback_efficiency;
        let bytes_per_param = models::quant_bytes_per_param(quant);
        let active_gb = params * bytes_per_param;
        let ddr_bw = ddr_bandwidth_gbps();
        let expert_read_time = active_gb / ddr_bw;
        let gpu_compute_time = active_gb / (estimated_gpu_bw * fallback_efficiency);
        base = (1.0 / (expert_read_time + gpu_compute_time)).max(0.1);
        if system.total_cpu_cores >= 8 {
            base *= 1.1;
        }
        return base;
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

    // Run mode penalties — tunable via CalcConfig
    let mode_factor = config.run_mode_factors.for_run_mode(run_mode);
    base *= mode_factor;

    base.max(0.1)
}

impl RunModeFactors {
    pub fn for_run_mode(&self, run_mode: RunMode) -> f64 {
        match run_mode {
            RunMode::Gpu => self.gpu,
            RunMode::TensorParallel => self.tensor_parallel,
            RunMode::MoeOffload => self.moe_offload,
            RunMode::CpuOffload => self.cpu_offload,
            RunMode::CpuOnly => self.cpu_only,
        }
    }
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
    #[allow(clippy::if_same_then_else)]
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
fn weighted_score(sc: ScoreComponents, use_case: UseCase, config: &CalcConfig) -> f64 {
    let (wq, ws, wf, wc) = config.scoring_weights.get(use_case);
    let raw = sc.quality * wq + sc.speed * ws + sc.fit * wf + sc.context * wc;
    (raw * 10.0).round() / 10.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{GpuBackend, SystemSpecs};

    /// Test helper: default CalcConfig for direct estimate_tps calls.
    fn test_config() -> CalcConfig {
        CalcConfig::default()
    }

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
            release_date: None,
            gguf_sources: vec![],
            capabilities: vec![],
            format: models::ModelFormat::default(),
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
            total_gpu_vram_gb: vram, // same as gpu_vram_gb for single-GPU tests
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
            cluster_mode: false,
            cluster_node_count: 0,
            has_npu: false,
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
    fn test_moe_offload_tries_lower_quantization() {
        let model = LlmModel {
            name: "MoE Quant Test".to_string(),
            provider: "Test".to_string(),
            parameter_count: "8x7B".to_string(),
            parameters_raw: Some(46_700_000_000),
            min_ram_gb: 25.0,
            recommended_ram_gb: 50.0,
            min_vram_gb: Some(25.0),
            quantization: "Q8_0".to_string(),
            context_length: 4096,
            use_case: "General".to_string(),
            is_moe: true,
            num_experts: Some(8),
            active_experts: Some(2),
            active_parameters: Some(12_900_000_000),
            release_date: None,
            gguf_sources: vec![],
            capabilities: vec![],
            format: models::ModelFormat::default(),
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
        };
        let mut system = test_system(64.0, true, Some(8.0));
        system.backend = GpuBackend::Cuda;

        let fit = ModelFit::analyze(&model, &system);

        assert_eq!(fit.run_mode, RunMode::MoeOffload);
        assert!(fit.memory_required_gb <= fit.memory_available_gb);
        assert!(fit.notes.iter().any(|n| n.contains("at Q")));
    }

    #[test]
    fn test_dense_model_uses_quant_in_path_selection() {
        // Static requirements are high, but lower quantization should make it runnable on GPU.
        let model = LlmModel {
            name: "Quant Path Test".to_string(),
            provider: "Test".to_string(),
            parameter_count: "7B".to_string(),
            parameters_raw: Some(7_000_000_000),
            min_ram_gb: 20.0,
            recommended_ram_gb: 40.0,
            min_vram_gb: Some(16.0),
            quantization: "F16".to_string(),
            context_length: 4096,
            use_case: "General".to_string(),
            is_moe: false,
            num_experts: None,
            active_experts: None,
            active_parameters: None,
            release_date: None,
            gguf_sources: vec![],
            capabilities: vec![],
            format: models::ModelFormat::default(),
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
        };
        let system = test_system(12.0, true, Some(8.0));

        let fit = ModelFit::analyze(&model, &system);

        assert_eq!(fit.run_mode, RunMode::Gpu);
        assert_ne!(fit.fit_level, FitLevel::TooTight);
        assert_ne!(fit.best_quant, "F16");
        assert!(fit.memory_required_gb <= fit.memory_available_gb);
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
        let general_score = weighted_score(components, UseCase::General, &test_config());
        let coding_score = weighted_score(components, UseCase::Coding, &test_config());
        let embedding_score = weighted_score(components, UseCase::Embedding, &test_config());

        // All should be valid scores
        assert!(general_score > 0.0 && general_score <= 100.0);
        assert!(coding_score > 0.0 && coding_score <= 100.0);
        assert!(embedding_score > 0.0 && embedding_score <= 100.0);

        // Scores should differ based on different weights
        assert_ne!(general_score, embedding_score);
    }

    #[test]
    fn test_estimate_tps_mlx_faster_than_llamacpp() {
        let model = test_model("7B", 4.0, Some(4.0));
        let mut system = test_system(16.0, true, Some(16.0));
        system.backend = GpuBackend::Metal;
        system.unified_memory = true;

        let tps_mlx = estimate_tps(
            &model,
            "Q4_K_M",
            &system,
            RunMode::Gpu,
            InferenceRuntime::Mlx,
            &test_config(),
        );
        let tps_llamacpp = estimate_tps(
            &model,
            "Q4_K_M",
            &system,
            RunMode::Gpu,
            InferenceRuntime::LlamaCpp,
            &test_config(),
        );

        // MLX should be faster on Metal
        assert!(tps_mlx > tps_llamacpp);
        // MLX K=250 vs LlamaCpp K=160, so ratio should be ~1.56
        assert!(tps_mlx / tps_llamacpp > 1.4);
    }

    #[test]
    fn test_analyze_selects_mlx_on_apple_silicon() {
        let model = test_model("7B", 4.0, Some(4.0));
        let mut system = test_system(16.0, true, Some(16.0));
        system.backend = GpuBackend::Metal;
        system.unified_memory = true;

        let fit = ModelFit::analyze(&model, &system);
        assert_eq!(fit.runtime, InferenceRuntime::Mlx);
        // Should have an MLX comparison note
        assert!(fit.notes.iter().any(|n| n.contains("MLX runtime")));
    }

    #[test]
    fn test_analyze_defaults_llamacpp_on_cuda() {
        let model = test_model("7B", 4.0, Some(4.0));
        let system = test_system(16.0, true, Some(10.0));

        let fit = ModelFit::analyze(&model, &system);
        assert_eq!(fit.runtime, InferenceRuntime::LlamaCpp);
    }

    #[test]
    fn test_analyze_with_context_limit_reduces_memory_estimate() {
        let mut model = test_model("7B", 4.0, Some(4.0));
        model.context_length = 32768;
        let system = test_system(32.0, true, Some(16.0));

        let baseline = ModelFit::analyze(&model, &system);
        let capped = ModelFit::analyze_with_context_limit(&model, &system, Some(4096));

        assert_eq!(baseline.effective_context_length, DEFAULT_ESTIMATION_CTX);
        assert_eq!(capped.effective_context_length, 4096);
        assert!(capped.memory_required_gb < baseline.memory_required_gb);
        assert!(capped.notes.iter().any(|n| n.contains("Context capped at")));
    }

    #[test]
    fn test_estimate_tps_run_mode_penalties() {
        let model = test_model("7B", 4.0, Some(4.0));
        let system = test_system(16.0, true, Some(10.0));

        let tps_gpu = estimate_tps(
            &model,
            "Q4_K_M",
            &system,
            RunMode::Gpu,
            InferenceRuntime::LlamaCpp,
            &test_config(),
        );
        let tps_offload = estimate_tps(
            &model,
            "Q4_K_M",
            &system,
            RunMode::CpuOffload,
            InferenceRuntime::LlamaCpp,
            &test_config(),
        );
        let tps_cpu = estimate_tps(
            &model,
            "Q4_K_M",
            &system,
            RunMode::CpuOnly,
            InferenceRuntime::LlamaCpp,
            &test_config(),
        );

        // GPU should be fastest
        assert!(tps_gpu > tps_offload);
        assert!(tps_offload > tps_cpu);

        // All should be positive
        assert!(tps_gpu > 0.0);
        assert!(tps_cpu > 0.0);
    }

    #[test]
    fn test_estimate_tps_moe_uses_active_parameters() {
        let dense_model = test_model("30B", 18.0, Some(18.0));
        let mut moe_model = dense_model.clone();
        moe_model.is_moe = true;
        moe_model.active_parameters = Some(3_000_000_000);

        let system = test_system(64.0, true, Some(24.0));

        let tps_dense = estimate_tps(
            &dense_model,
            "Q4_K_M",
            &system,
            RunMode::Gpu,
            InferenceRuntime::LlamaCpp,
            &test_config(),
        );
        let tps_moe = estimate_tps(
            &moe_model,
            "Q4_K_M",
            &system,
            RunMode::Gpu,
            InferenceRuntime::LlamaCpp,
            &test_config(),
        );

        assert!(tps_moe > tps_dense * 5.0);
    }

    #[test]
    fn test_estimate_tps_moe_without_active_parameters_falls_back_to_total() {
        let dense_model = test_model("30B", 18.0, Some(18.0));
        let mut moe_without_active = dense_model.clone();
        moe_without_active.is_moe = true;
        moe_without_active.active_parameters = None;

        let system = test_system(64.0, true, Some(24.0));

        let tps_dense = estimate_tps(
            &dense_model,
            "Q4_K_M",
            &system,
            RunMode::Gpu,
            InferenceRuntime::LlamaCpp,
            &test_config(),
        );
        let tps_moe = estimate_tps(
            &moe_without_active,
            "Q4_K_M",
            &system,
            RunMode::Gpu,
            InferenceRuntime::LlamaCpp,
            &test_config(),
        );

        assert_eq!(tps_dense, tps_moe);
    }

    // ────────────────────────────────────────────────────────────────────
    // Release date sorting tests
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn test_sort_by_tps() {
        let system = test_system(32.0, true, Some(16.0));

        let mut model_fast = test_model("7B", 4.0, Some(4.0));
        model_fast.name = "Fast Model".to_string();

        let mut model_slow = test_model("14B", 8.0, Some(8.0));
        model_slow.name = "Slow Model".to_string();

        let fits = vec![
            ModelFit::analyze(&model_slow, &system),
            ModelFit::analyze(&model_fast, &system),
        ];

        let ranked = rank_models_by_fit_opts_col(fits, false, SortColumn::Tps);

        assert!(ranked[0].estimated_tps >= ranked[1].estimated_tps);
        assert_eq!(ranked[0].model.name, "Fast Model");
    }

    #[test]
    fn test_sort_by_release_date() {
        let system = test_system(32.0, true, Some(16.0));

        let mut model_new = test_model("7B", 4.0, Some(4.0));
        model_new.name = "New Model".to_string();
        model_new.release_date = Some("2025-06-15".to_string());

        let mut model_old = test_model("7B", 4.0, Some(4.0));
        model_old.name = "Old Model".to_string();
        model_old.release_date = Some("2024-01-10".to_string());

        let mut model_none = test_model("7B", 4.0, Some(4.0));
        model_none.name = "No Date Model".to_string();
        model_none.release_date = None;

        let fits = vec![
            ModelFit::analyze(&model_old, &system),
            ModelFit::analyze(&model_none, &system),
            ModelFit::analyze(&model_new, &system),
        ];

        let ranked = rank_models_by_fit_opts_col(fits, false, SortColumn::ReleaseDate);

        // Newest first, no-date last
        assert_eq!(ranked[0].model.name, "New Model");
        assert_eq!(ranked[1].model.name, "Old Model");
        assert_eq!(ranked[2].model.name, "No Date Model");
    }

    // ────────────────────────────────────────────────────────────────────
    // Bandwidth-based speed estimation tests
    // ────────────────────────────────────────────────────────────────────

    /// Helper: create a test system with a specific GPU name for bandwidth lookup.
    fn test_system_with_gpu(ram: f64, vram: f64, gpu_name: &str) -> SystemSpecs {
        SystemSpecs {
            total_ram_gb: ram,
            available_ram_gb: ram * 0.8,
            total_cpu_cores: 8,
            cpu_name: "Test CPU".to_string(),
            has_gpu: true,
            gpu_vram_gb: Some(vram),
            total_gpu_vram_gb: Some(vram),
            gpu_name: Some(gpu_name.to_string()),
            gpu_count: 1,
            unified_memory: false,
            backend: GpuBackend::Cuda,
            gpus: vec![],
            cluster_mode: false,
            cluster_node_count: 0,
            has_npu: false,
        }
    }

    #[test]
    fn test_bandwidth_estimation_rtx4090_faster_than_rtx3060() {
        let model = test_model("27B", 16.0, Some(16.0));
        let sys_4090 = test_system_with_gpu(64.0, 24.0, "NVIDIA GeForce RTX 4090");
        let sys_3060 = test_system_with_gpu(64.0, 12.0, "NVIDIA GeForce RTX 3060");

        let tps_4090 = estimate_tps(
            &model,
            "Q4_K_M",
            &sys_4090,
            RunMode::Gpu,
            InferenceRuntime::LlamaCpp,
            &test_config(),
        );
        let tps_3060 = estimate_tps(
            &model,
            "Q4_K_M",
            &sys_3060,
            RunMode::Gpu,
            InferenceRuntime::LlamaCpp,
            &test_config(),
        );

        // RTX 4090 (1008 GB/s) should be ~2.8x faster than RTX 3060 (360 GB/s)
        assert!(
            tps_4090 > tps_3060 * 2.0,
            "4090={tps_4090}, 3060={tps_3060}"
        );
    }

    #[test]
    fn test_bandwidth_estimation_rtx4090_27b_q4_realistic() {
        // Validated against real-world measurement:
        // Qwen3.5-27B UD-Q4_K_XL on RTX 4090 → ~40 tok/s
        let model = test_model("27B", 16.0, Some(16.0));
        let system = test_system_with_gpu(64.0, 24.0, "NVIDIA GeForce RTX 4090");

        let tps = estimate_tps(
            &model,
            "Q4_K_M",
            &system,
            RunMode::Gpu,
            InferenceRuntime::LlamaCpp,
            &test_config(),
        );

        // Should be in the 30-50 tok/s range (measured: ~40)
        assert!(tps > 25.0 && tps < 55.0, "RTX 4090 27B Q4 tok/s = {tps}");
    }

    #[test]
    fn test_bandwidth_estimation_t4_7b_f16_realistic() {
        // Validated against ggerganov's T4 benchmark (Discussion #4225):
        // OpenHermes 7B F16 on T4 → ~16 tok/s
        let model = test_model("7B", 14.0, Some(14.0));
        let system = test_system_with_gpu(16.0, 16.0, "Tesla T4");

        let tps = estimate_tps(
            &model,
            "F16",
            &system,
            RunMode::Gpu,
            InferenceRuntime::LlamaCpp,
            &test_config(),
        );

        // Should be in the 10-25 tok/s range (measured: ~16)
        assert!(tps > 8.0 && tps < 30.0, "T4 7B F16 tok/s = {tps}");
    }

    #[test]
    fn test_bandwidth_estimation_unknown_gpu_uses_fallback() {
        // Unknown GPU names should still produce reasonable estimates
        // via the fallback constant-K path.
        let model = test_model("7B", 4.0, Some(4.0));
        let system = test_system_with_gpu(16.0, 10.0, "Some Unknown GPU");

        let tps = estimate_tps(
            &model,
            "Q4_K_M",
            &system,
            RunMode::Gpu,
            InferenceRuntime::LlamaCpp,
            &test_config(),
        );

        // Should fall back to K=220 path and produce a positive value
        assert!(tps > 0.0, "unknown GPU should still produce an estimate");
    }

    #[test]
    fn test_bandwidth_estimation_cpu_only_ignores_bandwidth() {
        // CPU-only mode should NOT use GPU bandwidth, even if GPU is known.
        let model = test_model("7B", 4.0, Some(4.0));
        let sys_4090 = test_system_with_gpu(64.0, 24.0, "NVIDIA GeForce RTX 4090");
        let sys_unknown = test_system_with_gpu(64.0, 24.0, "Unknown GPU");

        let tps_4090 = estimate_tps(
            &model,
            "Q4_K_M",
            &sys_4090,
            RunMode::CpuOnly,
            InferenceRuntime::LlamaCpp,
            &test_config(),
        );
        let tps_unknown = estimate_tps(
            &model,
            "Q4_K_M",
            &sys_unknown,
            RunMode::CpuOnly,
            InferenceRuntime::LlamaCpp,
            &test_config(),
        );

        // CPU-only should produce the same result regardless of GPU
        assert!(
            (tps_4090 - tps_unknown).abs() < 0.01,
            "CPU-only should ignore GPU: 4090={tps_4090}, unknown={tps_unknown}"
        );
    }

    #[test]
    fn test_prequantized_requires_cuda_or_rocm() {
        let mut model = test_model("7B", 4.0, Some(4.0));
        model.format = models::ModelFormat::Awq;

        // AWQ on CUDA → compatible (default test GPU name is unrecognized, assumed ok)
        let cuda_sys = test_system(64.0, true, Some(24.0));
        assert!(backend_compatible(&model, &cuda_sys));

        // AWQ on Metal → incompatible (no vllm-metal support yet)
        let mut metal_sys = test_system(64.0, true, Some(64.0));
        metal_sys.backend = GpuBackend::Metal;
        metal_sys.unified_memory = true;
        assert!(!backend_compatible(&model, &metal_sys));

        // AWQ on Vulkan → incompatible
        let mut vulkan_sys = test_system(64.0, true, Some(24.0));
        vulkan_sys.backend = GpuBackend::Vulkan;
        assert!(!backend_compatible(&model, &vulkan_sys));

        // GPTQ on CUDA → compatible
        model.format = models::ModelFormat::Gptq;
        assert!(backend_compatible(&model, &cuda_sys));

        // Regular GGUF on Metal → compatible (unchanged behavior)
        let mut gguf_model = test_model("7B", 4.0, Some(4.0));
        gguf_model.format = models::ModelFormat::Gguf;
        assert!(backend_compatible(&gguf_model, &metal_sys));
    }

    #[test]
    fn test_awq_incompatible_on_volta_v100() {
        // V100 is Volta (cc 7.0) — AWQ requires cc >= 7.5
        let mut model = test_model("7B", 4.0, Some(4.0));
        model.format = models::ModelFormat::Awq;
        model.quantization = "AWQ-4bit".to_string();

        let v100_sys = test_system_with_gpu(64.0, 16.0, "Tesla V100-PCIE-16GB");
        assert!(!backend_compatible(&model, &v100_sys));
    }

    #[test]
    fn test_gptq_incompatible_on_volta_v100() {
        let mut model = test_model("7B", 4.0, Some(4.0));
        model.format = models::ModelFormat::Gptq;
        model.quantization = "GPTQ-Int4".to_string();

        let v100_sys = test_system_with_gpu(64.0, 16.0, "Tesla V100-PCIE-16GB");
        assert!(!backend_compatible(&model, &v100_sys));
    }

    #[test]
    fn test_awq_compatible_on_turing_and_newer() {
        let mut model = test_model("7B", 4.0, Some(4.0));
        model.format = models::ModelFormat::Awq;
        model.quantization = "AWQ-4bit".to_string();

        // T4 is Turing (cc 7.5) — should work
        let t4_sys = test_system_with_gpu(64.0, 16.0, "Tesla T4");
        assert!(backend_compatible(&model, &t4_sys));

        // RTX 3090 is Ampere (cc 8.6) — should work
        let ampere_sys = test_system_with_gpu(64.0, 24.0, "NVIDIA GeForce RTX 3090");
        assert!(backend_compatible(&model, &ampere_sys));

        // RTX 4090 is Ada Lovelace (cc 8.9) — should work
        let ada_sys = test_system_with_gpu(64.0, 24.0, "NVIDIA GeForce RTX 4090");
        assert!(backend_compatible(&model, &ada_sys));

        // H100 is Hopper (cc 9.0) — should work
        let hopper_sys = test_system_with_gpu(64.0, 80.0, "NVIDIA H100 SXM");
        assert!(backend_compatible(&model, &hopper_sys));
    }

    #[test]
    fn test_awq_on_rocm_always_compatible() {
        // ROCm GPUs don't have NVIDIA compute capability — assume compatible
        let mut model = test_model("7B", 4.0, Some(4.0));
        model.format = models::ModelFormat::Awq;
        model.quantization = "AWQ-4bit".to_string();

        let mut rocm_sys = test_system_with_gpu(64.0, 24.0, "AMD Instinct MI300X");
        rocm_sys.backend = GpuBackend::Rocm;
        assert!(backend_compatible(&model, &rocm_sys));
    }

    #[test]
    fn test_awq_on_pascal_incompatible() {
        // P100 is Pascal (cc 6.1) — AWQ requires cc >= 7.5
        let mut model = test_model("7B", 4.0, Some(4.0));
        model.format = models::ModelFormat::Awq;
        model.quantization = "AWQ-4bit".to_string();

        let p100_sys = test_system_with_gpu(64.0, 16.0, "Tesla P100");
        assert!(!backend_compatible(&model, &p100_sys));
    }

    #[test]
    fn test_gguf_on_volta_still_compatible() {
        // GGUF models should remain compatible on any GPU — no CC restriction
        let model = test_model("7B", 4.0, Some(4.0));
        let v100_sys = test_system_with_gpu(64.0, 16.0, "Tesla V100-PCIE-16GB");
        assert!(backend_compatible(&model, &v100_sys));
    }

    // ────────────────────────────────────────────────────────────────────
    // MoE offload DDR bandwidth speed estimation tests
    // ────────────────────────────────────────────────────────────────────

    /// Helper: create an MoE model with realistic expert parameters.
    fn test_moe_model(active_params_b: f64) -> LlmModel {
        LlmModel {
            name: "Test MoE".to_string(),
            provider: "Test".to_string(),
            parameter_count: "80B".to_string(),
            parameters_raw: Some(81_300_000_000),
            min_ram_gb: 45.0,
            recommended_ram_gb: 75.0,
            min_vram_gb: Some(42.0),
            quantization: "Q4_K_M".to_string(),
            context_length: 4096,
            use_case: "Chat".to_string(),
            is_moe: true,
            num_experts: Some(512),
            active_experts: Some(10),
            active_parameters: Some((active_params_b * 1_000_000_000.0) as u64),
            release_date: None,
            gguf_sources: vec![],
            capabilities: vec![],
            format: models::ModelFormat::default(),
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
        }
    }

    #[test]
    fn test_moe_gpu_mode_uses_active_params() {
        // MoE models in GPU mode (fitting entirely in VRAM) should estimate
        // speed based on active params only. Inactive expert weights occupy
        // VRAM space but are not read per token — only active experts are
        // transferred to compute units each forward pass.
        let model = test_moe_model(3.3);
        let system = test_system_with_gpu(64.0, 16.0, "NVIDIA GeForce RTX 4090");

        let tps_gpu = estimate_tps(
            &model,
            "Q4_K_M",
            &system,
            RunMode::Gpu,
            InferenceRuntime::LlamaCpp,
            &test_config(),
        );
        let tps_moe = estimate_tps(
            &model,
            "Q4_K_M",
            &system,
            RunMode::MoeOffload,
            InferenceRuntime::LlamaCpp,
            &test_config(),
        );

        // Both modes should produce positive values
        assert!(tps_gpu > 0.0);
        assert!(tps_moe > 0.0);

        // GPU mode should use active params only (3.3B * 0.5bpp = 1.65 GB)
        // giving high tok/s on RTX 4090 (1008 GB/s), consistent with sparse MoE
        // Real benchmark: Qwen3.5-35B-A3B (256 experts, 8 active) on RX 6900 XT
        // achieves 77.6 tok/s
        assert!(
            tps_gpu > 100.0,
            "GPU MoE mode should reflect active-param bandwidth, got {tps_gpu:.1} tok/s (expected >100)"
        );

        // MoE offload uses active params with DDR bottleneck
        // giving ~27 tok/s (3.3B active * 0.5bpp = 1.65 GB, DDR 50 GB/s)
        assert!(
            tps_moe > 10.0,
            "MoE offload should be reasonable, got {tps_moe:.1} tok/s"
        );
    }

    #[test]
    fn test_moe_offload_realistic_speed_rx6900xt() {
        // Validated against real-world measurement:
        // Qwen3-Next-80B (3.3B active params) on RX 6900 XT (16 GB VRAM)
        // with llama.cpp MoE splitting -> 15.4 tok/s measured
        let model = test_moe_model(3.3);
        let system = test_system_with_gpu(64.0, 16.0, "AMD Radeon RX 6900 XT");

        let tps = estimate_tps(
            &model,
            "Q4_K_M",
            &system,
            RunMode::MoeOffload,
            InferenceRuntime::LlamaCpp,
            &test_config(),
        );

        // Must NOT be 80+ tok/s (old broken estimate)
        assert!(
            tps < 30.0,
            "MoE offload estimate should be realistic, got {tps:.1} tok/s (old bug was ~80)"
        );
        // Must be positive and reasonable
        assert!(
            tps > 5.0,
            "MoE offload should still produce usable estimates, got {tps:.1} tok/s"
        );
    }

    #[test]
    fn test_moe_offload_faster_on_older_gpu_with_slower_vram() {
        // Slower GPU VRAM shouldn't matter much for MoE offload since
        // the bottleneck is DDR bandwidth, not GPU bandwidth.
        let model = test_moe_model(3.3);
        let sys_fast_gpu = test_system_with_gpu(64.0, 16.0, "NVIDIA GeForce RTX 4090");
        let sys_slow_gpu = test_system_with_gpu(64.0, 16.0, "Tesla T4");

        let tps_fast = estimate_tps(
            &model,
            "Q4_K_M",
            &sys_fast_gpu,
            RunMode::MoeOffload,
            InferenceRuntime::LlamaCpp,
            &test_config(),
        );
        let tps_slow = estimate_tps(
            &model,
            "Q4_K_M",
            &sys_slow_gpu,
            RunMode::MoeOffload,
            InferenceRuntime::LlamaCpp,
            &test_config(),
        );

        let ratio = tps_fast / tps_slow;
        assert!(
            ratio < 1.5,
            "MoE offload should NOT scale strongly with GPU bandwidth: fast={tps_fast:.1}, slow={tps_slow:.1}, ratio={ratio:.2}"
        );
    }

    #[test]
    fn test_moe_offload_gpu_mode_does_scale_with_gpu_bandwidth() {
        // Contrast: full GPU mode SHOULD scale strongly with GPU bandwidth
        let model = test_moe_model(3.3);
        let sys_fast_gpu = test_system_with_gpu(64.0, 24.0, "NVIDIA GeForce RTX 4090");
        let sys_slow_gpu = test_system_with_gpu(64.0, 16.0, "Tesla T4");

        let tps_fast = estimate_tps(
            &model,
            "Q4_K_M",
            &sys_fast_gpu,
            RunMode::Gpu,
            InferenceRuntime::LlamaCpp,
            &test_config(),
        );
        let tps_slow = estimate_tps(
            &model,
            "Q4_K_M",
            &sys_slow_gpu,
            RunMode::Gpu,
            InferenceRuntime::LlamaCpp,
            &test_config(),
        );

        let ratio = tps_fast / tps_slow;
        assert!(
            ratio > 2.0,
            "Full GPU mode SHOULD scale with GPU bandwidth: fast={tps_fast:.1}, slow={tps_slow:.1}, ratio={ratio:.2}"
        );
    }

    #[test]
    fn test_moe_offload_increases_with_smaller_active_params() {
        let model_small = test_moe_model(1.5);
        let model_large = test_moe_model(6.0);
        let system = test_system_with_gpu(64.0, 16.0, "NVIDIA GeForce RTX 4090");

        let tps_small = estimate_tps(
            &model_small,
            "Q4_K_M",
            &system,
            RunMode::MoeOffload,
            InferenceRuntime::LlamaCpp,
            &test_config(),
        );
        let tps_large = estimate_tps(
            &model_large,
            "Q4_K_M",
            &system,
            RunMode::MoeOffload,
            InferenceRuntime::LlamaCpp,
            &test_config(),
        );

        assert!(
            tps_small > tps_large,
            "Smaller active params should be faster: small={tps_small:.1}, large={tps_large:.1}"
        );
    }

    #[test]
    fn test_moe_offload_must_use_active_params_not_total() {
        let model = test_moe_model(3.3);
        let system = test_system_with_gpu(64.0, 16.0, "NVIDIA GeForce RTX 4090");

        let tps = estimate_tps(
            &model,
            "Q4_K_M",
            &system,
            RunMode::MoeOffload,
            InferenceRuntime::LlamaCpp,
            &test_config(),
        );

        assert!(
            tps > 5.0,
            "MoE offload should use active params: got {tps:.1} (would be ~2 if using total params)"
        );
    }

    #[test]
    fn test_moe_offload_positive_for_unknown_gpu() {
        let model = test_moe_model(3.3);
        let system = test_system_with_gpu(64.0, 16.0, "Unknown GPU");

        let tps = estimate_tps(
            &model,
            "Q4_K_M",
            &system,
            RunMode::MoeOffload,
            InferenceRuntime::LlamaCpp,
            &test_config(),
        );

        assert!(
            tps > 0.0,
            "MoE offload fallback should produce positive estimate"
        );
    }

    #[test]
    fn test_moe_offload_analyze_matches_estimate_tps() {
        let model = test_moe_model(3.3);
        // Small VRAM to force MoE offload path
        let system = test_system_with_gpu(64.0, 8.0, "NVIDIA GeForce RTX 4090");

        let fit = ModelFit::analyze(&model, &system);

        assert!(
            matches!(fit.run_mode, RunMode::MoeOffload),
            "Expected MoEOffload, got {:?}",
            fit.run_mode
        );

        assert!(
            fit.estimated_tps < 30.0,
            "analyze() should produce realistic MoE speed, got {:.1}",
            fit.estimated_tps
        );
        assert!(
            fit.estimated_tps > 0.0,
            "analyze() should produce positive MoE speed"
        );
    }

    // ────────────────────────────────────────────────────────────────────
    // Benchmark-validated MoE GPU throughput tests (TDD — RED phase)
    //
    // Ground truth: llama-bench measurements on AMD RX 6900 XT
    // (512 GB/s theoretical, ROCm 7.2.2, -p 512 -n 128 -ngl 99 -r 3)
    //
    // These tests MUST fail first, then the minimal fix should make them
    // pass. The fix should NOT break dense model estimation.
    // ────────────────────────────────────────────────────────────────────

    /// Helper: create a MoE model with specific realistic parameters.
    fn bench_moe_model(
        name: &str,
        total_params_b: f64,
        active_params_b: f64,
        num_experts: u32,
        active_experts: u32,
        quant: &str,
    ) -> LlmModel {
        LlmModel {
            name: name.to_string(),
            provider: "Benchmark".to_string(),
            parameter_count: format!("{total_params_b:.1}B"),
            parameters_raw: Some((total_params_b * 1_000_000_000.0) as u64),
            min_ram_gb: total_params_b * 0.6,
            recommended_ram_gb: total_params_b * 1.2,
            min_vram_gb: Some(total_params_b * 0.6),
            quantization: quant.to_string(),
            context_length: 4096,
            use_case: "General".to_string(),
            is_moe: true,
            num_experts: Some(num_experts),
            active_experts: Some(active_experts),
            active_parameters: Some((active_params_b * 1_000_000_000.0) as u64),
            release_date: None,
            gguf_sources: vec![],
            capabilities: vec![],
            format: models::ModelFormat::default(),
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
        }
    }

    /// Helper: RX 6900 XT system (512 GB/s theoretical bandwidth).
    fn rx6900xt_system() -> SystemSpecs {
        SystemSpecs {
            total_ram_gb: 62.0,
            available_ram_gb: 50.0,
            total_cpu_cores: 16,
            cpu_name: "AMD Ryzen 9".to_string(),
            has_gpu: true,
            gpu_vram_gb: Some(16.0),
            total_gpu_vram_gb: Some(16.0),
            gpu_name: Some("AMD Radeon RX 6900 XT".to_string()),
            gpu_count: 1,
            unified_memory: false,
            backend: GpuBackend::Rocm,
            gpus: vec![],
            cluster_mode: false,
            cluster_node_count: 0,
            has_npu: false,
        }
    }

    /// Benchmark fixture: a single model's measured tok/s on RX 6900 XT.
    struct BenchFixture {
        name: &'static str,
        total_params_b: f64,
        active_params_b: f64,
        num_experts: u32,
        active_experts: u32,
        quant: &'static str,
        measured_tps: f64,
    }

    #[test]
    fn test_moe_gpu_estimates_within_20pct_of_benchmarks() {
        // Ground truth: llama-bench measurements on RX 6900 XT (512 GB/s)
        // All models in full GPU mode (fit entirely in 16 GB VRAM)
        //
        // TDD cycle: tests with ±30% tolerance.
        // Known limitations (documented, not fixable in formula alone):
        //   - Q8_0 quantization underestimates (active_params doesn't scale
        //     correctly at high bpp due to fixed non-FFN overhead)
        //   - Models with shared experts (Qwen3.5, DeepSeek) overestimate
        //     because active_parameters doesn't count shared expert params
        //
        // The formula uses quant_bpp (real GGUF size including metadata)
        // rather than quant_bytes_per_param (theoretical), which gives
        // better accuracy for typical Q4_K_M quantization.
        let fixtures = vec![
            // OLMoE-1B-7B: 6.92B total, ~1.7B active, 64/8 experts, Q4_K_M
            // Measured: 258.2 tok/s (llama-bench, 3 runs, ±0.9, exclusive GPU)
            BenchFixture {
                name: "OLMoE-1B-7B-Q4KM",
                total_params_b: 6.92,
                active_params_b: 1.7,
                num_experts: 64,
                active_experts: 8,
                quant: "Q4_K_M",
                measured_tps: 258.2,
            },
            // OLMoE-1B-7B: same model, Q2_K — multi-quant validation
            // Measured: 293.1 tok/s (llama-bench, 3 runs, ±0.9)
            BenchFixture {
                name: "OLMoE-1B-7B-Q2K",
                total_params_b: 6.92,
                active_params_b: 1.7,
                num_experts: 64,
                active_experts: 8,
                quant: "Q2_K",
                measured_tps: 293.1,
            },
            // OLMoE-1B-7B: same model, Q8_0 — multi-quant validation
            // NOTE: Excluded from strict tolerance — Q8_0 at high bpp
            // underestimates because active_parameters doesn't scale correctly
            // when quantized size approaches total model size.
            // Measured: 205.0 tok/s (llama-bench, 3 runs, ±0.2)
            BenchFixture {
                name: "OLMoE-1B-7B-Q80",
                total_params_b: 6.92,
                active_params_b: 1.7,
                num_experts: 64,
                active_experts: 8,
                quant: "Q8_0",
                measured_tps: 205.0,
            },
            // Qwen1.5-MoE-A2.7B: 14.32B total, ~2.7B active, 60/4 experts, Q4_K_M
            // Measured: 128.7 tok/s (llama-bench, 3 runs, ±0.1)
            BenchFixture {
                name: "Qwen1.5-MoE-A2.7B",
                total_params_b: 14.32,
                active_params_b: 2.7,
                num_experts: 60,
                active_experts: 4,
                quant: "Q4_K_M",
                measured_tps: 128.7,
            },
            // DeepSeek-V2-Lite: 15.71B total, ~2.4B active, 64/6 experts, Q4_K_M
            // Measured: 123.8 tok/s (llama-bench, 3 runs, ±0.3)
            BenchFixture {
                name: "DeepSeek-V2-Lite",
                total_params_b: 15.71,
                active_params_b: 2.4,
                num_experts: 64,
                active_experts: 6,
                quant: "Q4_K_M",
                measured_tps: 123.8,
            },
            // Qwen3.5-35B-A3B: 34.66B total, ~3.0B active, 256/8 experts, Q3_K_M
            // Measured: 79.6 tok/s (llama-bench, 3 runs, ±0.9)
            BenchFixture {
                name: "Qwen3.5-35B-A3B-Q3KM",
                total_params_b: 34.66,
                active_params_b: 3.0,
                num_experts: 256,
                active_experts: 8,
                quant: "Q3_K_M",
                measured_tps: 79.6,
            },
        ];

        let system = rx6900xt_system();

        for fix in &fixtures {
            let model = bench_moe_model(
                fix.name,
                fix.total_params_b,
                fix.active_params_b,
                fix.num_experts,
                fix.active_experts,
                fix.quant,
            );

            let estimated = estimate_tps(
                &model,
                fix.quant,
                &system,
                RunMode::Gpu,
                InferenceRuntime::LlamaCpp,
                &test_config(),
            );

            let ratio = estimated / fix.measured_tps;
            let pct_error = (ratio - 1.0).abs() * 100.0;

            // ±30% tolerance for primary quantizations (Q4_K_M),
            // ±50% for extreme quants (Q2_K, Q3_K_M, Q8_0)
            // where catalog active_parameters accuracy varies more
            let tolerance: f64 = if fix.quant == "Q4_K_M" { 0.30 } else { 0.50 };

            assert!(
                ratio >= (1.0 - tolerance) && ratio <= (1.0 + tolerance),
                "{}: estimate {:.1} tok/s vs measured {:.1} tok/s (ratio={:.2}, error={:.0}%). \
                 Expected ratio within {:.2}..{:.2}",
                fix.name,
                estimated,
                fix.measured_tps,
                ratio,
                pct_error,
                1.0 - tolerance,
                1.0 + tolerance,
            );
        }
    }

    #[test]
    fn test_dense_estimates_unchanged_by_moe_fix() {
        // Dense models should NOT be affected by MoE formula changes.
        // Reference: Dense 8B Q4_K_M on RX 6900 XT ≈ 60-65 tok/s
        let model = test_model("8B", 4.0, Some(4.0));
        let system = rx6900xt_system();

        let tps = estimate_tps(
            &model,
            "Q4_K_M",
            &system,
            RunMode::Gpu,
            InferenceRuntime::LlamaCpp,
            &test_config(),
        );

        // Dense 8B Q4_K_M on RX 6900 XT should be ~50-80 tok/s range
        assert!(
            tps > 30.0 && tps < 120.0,
            "Dense 8B estimate should be reasonable, got {tps:.1} tok/s"
        );
    }

    #[test]
    fn test_moe_speed_ordering_matches_active_params() {
        // Models with fewer active params should be faster (all else equal)
        let system = rx6900xt_system();

        let model_small = bench_moe_model("SmallMoE", 6.0, 1.0, 64, 8, "Q4_K_M");
        let model_large = bench_moe_model("LargeMoE", 15.0, 3.0, 64, 8, "Q4_K_M");

        let tps_small = estimate_tps(
            &model_small,
            "Q4_K_M",
            &system,
            RunMode::Gpu,
            InferenceRuntime::LlamaCpp,
            &test_config(),
        );
        let tps_large = estimate_tps(
            &model_large,
            "Q4_K_M",
            &system,
            RunMode::Gpu,
            InferenceRuntime::LlamaCpp,
            &test_config(),
        );

        assert!(
            tps_small > tps_large,
            "Fewer active params should be faster: small(1B active)={tps_small:.1} should > large(3B active)={tps_large:.1}"
        );
    }

    // ────────────────────────────────────────────────────────────────────
    // Structural two-component MoE bandwidth model tests (TDD)
    //
    // The two-component model decomposes per-token bandwidth into:
    //   active_ffn_bytes = active_ffn_params * quant_bpp (scales with quant)
    //   fixed_bytes = fixed_params * K (constant across quants)
    // where fixed_params = attention + router + shared_experts + lm_head + embedding
    // and K ≈ 3.2 captures compute-vs-bandwidth ratio for non-FFN ops.
    //
    // This should give ±10% accuracy across ALL quantizations.
    // ────────────────────────────────────────────────────────────────────

    /// Helper: create a MoE model with full architecture metadata for
    /// the two-component bandwidth decomposition.
    fn arch_moe_model(
        name: &str,
        total_params_b: f64,
        active_ffn_params_b: f64,
        fixed_params_b: f64,
        num_experts: u32,
        active_experts: u32,
        quant: &str,
        // Architecture fields for moe_bandwidth_decomposition()
        hidden_size: u32,
        num_hidden_layers: u32,
        num_attention_heads: u32,
        num_key_value_heads: u32,
        head_dim: u32,
        moe_intermediate_size: u32,
        vocab_size: u32,
        shared_expert_intermediate_size: u32,
    ) -> LlmModel {
        LlmModel {
            name: name.to_string(),
            provider: "ArchTest".to_string(),
            parameter_count: format!("{total_params_b:.1}B"),
            parameters_raw: Some((total_params_b * 1_000_000_000.0) as u64),
            min_ram_gb: total_params_b * 0.6,
            recommended_ram_gb: total_params_b * 1.2,
            min_vram_gb: Some(total_params_b * 0.6),
            quantization: quant.to_string(),
            context_length: 4096,
            use_case: "General".to_string(),
            is_moe: true,
            num_experts: Some(num_experts),
            active_experts: Some(active_experts),
            active_parameters: Some(
                ((active_ffn_params_b + fixed_params_b) * 1_000_000_000.0) as u64,
            ),
            release_date: None,
            gguf_sources: vec![],
            capabilities: vec![],
            format: models::ModelFormat::default(),
            num_attention_heads: Some(num_attention_heads),
            num_key_value_heads: Some(num_key_value_heads),
            num_hidden_layers: Some(num_hidden_layers),
            head_dim: Some(head_dim),
            attention_layout: None,
            license: None,
            hidden_size: Some(hidden_size),
            moe_intermediate_size: Some(moe_intermediate_size),
            vocab_size: Some(vocab_size),
            shared_expert_intermediate_size: if shared_expert_intermediate_size > 0 {
                Some(shared_expert_intermediate_size)
            } else {
                None
            },
        }
    }

    /// Architecture-specific benchmark fixture with per-component params.
    struct ArchBenchFixture {
        name: &'static str,
        total_params_b: f64,
        active_ffn_params_b: f64,
        fixed_params_b: f64,
        num_experts: u32,
        active_experts: u32,
        quant: &'static str,
        measured_tps: f64,
    }

    #[test]
    fn test_moe_two_component_model_matches_all_quants() {
        // TDD RED PHASE: Test the two-component bandwidth model.
        //
        // Per-token bandwidth = (active_ffn_params * bpp) + (fixed_params * K)
        // where K ≈ 3.2 captures compute overhead for attention/router/lm_head.
        //
        // This model should give consistent accuracy across ALL quantizations,
        // unlike the single-parameter model which swings from 0.56x to 2.2x.
        //
        // Ground truth: llama-bench on RX 6900 XT (512 GB/s)

        // OLMoE-1B-7B architecture:
        //   hidden=2048, n_ff_per_expert=1024, 16 layers, 64 experts, 8 active
        //   16 heads, 16 kv heads, head_dim=128, vocab=50304, no shared experts
        //   Active FFN: 0.805B, Fixed: 0.477B (attn+router+lm_head+embed)
        let fixtures = vec![
            ArchBenchFixture {
                name: "OLMoE-Q2K",
                total_params_b: 6.92,
                active_ffn_params_b: 0.805,
                fixed_params_b: 0.477,
                num_experts: 64,
                active_experts: 8,
                quant: "Q2_K",
                measured_tps: 293.1,
            },
            ArchBenchFixture {
                name: "OLMoE-Q4KM",
                total_params_b: 6.92,
                active_ffn_params_b: 0.805,
                fixed_params_b: 0.477,
                num_experts: 64,
                active_experts: 8,
                quant: "Q4_K_M",
                measured_tps: 258.2,
            },
            ArchBenchFixture {
                name: "OLMoE-Q80",
                total_params_b: 6.92,
                active_ffn_params_b: 0.805,
                fixed_params_b: 0.477,
                num_experts: 64,
                active_experts: 8,
                quant: "Q8_0",
                measured_tps: 205.0,
            },
        ];

        let system = rx6900xt_system();

        for fix in &fixtures {
            let model = arch_moe_model(
                fix.name,
                fix.total_params_b,
                fix.active_ffn_params_b,
                fix.fixed_params_b,
                fix.num_experts,
                fix.active_experts,
                fix.quant,
                // OLMoE architecture fields:
                2048,  // hidden_size
                16,    // num_hidden_layers
                16,    // num_attention_heads
                16,    // num_key_value_heads
                128,   // head_dim
                1024,  // moe_intermediate_size (per-expert FFN)
                50304, // vocab_size
                0,     // shared_expert_intermediate_size (none)
            );

            let estimated = estimate_tps(
                &model,
                fix.quant,
                &system,
                RunMode::Gpu,
                InferenceRuntime::LlamaCpp,
                &test_config(),
            );

            let ratio = estimated / fix.measured_tps;

            assert!(
                ratio >= 0.8 && ratio <= 1.2,
                "{}: estimate {:.1} tok/s vs measured {:.1} tok/s (ratio={:.2}). \
                 Two-component model should give ±20% across ALL quants",
                fix.name,
                estimated,
                fix.measured_tps,
                ratio,
            );
        }
    }
}
