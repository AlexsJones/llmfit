use crate::fit::{FitLevel, RunMode};
use crate::hardware::{GpuBackend, SystemSpecs};
use crate::models::{KvQuant, LlmModel, quant_speed_multiplier};

const SUPPORTED_QUANTS: &[&str] = &[
    "F32",
    "F16",
    "BF16",
    "Q8_0",
    "Q6_K",
    "Q5_K_M",
    "Q4_K_M",
    "Q4_0",
    "Q3_K_M",
    "Q2_K",
    "mlx-8bit",
    "mlx-4bit",
    "AWQ-4bit",
    "AWQ-8bit",
    "GPTQ-Int4",
    "GPTQ-Int8",
];

#[derive(Debug, Clone, serde::Serialize)]
pub struct PlanRequest {
    pub context: u32,
    pub quant: Option<String>,
    pub target_tps: Option<f64>,
    /// KV cache element representation. Defaults to fp16.
    #[serde(default)]
    pub kv_quant: Option<KvQuant>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct HardwareEstimate {
    pub vram_gb: Option<f64>,
    pub ram_gb: f64,
    pub cpu_cores: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanRunPath {
    Gpu,
    CpuOffload,
    CpuOnly,
}

impl PlanRunPath {
    pub fn label(&self) -> &'static str {
        match self {
            PlanRunPath::Gpu => "GPU",
            PlanRunPath::CpuOffload => "CPU offload",
            PlanRunPath::CpuOnly => "CPU-only",
        }
    }

    fn run_mode(self) -> RunMode {
        match self {
            PlanRunPath::Gpu => RunMode::Gpu,
            PlanRunPath::CpuOffload => RunMode::CpuOffload,
            PlanRunPath::CpuOnly => RunMode::CpuOnly,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PathEstimate {
    pub path: PlanRunPath,
    pub feasible: bool,
    pub minimum: Option<HardwareEstimate>,
    pub recommended: Option<HardwareEstimate>,
    pub estimated_tps: Option<f64>,
    pub fit_level: Option<FitLevel>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct UpgradeDelta {
    pub resource: String,
    pub add_gb: Option<f64>,
    pub add_cores: Option<usize>,
    pub target_fit: Option<FitLevel>,
    pub path: PlanRunPath,
    pub description: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PlanCurrentStatus {
    pub fit_level: FitLevel,
    pub run_mode: RunMode,
    pub estimated_tps: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct KvQuantAlternative {
    pub kv_quant: KvQuant,
    /// Total memory required (weights + KV + overhead) at this KV quant.
    pub memory_required_gb: f64,
    /// KV cache size only, for clarity in the UI.
    pub kv_cache_gb: f64,
    /// Savings vs the fp16 baseline, expressed as a fraction (0.0 to 1.0).
    pub savings_fraction: f64,
    /// Optional human readable note (e.g. TurboQuant compressibility caveat).
    pub note: Option<String>,
    /// True if the option is supported by the resolved runtime + backend.
    pub supported: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PlanEstimate {
    pub estimate_notice: String,
    pub model_name: String,
    pub provider: String,
    pub context: u32,
    pub quantization: String,
    pub kv_quant: KvQuant,
    pub target_tps: Option<f64>,
    pub minimum: HardwareEstimate,
    pub recommended: HardwareEstimate,
    pub run_paths: Vec<PathEstimate>,
    pub current: PlanCurrentStatus,
    pub upgrade_deltas: Vec<UpgradeDelta>,
    /// "What if" rows showing each KV quant option's memory footprint and
    /// savings vs the fp16 baseline. Surfaced by phase 5.
    #[serde(default)]
    pub kv_alternatives: Vec<KvQuantAlternative>,
}

pub fn normalize_quant(quant: &str) -> Option<String> {
    let trimmed = quant.trim();
    if trimmed.is_empty() {
        return None;
    }

    if trimmed.eq_ignore_ascii_case("mlx-4bit") {
        return Some("mlx-4bit".to_string());
    }
    if trimmed.eq_ignore_ascii_case("mlx-8bit") {
        return Some("mlx-8bit".to_string());
    }

    // AWQ quantization formats
    if trimmed.eq_ignore_ascii_case("awq-4bit") {
        return Some("AWQ-4bit".to_string());
    }
    if trimmed.eq_ignore_ascii_case("awq-8bit") {
        return Some("AWQ-8bit".to_string());
    }
    // GPTQ quantization formats
    if trimmed.eq_ignore_ascii_case("gptq-int4") {
        return Some("GPTQ-Int4".to_string());
    }
    if trimmed.eq_ignore_ascii_case("gptq-int8") {
        return Some("GPTQ-Int8".to_string());
    }

    let upper = trimmed.to_uppercase();
    if SUPPORTED_QUANTS.contains(&upper.as_str()) {
        Some(upper)
    } else {
        None
    }
}

fn estimate_tps(
    model: &LlmModel,
    quant: &str,
    backend: GpuBackend,
    path: PlanRunPath,
    cpu_cores: usize,
) -> f64 {
    estimate_tps_with_gpu(model, quant, backend, path, cpu_cores, None)
}

/// Bandwidth-aware tok/s estimation (mirrors fit.rs logic).
/// When `gpu_name` is provided and recognized, uses memory-bandwidth-based
/// estimation instead of fixed constants.
fn estimate_tps_with_gpu(
    model: &LlmModel,
    quant: &str,
    backend: GpuBackend,
    path: PlanRunPath,
    cpu_cores: usize,
    gpu_name: Option<&str>,
) -> f64 {
    use crate::hardware::gpu_memory_bandwidth_gbps;
    use crate::models::quant_bytes_per_param;

    let params = model.params_b().max(0.1);

    // Bandwidth-based estimation when GPU is recognized
    if path != PlanRunPath::CpuOnly
        && let Some(name) = gpu_name
        && let Some(bw) = gpu_memory_bandwidth_gbps(name)
    {
        let model_gb = params * quant_bytes_per_param(quant);
        let efficiency = 0.55;
        let raw_tps = (bw / model_gb) * efficiency;

        // CpuOnly is unreachable here (guarded above). The default fallback handles
        // any future PlanRunPath variants gracefully.
        let mode_factor = match path {
            PlanRunPath::Gpu => 1.0,
            PlanRunPath::CpuOffload => 0.5,
            _ => 1.0,
        };

        return (raw_tps * mode_factor).max(0.1);
    }

    // Fallback: fixed-constant approach
    let k: f64 = match backend {
        GpuBackend::Metal => 160.0,
        GpuBackend::Cuda => 220.0,
        GpuBackend::Rocm => 180.0,
        GpuBackend::Vulkan => 150.0,
        GpuBackend::Sycl => 100.0,
        GpuBackend::CpuArm => 90.0,
        GpuBackend::CpuX86 => 70.0,
        GpuBackend::Ascend => 390.0,
        GpuBackend::AmdNpu => 130.0,
    };

    let mut base = (k / params) * quant_speed_multiplier(quant);

    if cpu_cores >= 8 {
        base *= 1.1;
    }

    match path {
        PlanRunPath::Gpu => {}
        PlanRunPath::CpuOffload => base *= 0.5,
        PlanRunPath::CpuOnly => {
            let cpu_k = if cfg!(target_arch = "aarch64") {
                90.0
            } else {
                70.0
            };
            base = (cpu_k / params) * quant_speed_multiplier(quant);
            if cpu_cores >= 8 {
                base *= 1.1;
            }
        }
    }

    base.max(0.1)
}

fn fit_level_for(
    path: PlanRunPath,
    required_gb: f64,
    available_gb: f64,
    recommended_gb: f64,
) -> FitLevel {
    if required_gb > available_gb {
        return FitLevel::TooTight;
    }

    match path {
        PlanRunPath::Gpu => {
            if recommended_gb <= available_gb {
                FitLevel::Perfect
            } else if available_gb >= required_gb * 1.2 {
                FitLevel::Good
            } else {
                FitLevel::Marginal
            }
        }
        PlanRunPath::CpuOffload => {
            if available_gb >= required_gb * 1.2 {
                FitLevel::Good
            } else {
                FitLevel::Marginal
            }
        }
        PlanRunPath::CpuOnly => FitLevel::Marginal,
    }
}

fn minimum_cores_for_target(
    model: &LlmModel,
    quant: &str,
    backend: GpuBackend,
    path: PlanRunPath,
    target_tps: Option<f64>,
) -> Option<usize> {
    let Some(target) = target_tps else {
        return Some(4);
    };

    for cores in 1..=64 {
        let tps = estimate_tps(model, quant, backend, path, cores);
        if tps >= target {
            return Some(cores);
        }
    }

    None
}

fn default_gpu_backend(system: &SystemSpecs) -> GpuBackend {
    if system.has_gpu {
        system.backend
    } else {
        GpuBackend::Cuda
    }
}

fn evaluate_current(
    model: &LlmModel,
    quant: &str,
    context: u32,
    kv_quant: KvQuant,
    target_tps: Option<f64>,
    system: &SystemSpecs,
) -> PlanCurrentStatus {
    let model_mem = model.estimate_memory_gb_with_kv(quant, context, kv_quant);
    let gpu_vram = system
        .total_gpu_vram_gb
        .or(system.gpu_vram_gb)
        .unwrap_or(0.0);

    let mut candidates: Vec<(FitLevel, PlanRunPath, f64)> = Vec::new();

    if system.has_gpu && gpu_vram > 0.0 {
        let gpu_fit = fit_level_for(
            PlanRunPath::Gpu,
            model_mem,
            gpu_vram,
            model.recommended_ram_gb,
        );
        let gpu_name = system.gpu_name.as_deref();
        let gpu_tps = estimate_tps_with_gpu(
            model,
            quant,
            system.backend,
            PlanRunPath::Gpu,
            system.total_cpu_cores,
            gpu_name,
        );
        if target_tps.is_none_or(|t| gpu_tps >= t) {
            candidates.push((gpu_fit, PlanRunPath::Gpu, gpu_tps));
        }

        if !system.unified_memory {
            let offload_fit = fit_level_for(
                PlanRunPath::CpuOffload,
                model_mem,
                system.available_ram_gb,
                model.recommended_ram_gb,
            );
            let offload_tps = estimate_tps_with_gpu(
                model,
                quant,
                system.backend,
                PlanRunPath::CpuOffload,
                system.total_cpu_cores,
                gpu_name,
            );
            if target_tps.is_none_or(|t| offload_tps >= t) {
                candidates.push((offload_fit, PlanRunPath::CpuOffload, offload_tps));
            }
        }
    }

    let cpu_fit = fit_level_for(
        PlanRunPath::CpuOnly,
        model_mem,
        system.available_ram_gb,
        model.recommended_ram_gb,
    );
    let cpu_tps = estimate_tps(
        model,
        quant,
        system.backend,
        PlanRunPath::CpuOnly,
        system.total_cpu_cores,
    );
    if target_tps.is_none_or(|t| cpu_tps >= t) {
        candidates.push((cpu_fit, PlanRunPath::CpuOnly, cpu_tps));
    }

    candidates.sort_by(|a, b| {
        let rank = |fit: FitLevel| match fit {
            FitLevel::Perfect => 4,
            FitLevel::Good => 3,
            FitLevel::Marginal => 2,
            FitLevel::TooTight => 1,
        };
        rank(b.0).cmp(&rank(a.0)).then_with(|| {
            let p = |path: PlanRunPath| match path {
                PlanRunPath::Gpu => 3,
                PlanRunPath::CpuOffload => 2,
                PlanRunPath::CpuOnly => 1,
            };
            p(b.1).cmp(&p(a.1))
        })
    });

    if let Some((fit_level, path, tps)) = candidates.first() {
        PlanCurrentStatus {
            fit_level: *fit_level,
            run_mode: path.run_mode(),
            estimated_tps: *tps,
        }
    } else {
        PlanCurrentStatus {
            fit_level: FitLevel::TooTight,
            run_mode: RunMode::CpuOnly,
            estimated_tps: 0.0,
        }
    }
}

fn build_path_estimate(
    model: &LlmModel,
    quant: &str,
    context: u32,
    kv_quant: KvQuant,
    target_tps: Option<f64>,
    path: PlanRunPath,
    system: &SystemSpecs,
) -> PathEstimate {
    let model_mem = model.estimate_memory_gb_with_kv(quant, context, kv_quant);
    let backend = default_gpu_backend(system);
    let mut notes = vec![];

    let min_cores = match minimum_cores_for_target(model, quant, backend, path, target_tps) {
        Some(c) => c,
        None => {
            return PathEstimate {
                path,
                feasible: false,
                minimum: None,
                recommended: None,
                estimated_tps: None,
                fit_level: None,
                notes: vec![
                    "Target TPS is not reachable under current speed heuristics".to_string(),
                ],
            };
        }
    };

    let recommended_cores = min_cores.max(8);

    let gpu_name = system.gpu_name.as_deref();

    match path {
        PlanRunPath::Gpu => {
            let min_vram = model_mem;
            let rec_vram = model.recommended_ram_gb.max(model_mem * 1.2);
            let min_ram = (model_mem * 0.2).max(8.0);
            let rec_ram = (min_ram * 1.25).max(12.0);
            let tps = estimate_tps_with_gpu(model, quant, backend, path, min_cores, gpu_name);

            let fit = fit_level_for(path, min_vram, min_vram, model.recommended_ram_gb);
            notes.push(
                "Estimated from quant/context memory and fit headroom thresholds".to_string(),
            );

            PathEstimate {
                path,
                feasible: true,
                minimum: Some(HardwareEstimate {
                    vram_gb: Some(min_vram),
                    ram_gb: min_ram,
                    cpu_cores: min_cores,
                }),
                recommended: Some(HardwareEstimate {
                    vram_gb: Some(rec_vram),
                    ram_gb: rec_ram,
                    cpu_cores: recommended_cores,
                }),
                estimated_tps: Some(tps),
                fit_level: Some(fit),
                notes,
            }
        }
        PlanRunPath::CpuOffload => {
            if system.unified_memory {
                return PathEstimate {
                    path,
                    feasible: false,
                    minimum: None,
                    recommended: None,
                    estimated_tps: None,
                    fit_level: None,
                    notes: vec!["CPU offload is skipped on unified-memory systems".to_string()],
                };
            }

            let min_vram = 2.0;
            let rec_vram = 4.0;
            let min_ram = model_mem;
            let rec_ram = model_mem * 1.2;
            let fit = fit_level_for(path, min_ram, min_ram, model.recommended_ram_gb);
            let tps = estimate_tps_with_gpu(model, quant, backend, path, min_cores, gpu_name);
            notes.push("RAM is the primary memory pool for CPU offload".to_string());

            PathEstimate {
                path,
                feasible: true,
                minimum: Some(HardwareEstimate {
                    vram_gb: Some(min_vram),
                    ram_gb: min_ram,
                    cpu_cores: min_cores,
                }),
                recommended: Some(HardwareEstimate {
                    vram_gb: Some(rec_vram),
                    ram_gb: rec_ram,
                    cpu_cores: recommended_cores,
                }),
                estimated_tps: Some(tps),
                fit_level: Some(fit),
                notes,
            }
        }
        PlanRunPath::CpuOnly => {
            let min_ram = model_mem;
            let rec_ram = model_mem * 1.2;
            let fit = fit_level_for(path, min_ram, min_ram, model.recommended_ram_gb);
            let tps = estimate_tps(model, quant, GpuBackend::CpuX86, path, min_cores);
            notes.push(
                "CPU-only fit is always capped at Marginal in current heuristics".to_string(),
            );

            PathEstimate {
                path,
                feasible: true,
                minimum: Some(HardwareEstimate {
                    vram_gb: None,
                    ram_gb: min_ram,
                    cpu_cores: min_cores,
                }),
                recommended: Some(HardwareEstimate {
                    vram_gb: None,
                    ram_gb: rec_ram,
                    cpu_cores: recommended_cores,
                }),
                estimated_tps: Some(tps),
                fit_level: Some(fit),
                notes,
            }
        }
    }
}

pub fn estimate_model_plan(
    model: &LlmModel,
    request: &PlanRequest,
    system: &SystemSpecs,
) -> Result<PlanEstimate, String> {
    if request.context == 0 {
        return Err("--context must be greater than 0".to_string());
    }
    if let Some(target) = request.target_tps
        && target <= 0.0
    {
        return Err("--target-tps must be greater than 0".to_string());
    }

    let quant = if let Some(ref q) = request.quant {
        normalize_quant(q).ok_or_else(|| format!("Unsupported quantization '{}'.", q))?
    } else {
        model.quantization.clone()
    };

    let kv_quant = request.kv_quant.unwrap_or_default();

    // TurboQuant gating: only valid on vLLM + CUDA. We don't have an
    // explicit "runtime" field on PlanRequest yet, so use the system
    // backend as a proxy. The CLI passes through `force_runtime`
    // separately on `recommend`/`fit` but `plan` is single-model so we
    // gate on backend == Cuda. If the user really wants to model TQ on
    // a non-CUDA box for planning purposes they can override --memory.
    if kv_quant == KvQuant::TurboQuant && system.backend != GpuBackend::Cuda {
        return Err("TurboQuant KV cache is only supported on vLLM + CUDA. \
             It is not in upstream vLLM yet (see 0xSero/turboquant). \
             Use --kv-quant fp8 / q8_0 / q4_0 for llama.cpp backends."
            .to_string());
    }

    let context = request.context;
    let run_paths = vec![
        build_path_estimate(
            model,
            &quant,
            context,
            kv_quant,
            request.target_tps,
            PlanRunPath::Gpu,
            system,
        ),
        build_path_estimate(
            model,
            &quant,
            context,
            kv_quant,
            request.target_tps,
            PlanRunPath::CpuOffload,
            system,
        ),
        build_path_estimate(
            model,
            &quant,
            context,
            kv_quant,
            request.target_tps,
            PlanRunPath::CpuOnly,
            system,
        ),
    ];

    let current = evaluate_current(model, &quant, context, kv_quant, request.target_tps, system);
    let kv_alternatives = compute_kv_alternatives(model, &quant, context, system);

    let preferred = run_paths
        .iter()
        .find(|p| p.path == PlanRunPath::Gpu && p.feasible)
        .or_else(|| {
            run_paths
                .iter()
                .find(|p| p.path == PlanRunPath::CpuOffload && p.feasible)
        })
        .or_else(|| {
            run_paths
                .iter()
                .find(|p| p.path == PlanRunPath::CpuOnly && p.feasible)
        })
        .ok_or_else(|| "No feasible run path found for this configuration".to_string())?;

    let minimum = preferred
        .minimum
        .clone()
        .ok_or_else(|| "Missing minimum estimate".to_string())?;
    let recommended = preferred
        .recommended
        .clone()
        .ok_or_else(|| "Missing recommended estimate".to_string())?;

    let mut upgrade_deltas = Vec::new();

    let current_vram = system
        .total_gpu_vram_gb
        .or(system.gpu_vram_gb)
        .unwrap_or(0.0);
    if let Some(gpu_path) = run_paths.iter().find(|p| p.path == PlanRunPath::Gpu)
        && let Some(min_hw) = &gpu_path.minimum
    {
        let add_good = (min_hw.vram_gb.unwrap_or(0.0) - current_vram).max(0.0);
        upgrade_deltas.push(UpgradeDelta {
            resource: "vram_gb".to_string(),
            add_gb: Some(add_good),
            add_cores: None,
            target_fit: Some(FitLevel::Good),
            path: PlanRunPath::Gpu,
            description: format!("+{add_good:.1} GB VRAM -> Good"),
        });
    }
    if let Some(gpu_path) = run_paths.iter().find(|p| p.path == PlanRunPath::Gpu)
        && let Some(rec_hw) = &gpu_path.recommended
    {
        let add_perfect = (rec_hw.vram_gb.unwrap_or(0.0) - current_vram).max(0.0);
        upgrade_deltas.push(UpgradeDelta {
            resource: "vram_gb".to_string(),
            add_gb: Some(add_perfect),
            add_cores: None,
            target_fit: Some(FitLevel::Perfect),
            path: PlanRunPath::Gpu,
            description: format!("+{add_perfect:.1} GB VRAM -> Perfect"),
        });
    }

    let current_ram = system.available_ram_gb;
    if minimum.ram_gb > current_ram {
        let add_ram = minimum.ram_gb - current_ram;
        upgrade_deltas.push(UpgradeDelta {
            resource: "ram_gb".to_string(),
            add_gb: Some(add_ram),
            add_cores: None,
            target_fit: Some(FitLevel::Marginal),
            path: preferred.path,
            description: format!("+{add_ram:.1} GB RAM -> Runnable"),
        });
    }

    if minimum.cpu_cores > system.total_cpu_cores {
        let add_cores = minimum.cpu_cores - system.total_cpu_cores;
        upgrade_deltas.push(UpgradeDelta {
            resource: "cpu_cores".to_string(),
            add_gb: None,
            add_cores: Some(add_cores),
            target_fit: None,
            path: preferred.path,
            description: format!("+{add_cores} CPU cores -> Target TPS"),
        });
    }

    Ok(PlanEstimate {
        estimate_notice: "Estimate-based output using current llmfit fit/speed heuristics; not an exact benchmark."
            .to_string(),
        model_name: model.name.clone(),
        provider: model.provider.clone(),
        context,
        quantization: quant,
        kv_quant,
        target_tps: request.target_tps,
        minimum,
        recommended,
        run_paths,
        current,
        upgrade_deltas,
        kv_alternatives,
    })
}

/// Build the "what if" KV quant rows for the plan output. Includes every
/// option, marking unsupported ones (TurboQuant on non CUDA backends) so the
/// UI can render them with a caveat instead of hiding them.
fn compute_kv_alternatives(
    model: &LlmModel,
    quant: &str,
    context: u32,
    system: &SystemSpecs,
) -> Vec<KvQuantAlternative> {
    let baseline_kv = model.kv_cache_gb(context, KvQuant::Fp16);
    let layout = model.effective_attention_layout();

    KvQuant::all()
        .iter()
        .map(|&kv| {
            let kv_gb = model.kv_cache_gb(context, kv);
            let mem = model.estimate_memory_gb_with_kv(quant, context, kv);
            let savings = if baseline_kv > 0.0 {
                (1.0 - kv_gb / baseline_kv).max(0.0)
            } else {
                0.0
            };

            let supported = match kv {
                KvQuant::TurboQuant => system.backend == GpuBackend::Cuda,
                _ => true,
            };

            let note = match kv {
                KvQuant::TurboQuant => {
                    let mut parts: Vec<String> = Vec::new();
                    parts.push(
                        "Experimental: not in upstream vLLM, see 0xSero/turboquant".to_string(),
                    );
                    if let Some(l) = layout {
                        parts.push(format!(
                            "compresses {} of {} attention layers",
                            l.full,
                            l.total()
                        ));
                    }
                    if !supported {
                        parts.push("requires vLLM + CUDA".to_string());
                    }
                    Some(parts.join("; "))
                }
                KvQuant::Fp8 => Some("vLLM and llama.cpp builds with fp8 KV support".to_string()),
                KvQuant::Q8_0 => {
                    Some("llama.cpp --cache-type-k q8_0 --cache-type-v q8_0".to_string())
                }
                KvQuant::Q4_0 => Some(
                    "llama.cpp --cache-type-k q4_0 --cache-type-v q4_0 (quality drop)".to_string(),
                ),
                KvQuant::Fp16 => None,
            };

            KvQuantAlternative {
                kv_quant: kv,
                memory_required_gb: mem,
                kv_cache_gb: kv_gb,
                savings_fraction: savings,
                note,
                supported,
            }
        })
        .collect()
}

pub fn resolve_model_selector<'a>(
    models: &'a [LlmModel],
    selector: &str,
) -> Result<&'a LlmModel, String> {
    let needle = selector.trim().to_lowercase();
    if needle.is_empty() {
        return Err("Model selector cannot be empty".to_string());
    }

    let exact: Vec<&LlmModel> = models
        .iter()
        .filter(|m| m.name.to_lowercase() == needle)
        .collect();
    if exact.len() == 1 {
        return Ok(exact[0]);
    }

    let partial: Vec<&LlmModel> = models
        .iter()
        .filter(|m| m.name.to_lowercase().contains(&needle))
        .collect();

    match partial.len() {
        0 => Err(format!("No model found matching '{}'.", selector)),
        1 => Ok(partial[0]),
        _ => {
            let suggestions = partial
                .iter()
                .take(10)
                .map(|m| m.name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            Err(format!(
                "Model selector '{}' is ambiguous. Matches: {}",
                selector, suggestions
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_model() -> LlmModel {
        LlmModel {
            name: "Qwen-Test-7B".to_string(),
            provider: "Qwen".to_string(),
            parameter_count: "7B".to_string(),
            parameters_raw: Some(7_000_000_000),
            min_ram_gb: 6.0,
            recommended_ram_gb: 12.0,
            min_vram_gb: Some(6.0),
            quantization: "Q4_K_M".to_string(),
            context_length: 32768,
            use_case: "Coding".to_string(),
            is_moe: false,
            num_experts: None,
            active_experts: None,
            active_parameters: None,
            release_date: None,
            gguf_sources: vec![],
            capabilities: vec![],
            format: crate::models::ModelFormat::default(),
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

    fn test_specs() -> SystemSpecs {
        SystemSpecs {
            total_ram_gb: 32.0,
            available_ram_gb: 24.0,
            total_cpu_cores: 8,
            cpu_name: "Test CPU".to_string(),
            has_gpu: true,
            gpu_vram_gb: Some(12.0),
            total_gpu_vram_gb: Some(12.0),
            gpu_name: Some("Test GPU".to_string()),
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
    fn test_normalize_quant() {
        assert_eq!(normalize_quant("q4_k_m"), Some("Q4_K_M".to_string()));
        assert_eq!(normalize_quant("mlx-4bit"), Some("mlx-4bit".to_string()));
        assert_eq!(normalize_quant("bad"), None);
    }

    #[test]
    fn test_normalize_quant_all_supported() {
        for q in SUPPORTED_QUANTS {
            if q.starts_with("mlx-") || q.starts_with("AWQ-") || q.starts_with("GPTQ-") {
                continue; // handled by case-insensitive paths
            }
            assert_eq!(
                normalize_quant(&q.to_lowercase()),
                Some(q.to_string()),
                "lowercase '{}' should normalize",
                q
            );
        }
    }

    #[test]
    fn test_normalize_quant_whitespace_handling() {
        assert_eq!(normalize_quant("  q4_k_m  "), Some("Q4_K_M".to_string()));
        assert_eq!(normalize_quant(""), None);
        assert_eq!(normalize_quant("   "), None);
    }

    #[test]
    fn test_estimate_model_plan() {
        let req = PlanRequest {
            context: 8192,
            quant: Some("Q4_K_M".to_string()),
            target_tps: Some(8.0),
            kv_quant: None,
        };
        let plan =
            estimate_model_plan(&test_model(), &req, &test_specs()).expect("plan should build");
        assert_eq!(plan.quantization, "Q4_K_M");
        assert!(!plan.run_paths.is_empty());
        assert!(plan.minimum.ram_gb > 0.0);
    }

    #[test]
    fn test_estimate_model_plan_zero_context_errors() {
        let req = PlanRequest {
            context: 0,
            quant: None,
            target_tps: None,
            kv_quant: None,
        };
        let result = estimate_model_plan(&test_model(), &req, &test_specs());
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .contains("--context must be greater than 0")
        );
    }

    #[test]
    fn test_estimate_model_plan_negative_tps_errors() {
        let req = PlanRequest {
            context: 4096,
            quant: None,
            target_tps: Some(-5.0),
            kv_quant: None,
        };
        let result = estimate_model_plan(&test_model(), &req, &test_specs());
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .contains("--target-tps must be greater than 0")
        );
    }

    #[test]
    fn test_estimate_model_plan_invalid_quant_errors() {
        let req = PlanRequest {
            context: 4096,
            quant: Some("INVALID_QUANT".to_string()),
            target_tps: None,
            kv_quant: None,
        };
        let result = estimate_model_plan(&test_model(), &req, &test_specs());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unsupported quantization"));
    }

    #[test]
    fn test_estimate_model_plan_uses_model_quant_when_none() {
        let req = PlanRequest {
            context: 4096,
            quant: None,
            target_tps: None,
            kv_quant: None,
        };
        let plan = estimate_model_plan(&test_model(), &req, &test_specs()).unwrap();
        assert_eq!(plan.quantization, "Q4_K_M"); // model default
    }

    #[test]
    fn test_estimate_model_plan_has_three_run_paths() {
        let req = PlanRequest {
            context: 4096,
            quant: None,
            target_tps: None,
            kv_quant: None,
        };
        let plan = estimate_model_plan(&test_model(), &req, &test_specs()).unwrap();
        assert_eq!(plan.run_paths.len(), 3);
        assert_eq!(plan.run_paths[0].path, PlanRunPath::Gpu);
        assert_eq!(plan.run_paths[1].path, PlanRunPath::CpuOffload);
        assert_eq!(plan.run_paths[2].path, PlanRunPath::CpuOnly);
    }

    #[test]
    fn test_estimate_model_plan_gpu_path_feasible() {
        let req = PlanRequest {
            context: 4096,
            quant: Some("Q4_K_M".to_string()),
            target_tps: None,
            kv_quant: None,
        };
        let plan = estimate_model_plan(&test_model(), &req, &test_specs()).unwrap();
        let gpu_path = &plan.run_paths[0];
        assert!(gpu_path.feasible);
        assert!(gpu_path.minimum.is_some());
        assert!(gpu_path.recommended.is_some());
        assert!(gpu_path.estimated_tps.unwrap() > 0.0);
    }

    // ── fit_level_for ────────────────────────────────────────────────

    #[test]
    fn test_fit_level_for_gpu_perfect() {
        let fit = fit_level_for(PlanRunPath::Gpu, 8.0, 24.0, 12.0);
        assert_eq!(fit, FitLevel::Perfect);
    }

    #[test]
    fn test_fit_level_for_gpu_good() {
        // required*1.2 = 9.6, available = 10.0 > 9.6, but recommended = 12.0 > 10.0
        let fit = fit_level_for(PlanRunPath::Gpu, 8.0, 10.0, 12.0);
        assert_eq!(fit, FitLevel::Good);
    }

    #[test]
    fn test_fit_level_for_gpu_marginal() {
        // available barely exceeds required, but less than required*1.2
        let fit = fit_level_for(PlanRunPath::Gpu, 8.0, 8.5, 12.0);
        assert_eq!(fit, FitLevel::Marginal);
    }

    #[test]
    fn test_fit_level_for_too_tight() {
        let fit = fit_level_for(PlanRunPath::Gpu, 24.0, 8.0, 32.0);
        assert_eq!(fit, FitLevel::TooTight);
    }

    #[test]
    fn test_fit_level_for_cpu_offload_caps_at_good() {
        let fit = fit_level_for(PlanRunPath::CpuOffload, 8.0, 24.0, 12.0);
        assert_eq!(fit, FitLevel::Good);
    }

    #[test]
    fn test_fit_level_for_cpu_only_always_marginal() {
        let fit = fit_level_for(PlanRunPath::CpuOnly, 4.0, 64.0, 8.0);
        assert_eq!(fit, FitLevel::Marginal);
    }

    // ── PlanRunPath ──────────────────────────────────────────────────

    #[test]
    fn test_plan_run_path_labels() {
        assert_eq!(PlanRunPath::Gpu.label(), "GPU");
        assert_eq!(PlanRunPath::CpuOffload.label(), "CPU offload");
        assert_eq!(PlanRunPath::CpuOnly.label(), "CPU-only");
    }

    #[test]
    fn test_plan_run_path_to_run_mode() {
        assert_eq!(PlanRunPath::Gpu.run_mode(), RunMode::Gpu);
        assert_eq!(PlanRunPath::CpuOffload.run_mode(), RunMode::CpuOffload);
        assert_eq!(PlanRunPath::CpuOnly.run_mode(), RunMode::CpuOnly);
    }

    // ── estimate_tps ─────────────────────────────────────────────────

    #[test]
    fn test_estimate_tps_gpu_faster_than_cpu() {
        let model = test_model();
        let gpu_tps = estimate_tps(&model, "Q4_K_M", GpuBackend::Cuda, PlanRunPath::Gpu, 8);
        let cpu_tps = estimate_tps(
            &model,
            "Q4_K_M",
            GpuBackend::CpuX86,
            PlanRunPath::CpuOnly,
            8,
        );
        assert!(gpu_tps > cpu_tps);
    }

    #[test]
    fn test_estimate_tps_cpu_offload_slower_than_gpu() {
        let model = test_model();
        let gpu_tps = estimate_tps(&model, "Q4_K_M", GpuBackend::Cuda, PlanRunPath::Gpu, 8);
        let offload_tps = estimate_tps(
            &model,
            "Q4_K_M",
            GpuBackend::Cuda,
            PlanRunPath::CpuOffload,
            8,
        );
        assert!(gpu_tps > offload_tps);
    }

    #[test]
    fn test_estimate_tps_more_cores_helps() {
        let model = test_model();
        let tps_4 = estimate_tps(&model, "Q4_K_M", GpuBackend::Cuda, PlanRunPath::Gpu, 4);
        let tps_16 = estimate_tps(&model, "Q4_K_M", GpuBackend::Cuda, PlanRunPath::Gpu, 16);
        assert!(tps_16 >= tps_4);
    }

    #[test]
    fn test_estimate_tps_with_known_gpu_uses_bandwidth() {
        let model = test_model();
        let bw_tps = estimate_tps_with_gpu(
            &model,
            "Q4_K_M",
            GpuBackend::Cuda,
            PlanRunPath::Gpu,
            8,
            Some("NVIDIA RTX 4090"),
        );
        let fallback_tps = estimate_tps_with_gpu(
            &model,
            "Q4_K_M",
            GpuBackend::Cuda,
            PlanRunPath::Gpu,
            8,
            None,
        );
        // Known GPU should give a different (bandwidth-based) estimate
        assert!((bw_tps - fallback_tps).abs() > 0.01);
    }

    // ── minimum_cores_for_target ─────────────────────────────────────

    #[test]
    fn test_minimum_cores_no_target_returns_default() {
        let model = test_model();
        let cores =
            minimum_cores_for_target(&model, "Q4_K_M", GpuBackend::Cuda, PlanRunPath::Gpu, None);
        assert_eq!(cores, Some(4));
    }

    #[test]
    fn test_minimum_cores_with_reachable_target() {
        let model = test_model();
        let cores = minimum_cores_for_target(
            &model,
            "Q4_K_M",
            GpuBackend::Cuda,
            PlanRunPath::Gpu,
            Some(5.0),
        );
        assert!(cores.is_some());
        assert!(cores.unwrap() >= 1);
    }

    #[test]
    fn test_minimum_cores_unreachable_target_returns_none() {
        let model = test_model();
        let cores = minimum_cores_for_target(
            &model,
            "Q4_K_M",
            GpuBackend::CpuX86,
            PlanRunPath::CpuOnly,
            Some(999999.0),
        );
        assert!(cores.is_none());
    }

    // ── default_gpu_backend ──────────────────────────────────────────

    #[test]
    fn test_default_gpu_backend_uses_system_when_gpu() {
        let specs = test_specs();
        assert_eq!(default_gpu_backend(&specs), GpuBackend::Cuda);
    }

    #[test]
    fn test_default_gpu_backend_falls_back_to_cuda() {
        let mut specs = test_specs();
        specs.has_gpu = false;
        assert_eq!(default_gpu_backend(&specs), GpuBackend::Cuda);
    }

    // ── evaluate_current ─────────────────────────────────────────────

    #[test]
    fn test_evaluate_current_with_gpu() {
        let model = test_model();
        let specs = test_specs();
        let status = evaluate_current(&model, "Q4_K_M", 4096, KvQuant::Fp16, None, &specs);
        assert!(status.estimated_tps > 0.0);
        // With 12GB VRAM and 7B model, GPU should be preferred
        assert_eq!(status.run_mode, RunMode::Gpu);
    }

    #[test]
    fn test_evaluate_current_no_gpu_uses_cpu() {
        let model = test_model();
        let mut specs = test_specs();
        specs.has_gpu = false;
        specs.gpu_vram_gb = None;
        specs.total_gpu_vram_gb = None;
        let status = evaluate_current(&model, "Q4_K_M", 4096, KvQuant::Fp16, None, &specs);
        assert_eq!(status.run_mode, RunMode::CpuOnly);
        assert!(status.estimated_tps > 0.0);
    }

    #[test]
    fn test_evaluate_current_too_tight_when_no_memory() {
        let model = test_model();
        let mut specs = test_specs();
        specs.has_gpu = false;
        specs.gpu_vram_gb = None;
        specs.total_gpu_vram_gb = None;
        specs.available_ram_gb = 0.5; // too small for the model
        let status = evaluate_current(
            &model,
            "Q4_K_M",
            4096,
            KvQuant::Fp16,
            Some(999999.0),
            &specs,
        );
        assert_eq!(status.fit_level, FitLevel::TooTight);
    }

    // ── build_path_estimate ──────────────────────────────────────────

    #[test]
    fn test_build_path_estimate_gpu() {
        let model = test_model();
        let specs = test_specs();
        let estimate = build_path_estimate(
            &model,
            "Q4_K_M",
            4096,
            KvQuant::Fp16,
            None,
            PlanRunPath::Gpu,
            &specs,
        );
        assert!(estimate.feasible);
        let min = estimate.minimum.unwrap();
        assert!(min.vram_gb.unwrap() > 0.0);
        assert!(min.ram_gb > 0.0);
    }

    #[test]
    fn test_build_path_estimate_cpu_offload_on_unified_is_infeasible() {
        let model = test_model();
        let mut specs = test_specs();
        specs.unified_memory = true;
        let estimate = build_path_estimate(
            &model,
            "Q4_K_M",
            4096,
            KvQuant::Fp16,
            None,
            PlanRunPath::CpuOffload,
            &specs,
        );
        assert!(!estimate.feasible);
        assert!(estimate.notes.iter().any(|n| n.contains("unified-memory")));
    }

    #[test]
    fn test_build_path_estimate_cpu_only_no_vram() {
        let model = test_model();
        let specs = test_specs();
        let estimate = build_path_estimate(
            &model,
            "Q4_K_M",
            4096,
            KvQuant::Fp16,
            None,
            PlanRunPath::CpuOnly,
            &specs,
        );
        assert!(estimate.feasible);
        assert!(estimate.minimum.as_ref().unwrap().vram_gb.is_none());
    }

    // ── resolve_model_selector ───────────────────────────────────────

    #[test]
    fn test_resolve_model_selector() {
        let models = vec![test_model()];
        let found = resolve_model_selector(&models, "qwen-test-7b").expect("exact match");
        assert_eq!(found.name, "Qwen-Test-7B");
    }

    #[test]
    fn test_resolve_model_selector_empty_errors() {
        let models = vec![test_model()];
        let result = resolve_model_selector(&models, "");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("cannot be empty"));
    }

    #[test]
    fn test_resolve_model_selector_not_found() {
        let models = vec![test_model()];
        let result = resolve_model_selector(&models, "nonexistent-model");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No model found"));
    }

    #[test]
    fn test_resolve_model_selector_ambiguous() {
        let mut m1 = test_model();
        m1.name = "Qwen-Test-7B".to_string();
        let mut m2 = test_model();
        m2.name = "Qwen-Test-14B".to_string();
        let models = vec![m1, m2];
        let result = resolve_model_selector(&models, "qwen-test");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("ambiguous"));
    }

    #[test]
    fn test_resolve_model_selector_partial_match() {
        let models = vec![test_model()];
        let found = resolve_model_selector(&models, "test-7b").expect("partial match");
        assert_eq!(found.name, "Qwen-Test-7B");
    }

    // ── upgrade_deltas ───────────────────────────────────────────────

    #[test]
    fn test_plan_has_upgrade_deltas() {
        let model = test_model();
        let mut specs = test_specs();
        specs.gpu_vram_gb = Some(4.0); // small VRAM triggers upgrade suggestion
        specs.total_gpu_vram_gb = Some(4.0);
        let req = PlanRequest {
            context: 4096,
            quant: Some("Q4_K_M".to_string()),
            target_tps: None,
            kv_quant: None,
        };
        let plan = estimate_model_plan(&model, &req, &specs).unwrap();
        assert!(!plan.upgrade_deltas.is_empty());
    }

    #[test]
    fn test_normalize_awq_gptq_quants() {
        assert_eq!(normalize_quant("awq-4bit"), Some("AWQ-4bit".to_string()));
        assert_eq!(normalize_quant("AWQ-4BIT"), Some("AWQ-4bit".to_string()));
        assert_eq!(normalize_quant("awq-8bit"), Some("AWQ-8bit".to_string()));
        assert_eq!(normalize_quant("gptq-int4"), Some("GPTQ-Int4".to_string()));
        assert_eq!(normalize_quant("GPTQ-INT8"), Some("GPTQ-Int8".to_string()));
    }

    // ── KV quant flag ────────────────────────────────────────────────

    #[test]
    fn test_plan_includes_kv_alternatives_for_all_options() {
        let req = PlanRequest {
            context: 8192,
            quant: Some("Q4_K_M".to_string()),
            target_tps: None,
            kv_quant: None,
        };
        let plan = estimate_model_plan(&test_model(), &req, &test_specs()).unwrap();
        // One row per KvQuant variant
        assert_eq!(plan.kv_alternatives.len(), KvQuant::all().len());
        // Default kv_quant resolves to fp16
        assert_eq!(plan.kv_quant, KvQuant::Fp16);
    }

    #[test]
    fn test_plan_with_q4_kv_reduces_total_memory() {
        let base = PlanRequest {
            context: 32_768,
            quant: Some("Q4_K_M".to_string()),
            target_tps: None,
            kv_quant: None,
        };
        let mut q4 = base.clone();
        q4.kv_quant = Some(KvQuant::Q4_0);
        let plan_fp16 = estimate_model_plan(&test_model(), &base, &test_specs()).unwrap();
        let plan_q4 = estimate_model_plan(&test_model(), &q4, &test_specs()).unwrap();
        assert!(plan_q4.minimum.ram_gb <= plan_fp16.minimum.ram_gb);
    }

    #[test]
    fn test_plan_with_turboquant_on_non_cuda_errors() {
        let mut specs = test_specs();
        specs.backend = GpuBackend::Metal;
        let req = PlanRequest {
            context: 8192,
            quant: Some("Q4_K_M".to_string()),
            target_tps: None,
            kv_quant: Some(KvQuant::TurboQuant),
        };
        let result = estimate_model_plan(&test_model(), &req, &specs);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("TurboQuant"), "got: {}", err);
        assert!(err.contains("vLLM"), "got: {}", err);
    }

    #[test]
    fn test_plan_with_turboquant_on_cuda_succeeds() {
        let req = PlanRequest {
            context: 8192,
            quant: Some("Q4_K_M".to_string()),
            target_tps: None,
            kv_quant: Some(KvQuant::TurboQuant),
        };
        let plan = estimate_model_plan(&test_model(), &req, &test_specs())
            .expect("CUDA backend should allow TQ");
        assert_eq!(plan.kv_quant, KvQuant::TurboQuant);
    }

    #[test]
    fn test_kv_alternatives_mark_turboquant_unsupported_off_cuda() {
        let mut specs = test_specs();
        specs.backend = GpuBackend::Metal;
        let req = PlanRequest {
            context: 8192,
            quant: Some("Q4_K_M".to_string()),
            target_tps: None,
            kv_quant: None, // fp16 default, not TQ — so the request itself is fine
        };
        let plan = estimate_model_plan(&test_model(), &req, &specs).unwrap();
        let tq = plan
            .kv_alternatives
            .iter()
            .find(|a| a.kv_quant == KvQuant::TurboQuant)
            .expect("TQ row must exist");
        assert!(!tq.supported);
        assert!(
            tq.note.as_deref().unwrap_or("").contains("vLLM"),
            "expected vLLM hint in note"
        );
    }
}
