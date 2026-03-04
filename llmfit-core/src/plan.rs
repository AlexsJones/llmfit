use crate::fit::{FitLevel, RunMode};
use crate::hardware::{GpuBackend, SystemSpecs};
use crate::models::{LlmModel, quant_speed_multiplier};

const SUPPORTED_QUANTS: &[&str] = &[
    "F32", "F16", "BF16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0", "Q3_K_M", "Q2_K", "mlx-8bit",
    "mlx-4bit",
];

#[derive(Debug, Clone, serde::Serialize)]
pub struct PlanRequest {
    pub context: u32,
    pub quant: Option<String>,
    pub target_tps: Option<f64>,
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
pub struct PlanEstimate {
    pub estimate_notice: String,
    pub model_name: String,
    pub provider: String,
    pub context: u32,
    pub quantization: String,
    pub target_tps: Option<f64>,
    pub minimum: HardwareEstimate,
    pub recommended: HardwareEstimate,
    pub run_paths: Vec<PathEstimate>,
    pub current: PlanCurrentStatus,
    pub upgrade_deltas: Vec<UpgradeDelta>,
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
    if path != PlanRunPath::CpuOnly {
        if let Some(name) = gpu_name {
            if let Some(bw) = gpu_memory_bandwidth_gbps(name) {
                let model_gb = params * quant_bytes_per_param(quant);
                let efficiency = 0.55;
                let raw_tps = (bw / model_gb) * efficiency;

                let mode_factor = match path {
                    PlanRunPath::Gpu => 1.0,
                    PlanRunPath::CpuOffload => 0.5,
                    PlanRunPath::CpuOnly => unreachable!(),
                };

                return (raw_tps * mode_factor).max(0.1);
            }
        }
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
    target_tps: Option<f64>,
    system: &SystemSpecs,
) -> PlanCurrentStatus {
    let model_mem = model.estimate_memory_gb(quant, context);
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
    target_tps: Option<f64>,
    path: PlanRunPath,
    system: &SystemSpecs,
) -> PathEstimate {
    let model_mem = model.estimate_memory_gb(quant, context);
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

    let context = request.context;
    let run_paths = vec![
        build_path_estimate(
            model,
            &quant,
            context,
            request.target_tps,
            PlanRunPath::Gpu,
            system,
        ),
        build_path_estimate(
            model,
            &quant,
            context,
            request.target_tps,
            PlanRunPath::CpuOffload,
            system,
        ),
        build_path_estimate(
            model,
            &quant,
            context,
            request.target_tps,
            PlanRunPath::CpuOnly,
            system,
        ),
    ];

    let current = evaluate_current(model, &quant, context, request.target_tps, system);

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
        target_tps: request.target_tps,
        minimum,
        recommended,
        run_paths,
        current,
        upgrade_deltas,
    })
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
        }
    }

    #[test]
    fn test_normalize_quant() {
        assert_eq!(normalize_quant("q4_k_m"), Some("Q4_K_M".to_string()));
        assert_eq!(normalize_quant("mlx-4bit"), Some("mlx-4bit".to_string()));
        assert_eq!(normalize_quant("bad"), None);
    }

    #[test]
    fn test_estimate_model_plan() {
        let req = PlanRequest {
            context: 8192,
            quant: Some("Q4_K_M".to_string()),
            target_tps: Some(8.0),
        };
        let plan =
            estimate_model_plan(&test_model(), &req, &test_specs()).expect("plan should build");
        assert_eq!(plan.quantization, "Q4_K_M");
        assert!(!plan.run_paths.is_empty());
        assert!(plan.minimum.ram_gb > 0.0);
    }

    #[test]
    fn test_resolve_model_selector() {
        let models = vec![test_model()];
        let found = resolve_model_selector(&models, "qwen-test-7b").expect("exact match");
        assert_eq!(found.name, "Qwen-Test-7B");
    }
}
