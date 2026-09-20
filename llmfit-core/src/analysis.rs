use crate::fit::{CalcConfig, InferenceRuntime, ModelFit};
use crate::hardware::SystemSpecs;
use crate::models::{LlmModel, ModelDatabase};
use crate::providers::{
    self, DockerModelRunnerProvider, LlamaCppProvider, LmStudioProvider, MlxProvider,
    ModelProvider, OllamaProvider, RamaLamaProvider, VllmProvider,
};
use std::collections::HashSet;

/// Aggregated installed-model sets from all supported inference providers.
///
/// A single point of truth used by both the CLI and the TUI to check which
/// models are locally installed. Replaces the scattered `HashSet<String>` fields
/// that used to live on each caller's struct.
#[derive(Debug, Clone)]
pub struct InstalledIndex {
    pub ollama: HashSet<String>,
    pub ollama_count: usize,
    pub mlx: HashSet<String>,
    pub llamacpp: HashSet<String>,
    pub llamacpp_count: usize,
    pub docker_mr: HashSet<String>,
    pub docker_mr_count: usize,
    pub lmstudio: HashSet<String>,
    pub lmstudio_count: usize,
    /// Models found in LM Studio's models directory. Kept apart from
    /// `lmstudio` because the API ids there are matched by substring, while
    /// these directory-derived names are matched by equality.
    pub lmstudio_disk: HashSet<String>,
    pub lmstudio_disk_count: usize,
    pub vllm: HashSet<String>,
    pub vllm_count: usize,
    pub ramalama: HashSet<String>,
    pub ramalama_count: usize,
}

impl InstalledIndex {
    /// Build an empty index — used as a placeholder while providers load.
    pub fn empty() -> Self {
        Self {
            ollama: HashSet::new(),
            ollama_count: 0,
            mlx: HashSet::new(),
            llamacpp: HashSet::new(),
            llamacpp_count: 0,
            docker_mr: HashSet::new(),
            docker_mr_count: 0,
            lmstudio: HashSet::new(),
            lmstudio_count: 0,
            lmstudio_disk: HashSet::new(),
            lmstudio_disk_count: 0,
            vllm: HashSet::new(),
            vllm_count: 0,
            ramalama: HashSet::new(),
            ramalama_count: 0,
        }
    }

    /// Detect installed models across all providers in parallel.
    ///
    /// Each provider query is issued on its own thread so that a single
    /// offline/slow backend (worst case ~1.5 s timeout) doesn't serialize
    /// into ~9 s of total blocking time for the CLI path.
    pub fn detect_all() -> Self {
        std::thread::scope(|s| {
            let ollama = s.spawn(|| {
                let p = OllamaProvider::new();
                p.installed_models_counted()
            });
            let mlx = s.spawn(|| MlxProvider::new().installed_models());
            let llamacpp = s.spawn(|| {
                let p = LlamaCppProvider::new();
                p.installed_models_counted()
            });
            let docker_mr = s.spawn(|| {
                let p = DockerModelRunnerProvider::new();
                p.installed_models_counted()
            });
            let lmstudio = s.spawn(|| {
                let p = LmStudioProvider::new();
                p.installed_models_counted()
            });
            let lmstudio_disk = s.spawn(providers::scan_lmstudio_models_dir);
            let vllm = s.spawn(|| {
                let p = VllmProvider::new();
                p.installed_models_counted()
            });
            let ramalama = s.spawn(|| {
                let p = RamaLamaProvider::new();
                p.installed_models_counted()
            });

            let (ollama, ollama_count) = ollama.join().unwrap();
            let mlx = mlx.join().unwrap();
            let (llamacpp, llamacpp_count) = llamacpp.join().unwrap();
            let (docker_mr, docker_mr_count) = docker_mr.join().unwrap();
            let (lmstudio, lmstudio_count) = lmstudio.join().unwrap();
            // Enrichment rather than a load-bearing provider: if the scan
            // thread dies, report no disk models instead of taking the whole
            // installed-model analysis down with it.
            let (lmstudio_disk, lmstudio_disk_count) = lmstudio_disk.join().unwrap_or_default();
            let (vllm, vllm_count) = vllm.join().unwrap();
            let (ramalama, ramalama_count) = ramalama.join().unwrap();

            Self {
                ollama,
                ollama_count,
                mlx,
                llamacpp,
                llamacpp_count,
                docker_mr,
                docker_mr_count,
                lmstudio,
                lmstudio_count,
                lmstudio_disk,
                lmstudio_disk_count,
                vllm,
                vllm_count,
                ramalama,
                ramalama_count,
            }
        })
    }

    /// Returns `true` when the model is installed in **any** provider.
    ///
    /// Takes the whole model rather than its name so the Ollama matcher can
    /// use the catalog's parameter count to reject a family-level tag whose
    /// size disagrees (`deepseek-r1:14b` is not the 684B `DeepSeek-R1`).
    pub fn is_installed(&self, model: &LlmModel) -> bool {
        let model_name = model.name.as_str();
        providers::is_model_installed_sized(model_name, model.known_params_b(), &self.ollama)
            || providers::is_model_installed_mlx(model_name, &self.mlx)
            || providers::is_model_installed_llamacpp(model_name, &self.llamacpp)
            || providers::is_model_installed_docker_mr(model_name, &self.docker_mr)
            || providers::is_model_installed_lmstudio(model_name, &self.lmstudio)
            || providers::is_model_installed_lmstudio_disk(model_name, &self.lmstudio_disk)
            || providers::is_model_installed_vllm(model_name, &self.vllm)
            || providers::is_model_installed_ramalama(model_name, &self.ramalama)
    }

    /// Returns the display names of all providers that have this model
    /// installed. Used by the detail panel in the TUI.
    pub fn installed_providers(&self, model: &LlmModel) -> Vec<&'static str> {
        let model_name = model.name.as_str();
        let mut out = Vec::new();
        if providers::is_model_installed_sized(model_name, model.known_params_b(), &self.ollama) {
            out.push("Ollama");
        }
        if providers::is_model_installed_mlx(model_name, &self.mlx) {
            out.push("MLX");
        }
        if providers::is_model_installed_llamacpp(model_name, &self.llamacpp) {
            out.push("llama.cpp");
        }
        if providers::is_model_installed_docker_mr(model_name, &self.docker_mr) {
            out.push("Docker");
        }
        if providers::is_model_installed_lmstudio(model_name, &self.lmstudio)
            || providers::is_model_installed_lmstudio_disk(model_name, &self.lmstudio_disk)
        {
            out.push("LM Studio");
        }
        if providers::is_model_installed_vllm(model_name, &self.vllm) {
            out.push("vLLM");
        }
        if providers::is_model_installed_ramalama(model_name, &self.ramalama) {
            out.push("RamaLama");
        }
        out
    }
}

/// The catalog entries eligible for a ranked fit sweep on this hardware.
///
/// Two gates, in one place so no surface can drift from another:
///
/// * [`backend_compatible`](crate::fit::backend_compatible) — the model can
///   actually run on this backend.
/// * `!`[`LlmModel::is_sanitization_demoted`] — the entry isn't a
///   speculative-decoding draft head, a size/name divergence, or an
///   implausible footprint (issue #969, problem 3).
///
/// Demoted entries stay in [`ModelDatabase`] and remain reachable through
/// `get_all_models`/`find_model`, so `info` and `search` can still explain
/// them; they are only kept out of anything that *ranks* models.
///
/// Every ranked sweep — CLI, TUI, REST, MCP — filters through here rather
/// than repeating the predicate, so an API or MCP consumer can never be
/// offered a draft head that the CLI hides.
pub fn rankable_models<'a>(
    models: &'a [LlmModel],
    specs: &'a SystemSpecs,
) -> impl Iterator<Item = &'a LlmModel> {
    models
        .iter()
        .filter(move |m| crate::fit::backend_compatible(m, specs) && !m.is_sanitization_demoted())
}

/// How many catalog entries [`rankable_models`] drops on this hardware
/// because the backend can't run them. Reported by UIs so a shorter list
/// than expected is explained rather than mysterious.
pub fn backend_hidden_count(models: &[LlmModel], specs: &SystemSpecs) -> usize {
    models
        .iter()
        .filter(|m| !crate::fit::backend_compatible(m, specs))
        .count()
}

/// Analyze one model, honouring a hardware profile's [`CalcConfig`] when the
/// caller has one.
///
/// The two `ModelFit` entry points differ in more than a default —
/// `analyze_with_config` carries the context cap *inside* the config — so
/// surfaces that may or may not hold a profile would each have to repeat this
/// branch, and one of them would eventually drop the config (issue #969).
pub fn analyze_with_optional_config(
    model: &LlmModel,
    specs: &SystemSpecs,
    context_limit: Option<u32>,
    config: Option<&CalcConfig>,
) -> ModelFit {
    match config {
        Some(config) => {
            let mut config = config.clone();
            config.context_cap = context_limit.or(config.context_cap);
            ModelFit::analyze_with_config(model, specs, config)
        }
        None => ModelFit::analyze_with_context_limit(model, specs, context_limit),
    }
}

/// How each model in the sweep is analyzed.
enum FitMode {
    /// Automatic runtime selection, optionally forced to one runtime.
    Runtime(Option<InferenceRuntime>),
    /// Custom calculation parameters (e.g. from a hardware profile).
    Config(Box<CalcConfig>),
}

/// Build a complete `Vec<ModelFit>` with installed markers populated.
///
/// Filters models that are backend-incompatible, runs fit analysis, marks
/// each fit's `installed` flag from the given index, and returns the results
/// **unsorted** so the caller can apply its own sort criteria.
pub fn build_model_fits(
    db: &ModelDatabase,
    specs: &SystemSpecs,
    installed: &InstalledIndex,
    context_limit: Option<u32>,
    forced_runtime: Option<InferenceRuntime>,
) -> Vec<ModelFit> {
    build_fits(
        db,
        specs,
        installed,
        context_limit,
        FitMode::Runtime(forced_runtime),
    )
}

/// [`build_model_fits`] with custom calculation parameters, so a hardware
/// profile's bandwidth and efficiency reach every row of the sweep.
///
/// Runtime selection stays automatic: `ModelFit` exposes no analysis entry
/// point that takes both a forced runtime and a `CalcConfig`, so accepting one
/// here could only drop it silently.
pub fn build_model_fits_with_config(
    db: &ModelDatabase,
    specs: &SystemSpecs,
    installed: &InstalledIndex,
    context_limit: Option<u32>,
    config: CalcConfig,
) -> Vec<ModelFit> {
    build_fits(
        db,
        specs,
        installed,
        context_limit,
        FitMode::Config(Box::new(config)),
    )
}

fn build_fits(
    db: &ModelDatabase,
    specs: &SystemSpecs,
    installed: &InstalledIndex,
    context_limit: Option<u32>,
    mode: FitMode,
) -> Vec<ModelFit> {
    // Measured-throughput sources, most trustworthy first: the user's own
    // runs on this machine, llmfit community submissions recorded on
    // identical hardware, then localmaxxing medians on matching presets.
    let local_index = crate::share::LocalBenchIndex::load(specs);
    let community_index = crate::benchmarks::CommunityBenchIndex::for_specs(specs);
    let measured_index = crate::benchmarks::MeasuredTpsIndex::for_specs(specs);

    let mut fits: Vec<ModelFit> = rankable_models(db.get_all_models(), specs)
        .map(|m| {
            let mut fit = match &mode {
                FitMode::Runtime(forced_runtime) => {
                    ModelFit::analyze_with_forced_runtime(m, specs, context_limit, *forced_runtime)
                }
                FitMode::Config(config) => {
                    analyze_with_optional_config(m, specs, context_limit, Some(config.as_ref()))
                }
            };
            fit.installed = installed.is_installed(m);
            fit.measured_tps = local_index
                .as_ref()
                .and_then(|idx| idx.lookup(&m.name))
                .or_else(|| community_index.as_ref().and_then(|idx| idx.lookup(&m.name)))
                .or_else(|| {
                    measured_index
                        .as_ref()
                        .and_then(|idx| idx.lookup(&m.name, &fit.best_quant))
                });
            fit.refresh_estimate_confidence();
            fit
        })
        .collect();
    apply_local_calibration(&mut fits);
    fits
}

/// Calibrate formula estimates from benchmark runs made on this exact
/// hardware: the user's own local runs, plus llmfit community submissions
/// recorded on an identical configuration (so a fresh install benefits the
/// moment anyone contributed on the same machine class).
///
/// Anchors must map to a catalog entry with a trustworthy size (>= 1B
/// params, dense — MoE and tiny models don't scale like bandwidth-bound
/// dense generation). The median measured/estimated ratio, clamped to
/// [0.05, 3.0], scales every row's estimate and is recorded in
/// `estimate_basis.local_calibration`.
///
/// Idempotent: ratios and scaling always derive from the uncalibrated
/// estimate, so re-applying after a new bench never compounds.
pub fn apply_local_calibration(fits: &mut [ModelFit]) {
    use crate::benchmarks::MeasuredSource;

    fn uncalibrated(f: &ModelFit) -> f64 {
        match f.estimate_basis.local_calibration {
            Some(c) if c > 0.0 => f.estimated_tps / c,
            _ => f.estimated_tps,
        }
    }

    let mut ratios: Vec<f64> = fits
        .iter()
        .filter(|f| f.model.params_b() >= 1.0 && !f.model.is_moe)
        .filter_map(|f| {
            let m = f.measured_tps.as_ref()?;
            if !matches!(
                m.source,
                MeasuredSource::LocalBench | MeasuredSource::CommunityLlmfit
            ) {
                return None;
            }
            let est = uncalibrated(f);
            (est > 0.0 && m.tok_s > 0.0).then(|| m.tok_s / est)
        })
        .collect();
    if ratios.is_empty() {
        return;
    }
    ratios.sort_by(|a, b| a.partial_cmp(b).expect("ratios are finite"));
    let factor = median(&ratios).clamp(0.05, 3.0);

    for f in fits.iter_mut() {
        if f.estimated_tps <= 0.0 {
            continue;
        }
        f.estimated_tps = uncalibrated(f) * factor;
        f.estimate_basis.local_calibration = Some(factor);
        // A measured row keeps its measured confidence; this only promotes the
        // formula rows that the factor was applied to.
        f.refresh_estimate_confidence();
    }
}

fn median(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

#[cfg(test)]
mod calibration_tests {
    use super::*;

    #[test]
    fn median_of_sorted() {
        assert_eq!(median(&[0.1]), 0.1);
        assert_eq!(median(&[0.1, 0.3]), 0.2);
        assert_eq!(median(&[0.1, 0.2, 0.9]), 0.2);
    }
}

#[cfg(test)]
mod sanitization_gate_tests {
    use super::*;
    use crate::hardware::GpuBackend;

    fn specs_with_gpu() -> SystemSpecs {
        SystemSpecs {
            total_ram_gb: 64.0,
            available_ram_gb: 48.0,
            total_cpu_cores: 16,
            cpu_name: "Test CPU".to_string(),
            has_gpu: true,
            gpu_vram_gb: Some(24.0),
            total_gpu_vram_gb: Some(24.0),
            gpu_available_gb: None,
            gpu_name: Some("Test GPU".to_string()),
            gpu_count: 1,
            unified_memory: false,
            backend: GpuBackend::Cuda,
            gpus: vec![],
            cluster_mode: false,
            cluster_node_count: 0,
        }
    }

    fn demoted_names(db: &ModelDatabase) -> std::collections::HashSet<String> {
        let names: std::collections::HashSet<String> = db
            .get_all_models()
            .iter()
            .filter(|m| m.is_sanitization_demoted())
            .map(|m| m.name.clone())
            .collect();
        assert!(
            !names.is_empty(),
            "expected the embedded catalog to contain at least one sanitization-demoted entry"
        );
        names
    }

    /// `build_model_fits` must not surface a demoted catalog entry in its
    /// ranked output (issue #969, problem 3), even though the same entry
    /// stays queryable through `ModelDatabase::get_all_models`.
    #[test]
    fn build_model_fits_excludes_sanitization_demoted_entries() {
        let db = ModelDatabase::new();
        let specs = specs_with_gpu();
        let demoted = demoted_names(&db);

        let fits = build_model_fits(&db, &specs, &InstalledIndex::empty(), None, None);

        assert!(
            fits.iter().all(|f| !demoted.contains(&f.model.name)),
            "a sanitization-demoted model leaked into ranked fits"
        );
    }

    /// The gate every ranked surface shares. `build_model_fits` is only one
    /// caller — the TUI, REST and MCP sweeps filter through this directly, so
    /// it needs its own coverage rather than inheriting it (issue #969).
    #[test]
    fn rankable_models_drops_demoted_and_backend_incompatible_entries() {
        let db = ModelDatabase::new();
        let specs = specs_with_gpu();
        let demoted = demoted_names(&db);

        let rankable: Vec<&LlmModel> = rankable_models(db.get_all_models(), &specs).collect();

        assert!(!rankable.is_empty(), "expected some rankable models");
        assert!(
            rankable.iter().all(|m| !demoted.contains(&m.name)),
            "a sanitization-demoted model survived rankable_models"
        );
        assert!(
            rankable
                .iter()
                .all(|m| crate::fit::backend_compatible(m, &specs)),
            "a backend-incompatible model survived rankable_models"
        );
        assert!(
            rankable.len() < db.get_all_models().len(),
            "rankable_models should be a strict subset of the catalog"
        );
    }
}
