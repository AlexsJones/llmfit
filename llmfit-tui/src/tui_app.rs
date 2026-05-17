use llmfit_core::fit::{CalcConfig, FitLevel, ModelFit, SortColumn, backend_compatible};
use llmfit_core::hardware::SystemSpecs;
use llmfit_core::models::{Capability, ModelDatabase, UseCase};
use llmfit_core::plan::{PlanEstimate, PlanRequest, estimate_model_plan};
use llmfit_core::providers::{
    self, DockerModelRunnerProvider, LlamaCppProvider, LmStudioProvider, MlxProvider,
    ModelProvider, OllamaProvider, PullEvent, PullHandle, VllmProvider,
};
use llmfit_core::quality;

use std::collections::{HashMap, HashSet};
use std::sync::mpsc;
use std::thread;

use ratatui::widgets::TableState;

use crate::download_history::{DownloadHistory, DownloadRecord, DownloadResult};
use crate::filter_config::FilterConfig;
use crate::theme::Theme;

/// Messages sent from background provider-detection threads back to the main TUI.
pub enum ProviderDetectionMsg {
    Ollama {
        available: bool,
        binary_available: bool,
        installed: HashSet<String>,
        installed_count: usize,
        provider: OllamaProvider,
    },
    Mlx {
        available: bool,
        installed: HashSet<String>,
    },
    DockerMr {
        available: bool,
        installed: HashSet<String>,
        installed_count: usize,
    },
    LmStudio {
        available: bool,
        installed: HashSet<String>,
        installed_count: usize,
    },
    Vllm {
        available: bool,
        installed: HashSet<String>,
        installed_count: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputMode {
    Normal,
    Visual,
    Select,
    Search,
    Plan,
    ProviderPopup,
    UseCasePopup,
    CapabilityPopup,
    DownloadProviderPopup,
    QuantPopup,
    RunModePopup,
    ParamsBucketPopup,
    LicensePopup,
    RuntimePopup,
    HelpPopup,
    Simulation,
    AdvancedConfig,
    DownloadManager,
    FilterPopup,
    Benchmarks,
}

/// Fields in the Filter Popup modal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterPopupField {
    ParamsMin,
    ParamsMax,
    MemPctMin,
    MemPctMax,
    SortDirection,
    FitFilter,
}

impl FilterPopupField {
    pub fn next(self) -> Self {
        match self {
            Self::ParamsMin => Self::ParamsMax,
            Self::ParamsMax => Self::MemPctMin,
            Self::MemPctMin => Self::MemPctMax,
            Self::MemPctMax => Self::SortDirection,
            Self::SortDirection => Self::FitFilter,
            Self::FitFilter => Self::ParamsMin,
        }
    }

    pub fn prev(self) -> Self {
        match self {
            Self::ParamsMin => Self::FitFilter,
            Self::ParamsMax => Self::ParamsMin,
            Self::MemPctMin => Self::ParamsMax,
            Self::MemPctMax => Self::MemPctMin,
            Self::SortDirection => Self::MemPctMax,
            Self::FitFilter => Self::SortDirection,
        }
    }
}

/// Snapshot of filter state captured on popup open, restored on Esc.
#[derive(Debug, Clone)]
struct FilterSnapshot {
    params_min: String,
    params_max: String,
    mem_pct_min: String,
    mem_pct_max: String,
    sort_ascending: bool,
    fit_filter: FitFilter,
}

/// Fields in the Advanced Configuration modal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdvConfigField {
    Efficiency,       // Global efficiency factor
    FactorGpu,        // Run mode factor: GPU
    FactorCpuOffload, // Run mode factor: CPU offload
    FactorMoe,        // Run mode factor: MoE offload
    FactorTp,         // Run mode factor: Tensor parallel
    FactorCpuOnly,    // Run mode factor: CPU only
    ContextCap,       // Context window cap
}

impl AdvConfigField {
    fn next(self) -> Self {
        match self {
            AdvConfigField::Efficiency => AdvConfigField::FactorGpu,
            AdvConfigField::FactorGpu => AdvConfigField::FactorCpuOffload,
            AdvConfigField::FactorCpuOffload => AdvConfigField::FactorMoe,
            AdvConfigField::FactorMoe => AdvConfigField::FactorTp,
            AdvConfigField::FactorTp => AdvConfigField::FactorCpuOnly,
            AdvConfigField::FactorCpuOnly => AdvConfigField::ContextCap,
            AdvConfigField::ContextCap => AdvConfigField::Efficiency,
        }
    }

    fn prev(self) -> Self {
        match self {
            AdvConfigField::Efficiency => AdvConfigField::ContextCap,
            AdvConfigField::FactorGpu => AdvConfigField::Efficiency,
            AdvConfigField::FactorCpuOffload => AdvConfigField::FactorGpu,
            AdvConfigField::FactorMoe => AdvConfigField::FactorCpuOffload,
            AdvConfigField::FactorTp => AdvConfigField::FactorMoe,
            AdvConfigField::FactorCpuOnly => AdvConfigField::FactorTp,
            AdvConfigField::ContextCap => AdvConfigField::FactorCpuOnly,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimulationField {
    Ram,
    Vram,
    CpuCores,
}

impl SimulationField {
    pub fn next(self) -> Self {
        match self {
            SimulationField::Ram => SimulationField::Vram,
            SimulationField::Vram => SimulationField::CpuCores,
            SimulationField::CpuCores => SimulationField::Ram,
        }
    }

    pub fn prev(self) -> Self {
        match self {
            SimulationField::Ram => SimulationField::CpuCores,
            SimulationField::Vram => SimulationField::Ram,
            SimulationField::CpuCores => SimulationField::Vram,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownloadManagerFocus {
    Active,
    Config,
    History,
}

impl DownloadManagerFocus {
    pub fn next(self) -> Self {
        match self {
            Self::Active => Self::Config,
            Self::Config => Self::History,
            Self::History => Self::Active,
        }
    }

    pub fn prev(self) -> Self {
        match self {
            Self::Active => Self::History,
            Self::Config => Self::Active,
            Self::History => Self::Config,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanField {
    Context,
    Quant,
    KvQuant,
    TargetTps,
}

impl PlanField {
    fn next(self) -> Self {
        match self {
            PlanField::Context => PlanField::Quant,
            PlanField::Quant => PlanField::KvQuant,
            PlanField::KvQuant => PlanField::TargetTps,
            PlanField::TargetTps => PlanField::Context,
        }
    }

    fn prev(self) -> Self {
        match self {
            PlanField::Context => PlanField::TargetTps,
            PlanField::Quant => PlanField::Context,
            PlanField::KvQuant => PlanField::Quant,
            PlanField::TargetTps => PlanField::KvQuant,
        }
    }
}

/// Live-bench view mode (tab between results summary and routing matrix).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchViewMode {
    Results,
    Routing,
}

/// Messages sent from the bench worker thread to the UI.
#[allow(dead_code)]
pub enum BenchMsg {
    Progress(String),
    ModelStarted(String),
    TestDone {
        model: String,
        role: String,
        test_name: String,
    },
    RoleDone {
        model: String,
        role: String,
        quality: f64,
        speed: f64,
        tests_in_role: usize,
    },
    ModelDone(quality::ModelQualityResult),
    AllDone,
}

/// Per-model bench status for the live dashboard rows.
#[derive(Debug, Clone)]
pub struct BenchModelStatus {
    pub name: String,
    pub state: BenchModelState,
    pub roles_done: usize,
    pub roles_total: usize,
    pub current_role: String,
    pub last_quality: f64,
    pub last_speed: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum BenchModelState {
    Pending,
    Running,
    Complete,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitFilter {
    All,
    Perfect,
    Good,
    Marginal,
    TooTight,
    TurboQuantFit, // TooTight at fp16 but fits with TurboQuant KV compression
    Runnable,      // Perfect + Good + Marginal (excludes TooTight)
}

impl FitFilter {
    pub fn label(&self) -> &str {
        match self {
            FitFilter::All => "All",
            FitFilter::Perfect => "Perfect",
            FitFilter::Good => "Good",
            FitFilter::Marginal => "Marginal",
            FitFilter::TooTight => "Too Tight",
            FitFilter::TurboQuantFit => "TQ+ Fit",
            FitFilter::Runnable => "Runnable",
        }
    }

    pub fn from_label(s: &str) -> Self {
        match s {
            "Perfect" => FitFilter::Perfect,
            "Good" => FitFilter::Good,
            "Marginal" => FitFilter::Marginal,
            "Too Tight" => FitFilter::TooTight,
            "TQ+ Fit" => FitFilter::TurboQuantFit,
            "Runnable" => FitFilter::Runnable,
            _ => FitFilter::All,
        }
    }

    pub fn next(&self) -> Self {
        match self {
            FitFilter::All => FitFilter::Runnable,
            FitFilter::Runnable => FitFilter::Perfect,
            FitFilter::Perfect => FitFilter::Good,
            FitFilter::Good => FitFilter::Marginal,
            FitFilter::Marginal => FitFilter::TooTight,
            FitFilter::TooTight => FitFilter::TurboQuantFit,
            FitFilter::TurboQuantFit => FitFilter::All,
        }
    }
}

/// Filter by model availability / download readiness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AvailabilityFilter {
    All,
    HasGguf,   // Has GGUF download sources (unsloth, bartowski, etc.)
    Installed, // Already installed in a local runtime
}

impl AvailabilityFilter {
    pub fn label(&self) -> &str {
        match self {
            AvailabilityFilter::All => "All",
            AvailabilityFilter::HasGguf => "GGUF Avail",
            AvailabilityFilter::Installed => "Installed",
        }
    }

    pub fn from_label(s: &str) -> Self {
        match s {
            "GGUF Avail" => AvailabilityFilter::HasGguf,
            "Installed" => AvailabilityFilter::Installed,
            _ => AvailabilityFilter::All,
        }
    }

    pub fn next(&self) -> Self {
        match self {
            AvailabilityFilter::All => AvailabilityFilter::HasGguf,
            AvailabilityFilter::HasGguf => AvailabilityFilter::Installed,
            AvailabilityFilter::Installed => AvailabilityFilter::All,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpFilter {
    All,
    Tp2,
    Tp3,
    Tp4,
}

impl TpFilter {
    pub fn label(&self) -> &str {
        match self {
            TpFilter::All => "All",
            TpFilter::Tp2 => "TP=2",
            TpFilter::Tp3 => "TP=3",
            TpFilter::Tp4 => "TP=4",
        }
    }

    pub fn from_label(s: &str) -> Self {
        match s {
            "TP=2" => TpFilter::Tp2,
            "TP=3" => TpFilter::Tp3,
            "TP=4" => TpFilter::Tp4,
            _ => TpFilter::All,
        }
    }

    pub fn next(&self) -> Self {
        match self {
            TpFilter::All => TpFilter::Tp2,
            TpFilter::Tp2 => TpFilter::Tp3,
            TpFilter::Tp3 => TpFilter::Tp4,
            TpFilter::Tp4 => TpFilter::All,
        }
    }

    pub fn matches(&self, model: &llmfit_core::models::LlmModel) -> bool {
        match self {
            TpFilter::All => true,
            TpFilter::Tp2 => model.supports_tp(2),
            TpFilter::Tp3 => model.supports_tp(3),
            TpFilter::Tp4 => model.supports_tp(4),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownloadProvider {
    Ollama,
    Mlx,
    LlamaCpp,
    DockerModelRunner,
    LmStudio,
    Vllm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownloadCapability {
    Unknown,
    /// Bitfield: OLLAMA=1, LLAMACPP=2, DOCKER=4
    Known(u8),
}

pub const DL_OLLAMA: u8 = 0b0001;
pub const DL_LLAMACPP: u8 = 0b0010;
pub const DL_DOCKER: u8 = 0b0100;
pub const DL_LMSTUDIO: u8 = 0b1000;
pub const DL_VLLM: u8 = 0b1_0000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActivePullProvider {
    Ollama,
    Mlx,
    LlamaCpp,
    DockerModelRunner,
    LmStudio,
    Vllm,
}

impl ActivePullProvider {
    fn label(self) -> &'static str {
        match self {
            ActivePullProvider::Ollama => "Ollama",
            ActivePullProvider::Mlx => "MLX",
            ActivePullProvider::LlamaCpp => "llama.cpp",
            ActivePullProvider::DockerModelRunner => "Docker",
            ActivePullProvider::LmStudio => "LM Studio",
            ActivePullProvider::Vllm => "vLLM",
        }
    }
}

fn sort_column_from_label(s: &str) -> SortColumn {
    match s {
        "Score" => SortColumn::Score,
        "tok/s" => SortColumn::Tps,
        "Params" => SortColumn::Params,
        "Mem%" => SortColumn::MemPct,
        "Ctx" => SortColumn::Ctx,
        "Date" => SortColumn::ReleaseDate,
        "Use" => SortColumn::UseCase,
        _ => SortColumn::Score,
    }
}

pub struct App {
    pub should_quit: bool,
    pub input_mode: InputMode,
    pub search_query: String,
    pub cursor_position: usize,

    // Data
    pub specs: SystemSpecs,
    pub all_fits: Vec<ModelFit>,
    pub filtered_fits: Vec<usize>, // indices into all_fits
    pub providers: Vec<String>,
    pub selected_providers: Vec<bool>,
    pub use_cases: Vec<UseCase>,
    pub selected_use_cases: Vec<bool>,
    pub capabilities: Vec<Capability>,
    pub selected_capabilities: Vec<bool>,

    // Filters
    pub fit_filter: FitFilter,
    pub availability_filter: AvailabilityFilter,
    pub tp_filter: TpFilter,
    pub installed_first: bool,
    pub sort_column: SortColumn,
    pub sort_ascending: bool,

    // Table state
    pub selected_row: usize,
    pub table_state: TableState,

    // Detail view
    pub show_detail: bool,
    pub show_compare: bool,
    pub compare_mark_model: Option<String>,
    pub show_multi_compare: bool,
    pub compare_models: Vec<usize>, // indices into all_fits
    pub compare_scroll: usize,      // horizontal scroll for multi-compare
    pub show_plan: bool,
    plan_model_idx: Option<usize>,
    pub plan_field: PlanField,
    pub plan_context_input: String,
    pub plan_quant_input: String,
    pub plan_kv_quant_input: String,
    pub plan_target_tps_input: String,
    pub plan_cursor_position: usize,
    pub plan_estimate: Option<PlanEstimate>,
    pub plan_error: Option<String>,

    // Provider popup
    pub provider_cursor: usize,
    pub use_case_cursor: usize,
    pub capability_cursor: usize,
    pub download_provider_cursor: usize,
    pub download_provider_options: Vec<DownloadProvider>,
    pub download_provider_model: Option<String>,

    // Provider state
    pub ollama_available: bool,
    pub ollama_binary_available: bool,
    pub ollama_installed: HashSet<String>,
    pub ollama_installed_count: usize,
    ollama: OllamaProvider,
    pub mlx_available: bool,
    pub mlx_installed: HashSet<String>,
    mlx: MlxProvider,
    pub llamacpp_available: bool,
    pub llamacpp_installed: HashSet<String>,
    pub llamacpp_installed_count: usize,
    pub llamacpp_detection_hint: String,
    llamacpp: LlamaCppProvider,
    pub docker_mr_available: bool,
    pub docker_mr_installed: HashSet<String>,
    pub docker_mr_installed_count: usize,
    docker_mr: DockerModelRunnerProvider,
    pub lmstudio_available: bool,
    pub lmstudio_installed: HashSet<String>,
    pub lmstudio_installed_count: usize,
    lmstudio: LmStudioProvider,
    pub vllm_available: bool,
    pub vllm_installed: HashSet<String>,
    pub vllm_installed_count: usize,
    vllm: VllmProvider,

    // Download state
    pub pull_active: Option<PullHandle>,
    pub pull_status: Option<String>,
    pub pull_percent: Option<f64>,
    pub pull_model_name: Option<String>,
    pull_provider: Option<ActivePullProvider>,
    pub download_capabilities: HashMap<String, DownloadCapability>,
    download_capability_inflight: HashSet<String>,
    download_capability_tx: mpsc::Sender<(String, DownloadCapability)>,
    download_capability_rx: mpsc::Receiver<(String, DownloadCapability)>,
    /// Animation frame counter, incremented every tick while pulling.
    pub tick_count: u64,
    /// When true, the next 'd' press will confirm and start the download.
    pub confirm_download: bool,

    // Download manager view
    pub show_downloads: bool,
    pub dm_focus: DownloadManagerFocus,
    pub download_history: DownloadHistory,
    pub dm_history_cursor: usize,
    pub dm_history_scroll: usize,
    pub dm_confirm_delete: bool,
    pub dm_editing_dir: bool,
    pub dm_dir_input: String,
    pub dm_dir_cursor: usize,

    // Visual mode
    pub visual_anchor: Option<usize>,

    // Select mode
    pub select_column: usize,

    // Quant filter (popup)
    pub quants: Vec<String>,
    pub selected_quants: Vec<bool>,
    pub quant_cursor: usize,

    // RunMode filter (popup)
    pub run_modes: Vec<String>,
    pub selected_run_modes: Vec<bool>,
    pub run_mode_cursor: usize,

    // Params bucket filter (popup)
    pub params_buckets: Vec<String>,
    pub selected_params_buckets: Vec<bool>,
    pub params_bucket_cursor: usize,

    // License filter (popup)
    pub licenses: Vec<String>,
    pub selected_licenses: Vec<bool>,
    pub license_cursor: usize,

    // Runtime filter (popup)
    pub runtimes: Vec<String>,
    pub selected_runtimes: Vec<bool>,
    pub runtime_cursor: usize,

    // Help popup
    pub help_scroll: usize,

    // Hardware simulation
    pub real_specs: SystemSpecs,
    pub sim_active: bool,
    pub sim_field: SimulationField,
    pub sim_ram_input: String,
    pub sim_vram_input: String,
    pub sim_cpu_input: String,
    pub sim_cursor_position: usize,
    context_limit: Option<u32>,

    // Theme
    pub theme: Theme,

    // Advanced Configuration
    pub calc_config: CalcConfig,
    pub adv_config_field: AdvConfigField,
    pub adv_config_cursor_position: usize,
    pub adv_config_dirty: bool,
    pub adv_config_efficiency_input: String,
    pub adv_config_eff_factor_gpu: String,
    pub adv_config_eff_factor_cpu_offload: String,
    pub adv_config_eff_factor_moe: String,
    pub adv_config_eff_factor_tp: String,
    pub adv_config_eff_factor_cpu_only: String,
    pub adv_config_context_cap_input: String,

    // Filter Popup
    pub filter_field: FilterPopupField,
    pub filter_cursor_position: usize,
    pub filter_params_min_input: String,
    pub filter_params_max_input: String,
    pub filter_mem_pct_min_input: String,
    pub filter_mem_pct_max_input: String,
    pub filter_sort_ascending: bool,

    // Snapshot of filter state when popup is opened — restored on Esc.
    filter_snapshot: Option<FilterSnapshot>,

    /// How many models we silently dropped because they can't run on this
    /// hardware — shown in the system bar so users aren't left wondering
    /// why the list looks shorter than expected.
    pub backend_hidden_count: usize,

    // Benchmarks view (localmaxxing.com)
    pub show_benchmarks: bool,
    pub bench_entries: Vec<llmfit_core::benchmarks::LeaderboardEntry>,
    pub bench_cursor: usize,
    pub bench_scroll: usize,
    pub bench_loading: bool,
    pub bench_error: Option<String>,
    pub bench_total: u64,
    pub bench_api_key: Option<String>,
    /// Label for the currently selected hardware (None = auto-detected).
    pub bench_hw_label: Option<String>,
    /// Index into HardwarePreset::all() when the picker is open.
    pub bench_hw_picker_cursor: usize,
    pub bench_hw_picker_open: bool,

    // Live inference-bench view (llmfit bench — distinct from community benchmarks above)
    pub show_bench: bool,
    pub bench_model_status: Vec<BenchModelStatus>,
    pub bench_results: Vec<quality::ModelQualityResult>,
    pub bench_routing: Vec<quality::RoutingRecommendation>,
    pub bench_runner_ups: Vec<quality::RoutingRecommendation>,
    pub bench_running: bool,
    pub bench_progress: String,
    pub live_bench_scroll: usize,
    pub bench_selected_row: usize,
    pub bench_show_detail: bool,
    pub bench_view_mode: BenchViewMode,
    pub bench_confirm_quit: bool,
    pub bench_tests_done: usize,
    pub bench_tests_total: usize,
    bench_rx: Option<mpsc::Receiver<BenchMsg>>,

    // Background provider detection
    provider_detection_rx: mpsc::Receiver<ProviderDetectionMsg>,
    /// True while background provider detection is still in progress.
    pub providers_loading: bool,
}

impl App {
    pub fn with_specs_and_context(specs: SystemSpecs, context_limit: Option<u32>) -> Self {
        let real_specs = specs.clone();
        let db = ModelDatabase::new();

        // Detect llama.cpp synchronously (local filesystem check, fast)
        let mut llamacpp = LlamaCppProvider::new();
        if let Some(ref dir) = FilterConfig::load().download_dir {
            let path = std::path::PathBuf::from(dir);
            if path.is_dir() {
                llamacpp.set_models_dir(path);
            }
        }
        let llamacpp_available = llamacpp.is_available();
        let llamacpp_detection_hint = llamacpp.detection_hint().to_string();
        let (llamacpp_installed, llamacpp_installed_count) = llamacpp.installed_models_counted();

        // Start with empty provider state — detection runs in background
        let ollama = OllamaProvider::new();
        let ollama_available = false;
        let ollama_binary_available = false;
        let ollama_installed = HashSet::new();
        let ollama_installed_count = 0;
        let mlx = MlxProvider::new();
        let mlx_available = false;
        let mlx_installed = HashSet::new();
        let docker_mr = DockerModelRunnerProvider::new();
        let docker_mr_available = false;
        let docker_mr_installed = HashSet::new();
        let docker_mr_installed_count = 0;
        let lmstudio = LmStudioProvider::new();
        let lmstudio_available = false;
        let lmstudio_installed = HashSet::new();
        let lmstudio_installed_count = 0;
        let vllm = VllmProvider::new();
        let vllm_available = false;
        let vllm_installed = HashSet::new();
        let vllm_installed_count = 0;

        // Spawn background provider detection for network-based providers
        let (provider_tx, provider_detection_rx) = mpsc::channel();
        {
            let tx = provider_tx.clone();
            thread::spawn(move || {
                let mut ollama = OllamaProvider::new();
                let (available, installed, installed_count) = ollama.detect_with_installed();
                let binary_available = command_exists("ollama");
                let _ = tx.send(ProviderDetectionMsg::Ollama {
                    available,
                    binary_available,
                    installed,
                    installed_count,
                    provider: ollama,
                });
            });
        }
        {
            let tx = provider_tx.clone();
            thread::spawn(move || {
                let mlx = MlxProvider::new();
                let (available, installed) = mlx.detect_with_installed();
                let _ = tx.send(ProviderDetectionMsg::Mlx {
                    available,
                    installed,
                });
            });
        }
        {
            let tx = provider_tx.clone();
            thread::spawn(move || {
                let docker_mr = DockerModelRunnerProvider::new();
                let (available, installed, installed_count) = docker_mr.detect_with_installed();
                let _ = tx.send(ProviderDetectionMsg::DockerMr {
                    available,
                    installed,
                    installed_count,
                });
            });
        }
        {
            let tx = provider_tx.clone();
            thread::spawn(move || {
                let lmstudio = LmStudioProvider::new();
                let (available, installed, installed_count) = lmstudio.detect_with_installed();
                let _ = tx.send(ProviderDetectionMsg::LmStudio {
                    available,
                    installed,
                    installed_count,
                });
            });
        }
        {
            let tx = provider_tx;
            thread::spawn(move || {
                let vllm = VllmProvider::new();
                let (available, installed, installed_count) = vllm.detect_with_installed();
                let _ = tx.send(ProviderDetectionMsg::Vllm {
                    available,
                    installed,
                    installed_count,
                });
            });
        }

        // Track how many we're skipping so the UI can surface it.
        let backend_hidden_count = db
            .get_all_models()
            .iter()
            .filter(|m| !backend_compatible(m, &specs))
            .count();

        // Only analyze models that can actually run on this hardware.
        let mut all_fits: Vec<ModelFit> = db
            .get_all_models()
            .iter()
            .filter(|m| backend_compatible(m, &specs))
            .map(|m| {
                let mut fit = ModelFit::analyze_with_context_limit(m, &specs, context_limit);
                fit.installed = providers::is_model_installed(&m.name, &ollama_installed)
                    || providers::is_model_installed_mlx(&m.name, &mlx_installed)
                    || providers::is_model_installed_llamacpp(&m.name, &llamacpp_installed)
                    || providers::is_model_installed_docker_mr(&m.name, &docker_mr_installed)
                    || providers::is_model_installed_lmstudio(&m.name, &lmstudio_installed)
                    || providers::is_model_installed_vllm(&m.name, &vllm_installed);
                fit
            })
            .collect();

        // Sort by fit level then RAM usage
        all_fits = llmfit_core::fit::rank_models_by_fit(all_fits);

        // Extract unique providers
        let mut model_providers: Vec<String> = all_fits
            .iter()
            .map(|f| f.model.provider.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        model_providers.sort();

        let mut selected_providers = vec![true; model_providers.len()];
        let model_use_cases = [
            UseCase::General,
            UseCase::Coding,
            UseCase::Reasoning,
            UseCase::Chat,
            UseCase::Multimodal,
            UseCase::Embedding,
        ]
        .into_iter()
        .filter(|uc| all_fits.iter().any(|f| f.use_case == *uc))
        .collect::<Vec<_>>();
        let mut selected_use_cases = vec![true; model_use_cases.len()];

        let model_capabilities = Capability::all().to_vec();
        let mut selected_capabilities = vec![true; model_capabilities.len()];

        // Extract unique quantizations
        let mut model_quants: Vec<String> = all_fits
            .iter()
            .map(|f| f.best_quant.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        model_quants.sort();
        let mut selected_quants = vec![true; model_quants.len()];

        // Run modes
        let model_run_modes = vec![
            "GPU".to_string(),
            "MoE".to_string(),
            "CPU+GPU".to_string(),
            "CPU".to_string(),
        ];
        let mut selected_run_modes = vec![true; model_run_modes.len()];

        // Params buckets
        let params_buckets = vec![
            "<3B".to_string(),
            "3-7B".to_string(),
            "7-14B".to_string(),
            "14-30B".to_string(),
            "30-70B".to_string(),
            "70B+".to_string(),
        ];
        let mut selected_params_buckets = vec![true; params_buckets.len()];

        // Extract unique licenses (including "Unknown" for models without one)
        let mut model_licenses: Vec<String> = all_fits
            .iter()
            .map(|f| {
                f.model
                    .license
                    .clone()
                    .unwrap_or_else(|| "Unknown".to_string())
            })
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        // Move "Unknown" to the end if present
        if let Some(pos) = model_licenses.iter().position(|l| l == "Unknown") {
            let unknown = model_licenses.remove(pos);
            model_licenses.push(unknown);
        }
        let mut selected_licenses = vec![true; model_licenses.len()];

        // Static runtime options — filter by compatibility, not assigned runtime
        let model_runtimes = vec![
            "llama.cpp".to_string(),
            "MLX".to_string(),
            "vLLM".to_string(),
        ];
        let mut selected_runtimes = vec![true; model_runtimes.len()];

        // ── Restore persisted filters ────────────────────────────────
        let saved = FilterConfig::load();

        let fit_filter = saved
            .fit_filter
            .as_deref()
            .map(FitFilter::from_label)
            .unwrap_or(FitFilter::All);
        let availability_filter = saved
            .availability_filter
            .as_deref()
            .map(AvailabilityFilter::from_label)
            .unwrap_or(AvailabilityFilter::All);
        let tp_filter = saved
            .tp_filter
            .as_deref()
            .map(TpFilter::from_label)
            .unwrap_or(TpFilter::All);
        let sort_column = saved
            .sort_column
            .as_deref()
            .map(sort_column_from_label)
            .unwrap_or(SortColumn::Score);
        let sort_ascending = saved.sort_ascending.unwrap_or(false);
        let installed_first = saved.installed_first.unwrap_or(false);
        let search_query = saved.search_query.clone().unwrap_or_default();
        let cursor_position = search_query.len();

        if let Some(ref map) = saved.providers {
            FilterConfig::apply_map(&model_providers, &mut selected_providers, map);
        }
        if let Some(ref map) = saved.use_cases {
            let names: Vec<String> = model_use_cases
                .iter()
                .map(|uc| uc.label().to_string())
                .collect();
            FilterConfig::apply_map(&names, &mut selected_use_cases, map);
        }
        if let Some(ref map) = saved.capabilities {
            let names: Vec<String> = model_capabilities
                .iter()
                .map(|c| c.label().to_string())
                .collect();
            FilterConfig::apply_map(&names, &mut selected_capabilities, map);
        }
        if let Some(ref map) = saved.quants {
            FilterConfig::apply_map(&model_quants, &mut selected_quants, map);
        }
        if let Some(ref map) = saved.run_modes {
            FilterConfig::apply_map(&model_run_modes, &mut selected_run_modes, map);
        }
        if let Some(ref map) = saved.params_buckets {
            FilterConfig::apply_map(&params_buckets, &mut selected_params_buckets, map);
        }
        if let Some(ref map) = saved.licenses {
            FilterConfig::apply_map(&model_licenses, &mut selected_licenses, map);
        }
        if let Some(ref map) = saved.runtimes {
            FilterConfig::apply_map(&model_runtimes, &mut selected_runtimes, map);
        }

        let filtered_count = all_fits.len();

        let (download_capability_tx, download_capability_rx) = mpsc::channel();

        let mut app = App {
            should_quit: false,
            input_mode: InputMode::Normal,
            search_query,
            cursor_position,
            specs,
            all_fits,
            filtered_fits: (0..filtered_count).collect(),
            providers: model_providers,
            selected_providers,
            use_cases: model_use_cases,
            selected_use_cases,
            capabilities: model_capabilities,
            selected_capabilities,
            fit_filter,
            availability_filter,
            tp_filter,
            installed_first,
            sort_column,
            sort_ascending,
            selected_row: 0,
            table_state: TableState::default(),
            show_detail: false,
            show_compare: false,
            compare_mark_model: None,
            show_multi_compare: false,
            compare_models: Vec::new(),
            compare_scroll: 0,
            show_plan: false,
            plan_model_idx: None,
            plan_field: PlanField::Context,
            plan_context_input: String::new(),
            plan_quant_input: String::new(),
            plan_kv_quant_input: String::new(),
            plan_target_tps_input: String::new(),
            plan_cursor_position: 0,
            plan_estimate: None,
            plan_error: None,
            provider_cursor: 0,
            use_case_cursor: 0,
            capability_cursor: 0,
            download_provider_cursor: 0,
            download_provider_options: Vec::new(),
            download_provider_model: None,
            ollama_available,
            ollama_binary_available,
            ollama_installed,
            ollama_installed_count,
            ollama,
            mlx_available,
            mlx_installed,
            mlx,
            llamacpp_available,
            llamacpp_installed,
            llamacpp_installed_count,
            llamacpp_detection_hint,
            llamacpp,
            docker_mr_available,
            docker_mr_installed,
            docker_mr_installed_count,
            docker_mr,
            lmstudio_available,
            lmstudio_installed,
            lmstudio_installed_count,
            lmstudio,
            vllm_available,
            vllm_installed,
            vllm_installed_count,
            vllm,
            pull_active: None,
            pull_status: None,
            pull_percent: None,
            pull_model_name: None,
            pull_provider: None,
            download_capabilities: HashMap::new(),
            download_capability_inflight: HashSet::new(),
            download_capability_tx,
            download_capability_rx,
            tick_count: 0,
            confirm_download: false,
            show_downloads: false,
            dm_focus: DownloadManagerFocus::History,
            download_history: DownloadHistory::load(),
            dm_history_cursor: 0,
            dm_history_scroll: 0,
            dm_confirm_delete: false,
            dm_editing_dir: false,
            dm_dir_input: String::new(),
            dm_dir_cursor: 0,
            visual_anchor: None,
            select_column: 2, // start on Model column
            quants: model_quants,
            selected_quants,
            quant_cursor: 0,
            run_modes: model_run_modes,
            selected_run_modes,
            run_mode_cursor: 0,
            params_buckets,
            selected_params_buckets,
            params_bucket_cursor: 0,
            licenses: model_licenses,
            selected_licenses,
            license_cursor: 0,
            runtimes: model_runtimes,
            selected_runtimes,
            runtime_cursor: 0,
            help_scroll: 0,
            real_specs,
            sim_active: false,
            sim_field: SimulationField::Ram,
            sim_ram_input: String::new(),
            sim_vram_input: String::new(),
            sim_cpu_input: String::new(),
            sim_cursor_position: 0,
            context_limit,
            theme: Theme::load(),
            backend_hidden_count,
            // Advanced configuration defaults
            calc_config: CalcConfig::default(),
            adv_config_field: AdvConfigField::Efficiency,
            adv_config_cursor_position: 0,
            adv_config_dirty: false,
            adv_config_efficiency_input: "0.55".to_string(),
            adv_config_eff_factor_gpu: "1.0".to_string(),
            adv_config_eff_factor_cpu_offload: "0.5".to_string(),
            adv_config_eff_factor_moe: "0.8".to_string(),
            adv_config_eff_factor_tp: "0.9".to_string(),
            adv_config_eff_factor_cpu_only: "0.3".to_string(),
            adv_config_context_cap_input: String::new(), // empty = use default
            // Filter popup defaults
            filter_field: FilterPopupField::ParamsMin,
            filter_cursor_position: 0,
            filter_params_min_input: String::new(),
            filter_params_max_input: String::new(),
            filter_mem_pct_min_input: String::new(),
            filter_mem_pct_max_input: String::new(),
            filter_sort_ascending: sort_ascending,
            filter_snapshot: None,
            // Benchmarks
            show_benchmarks: false,
            bench_entries: Vec::new(),
            bench_cursor: 0,
            bench_scroll: 0,
            bench_loading: false,
            bench_error: None,
            bench_total: 0,
            bench_api_key: std::env::var("LOCALMAXXING_API_KEY").ok(),
            bench_hw_label: None,
            bench_hw_picker_cursor: 0,
            bench_hw_picker_open: false,
            // Live inference-bench view
            show_bench: false,
            bench_model_status: Vec::new(),
            bench_results: Vec::new(),
            bench_routing: Vec::new(),
            bench_runner_ups: Vec::new(),
            bench_running: false,
            bench_progress: String::new(),
            live_bench_scroll: 0,
            bench_selected_row: 0,
            bench_show_detail: false,
            bench_view_mode: BenchViewMode::Results,
            bench_confirm_quit: false,
            bench_tests_done: 0,
            bench_tests_total: 0,
            bench_rx: None,
            provider_detection_rx,
            providers_loading: true,
        };

        // Restore persisted range filters
        let saved = FilterConfig::load();
        if let Some(ref v) = saved.filter_params_min {
            app.filter_params_min_input = v.clone();
        }
        if let Some(ref v) = saved.filter_params_max {
            app.filter_params_max_input = v.clone();
        }
        if let Some(ref v) = saved.filter_mem_pct_min {
            app.filter_mem_pct_min_input = v.clone();
        }
        if let Some(ref v) = saved.filter_mem_pct_max {
            app.filter_mem_pct_max_input = v.clone();
        }

        app.apply_filters();
        app.re_sort();
        app.enqueue_capability_probes_for_visible(24);
        app
    }

    /// Persist the current filter state to disk.
    pub fn save_filters(&self) {
        let use_case_names: Vec<String> = self
            .use_cases
            .iter()
            .map(|uc| uc.label().to_string())
            .collect();
        let capability_names: Vec<String> = self
            .capabilities
            .iter()
            .map(|c| c.label().to_string())
            .collect();

        let config = FilterConfig {
            fit_filter: Some(self.fit_filter.label().to_string()),
            availability_filter: Some(self.availability_filter.label().to_string()),
            tp_filter: Some(self.tp_filter.label().to_string()),
            sort_column: Some(self.sort_column.label().to_string()),
            sort_ascending: Some(self.sort_ascending),
            installed_first: Some(self.installed_first),
            search_query: if self.search_query.is_empty() {
                None
            } else {
                Some(self.search_query.clone())
            },
            providers: Some(FilterConfig::build_map(
                &self.providers,
                &self.selected_providers,
            )),
            use_cases: Some(FilterConfig::build_map(
                &use_case_names,
                &self.selected_use_cases,
            )),
            capabilities: Some(FilterConfig::build_map(
                &capability_names,
                &self.selected_capabilities,
            )),
            quants: Some(FilterConfig::build_map(&self.quants, &self.selected_quants)),
            run_modes: Some(FilterConfig::build_map(
                &self.run_modes,
                &self.selected_run_modes,
            )),
            params_buckets: Some(FilterConfig::build_map(
                &self.params_buckets,
                &self.selected_params_buckets,
            )),
            licenses: Some(FilterConfig::build_map(
                &self.licenses,
                &self.selected_licenses,
            )),
            runtimes: Some(FilterConfig::build_map(
                &self.runtimes,
                &self.selected_runtimes,
            )),
            // Range filters
            filter_params_min: if self.filter_params_min_input.is_empty() {
                None
            } else {
                Some(self.filter_params_min_input.clone())
            },
            filter_params_max: if self.filter_params_max_input.is_empty() {
                None
            } else {
                Some(self.filter_params_max_input.clone())
            },
            filter_mem_pct_min: if self.filter_mem_pct_min_input.is_empty() {
                None
            } else {
                Some(self.filter_mem_pct_min_input.clone())
            },
            filter_mem_pct_max: if self.filter_mem_pct_max_input.is_empty() {
                None
            } else {
                Some(self.filter_mem_pct_max_input.clone())
            },
            // Preserve existing download_dir setting
            download_dir: FilterConfig::load().download_dir,
        };
        config.save();
    }

    pub fn apply_filters(&mut self) {
        let query = self.search_query.to_lowercase();
        // Split query into space-separated terms for fuzzy matching
        let terms: Vec<&str> = query.split_whitespace().collect();

        self.filtered_fits = self
            .all_fits
            .iter()
            .enumerate()
            .filter(|(_, fit)| {
                // Search filter: all terms must match (fuzzy/AND logic)
                let matches_search = if terms.is_empty() {
                    true
                } else {
                    let caps_text = fit
                        .model
                        .capabilities
                        .iter()
                        .map(|c| c.label().to_lowercase())
                        .collect::<Vec<_>>()
                        .join(" ");
                    // Combine all searchable fields into one string
                    let license_text = fit.model.license.as_deref().unwrap_or("").to_lowercase();
                    let searchable = format!(
                        "{} {} {} {} {} {} {}",
                        fit.model.name.to_lowercase(),
                        fit.model.provider.to_lowercase(),
                        fit.model.parameter_count.to_lowercase(),
                        fit.model.use_case.to_lowercase(),
                        fit.use_case.label().to_lowercase(),
                        caps_text,
                        license_text
                    );
                    // All terms must be present (AND logic)
                    terms.iter().all(|term| searchable.contains(term))
                };

                // Provider filter
                let provider_idx = self.providers.iter().position(|p| p == &fit.model.provider);
                let matches_provider = provider_idx
                    .map(|idx| self.selected_providers[idx])
                    .unwrap_or(true);
                let use_case_idx = self.use_cases.iter().position(|uc| *uc == fit.use_case);
                let matches_use_case = use_case_idx
                    .map(|idx| self.selected_use_cases[idx])
                    .unwrap_or(true);

                // Hide MLX-only models on non-Apple Silicon systems
                let is_apple_silicon = self.specs.backend
                    == llmfit_core::hardware::GpuBackend::Metal
                    && self.specs.unified_memory;
                if fit.model.is_mlx_only() && !is_apple_silicon {
                    return false;
                }

                // Fit filter
                let matches_fit = match self.fit_filter {
                    FitFilter::All => true,
                    FitFilter::Perfect => fit.fit_level == FitLevel::Perfect,
                    FitFilter::Good => fit.fit_level == FitLevel::Good,
                    FitFilter::Marginal => fit.fit_level == FitLevel::Marginal,
                    FitFilter::TooTight => fit.fit_level == FitLevel::TooTight,
                    FitFilter::TurboQuantFit => fit.fits_with_turboquant,
                    FitFilter::Runnable => fit.fit_level != FitLevel::TooTight,
                };

                // Availability filter
                let matches_availability = match self.availability_filter {
                    AvailabilityFilter::All => true,
                    AvailabilityFilter::HasGguf => !fit.model.gguf_sources.is_empty(),
                    AvailabilityFilter::Installed => fit.installed,
                };

                // Capability filter
                let matches_capability = {
                    let all_selected = self.selected_capabilities.iter().all(|&s| s);
                    if all_selected {
                        true
                    } else {
                        self.capabilities
                            .iter()
                            .zip(self.selected_capabilities.iter())
                            .filter(|(_, sel)| **sel)
                            .any(|(cap, _)| fit.model.capabilities.contains(cap))
                    }
                };

                // Quant filter
                let matches_quant = {
                    let all_selected = self.selected_quants.iter().all(|&s| s);
                    if all_selected {
                        true
                    } else {
                        self.quants
                            .iter()
                            .zip(self.selected_quants.iter())
                            .any(|(q, &sel)| sel && *q == fit.best_quant)
                    }
                };

                // RunMode filter
                let matches_run_mode = {
                    let all_selected = self.selected_run_modes.iter().all(|&s| s);
                    if all_selected {
                        true
                    } else {
                        let mode_text = fit.run_mode_text();
                        self.run_modes
                            .iter()
                            .zip(self.selected_run_modes.iter())
                            .any(|(m, &sel)| sel && *m == mode_text)
                    }
                };

                // Params bucket filter
                let matches_params_bucket = {
                    let all_selected = self.selected_params_buckets.iter().all(|&s| s);
                    if all_selected {
                        true
                    } else {
                        let params = fit.model.params_b();
                        let bucket_idx = if params < 3.0 {
                            0
                        } else if params < 7.0 {
                            1
                        } else if params < 14.0 {
                            2
                        } else if params < 30.0 {
                            3
                        } else if params < 70.0 {
                            4
                        } else {
                            5
                        };
                        self.selected_params_buckets
                            .get(bucket_idx)
                            .copied()
                            .unwrap_or(true)
                    }
                };

                let matches_tp = self.tp_filter.matches(&fit.model);

                // License filter
                let matches_license = {
                    let all_selected = self.selected_licenses.iter().all(|&s| s);
                    if all_selected || self.licenses.is_empty() {
                        true
                    } else {
                        let model_lic = fit.model.license.as_deref().unwrap_or("Unknown");
                        self.licenses
                            .iter()
                            .zip(self.selected_licenses.iter())
                            .any(|(l, &sel)| sel && l == model_lic)
                    }
                };

                // Runtime filter — match by compatibility, not assigned runtime
                let matches_runtime = {
                    let all_selected = self.selected_runtimes.iter().all(|&s| s);
                    if all_selected {
                        true
                    } else {
                        let is_apple_silicon = self.specs.backend
                            == llmfit_core::hardware::GpuBackend::Metal
                            && self.specs.unified_memory;
                        // Determine which runtimes this model is compatible with
                        let compat_llamacpp =
                            !fit.model.is_mlx_only() && !fit.model.is_prequantized();
                        let compat_mlx = is_apple_silicon
                            && (fit.model.is_mlx_model()
                                || (!fit.model.is_prequantized()
                                    && !fit.model.gguf_sources.is_empty()));
                        let compat_vllm = fit.model.is_prequantized();
                        // Check if any selected runtime matches
                        self.runtimes
                            .iter()
                            .zip(self.selected_runtimes.iter())
                            .any(|(r, &sel)| {
                                sel && match r.as_str() {
                                    "llama.cpp" => compat_llamacpp,
                                    "MLX" => compat_mlx,
                                    "vLLM" => compat_vllm,
                                    _ => false,
                                }
                            })
                    }
                };

                // Params range filter
                let matches_params_range = {
                    let params_b = fit.model.params_b();
                    let min_ok = self.filter_params_min_input.is_empty()
                        || params_b >= self.filter_params_min_input.parse::<f64>().unwrap_or(0.0);
                    let max_ok = self.filter_params_max_input.is_empty()
                        || params_b
                            <= self
                                .filter_params_max_input
                                .parse::<f64>()
                                .unwrap_or(f64::MAX);
                    min_ok && max_ok
                };

                // Memory % range filter
                let matches_mem_range = {
                    let mem_pct = fit.utilization_pct;
                    let min_ok = self.filter_mem_pct_min_input.is_empty()
                        || mem_pct >= self.filter_mem_pct_min_input.parse::<f64>().unwrap_or(0.0);
                    let max_ok = self.filter_mem_pct_max_input.is_empty()
                        || mem_pct
                            <= self
                                .filter_mem_pct_max_input
                                .parse::<f64>()
                                .unwrap_or(f64::MAX);
                    min_ok && max_ok
                };

                matches_search
                    && matches_provider
                    && matches_use_case
                    && matches_fit
                    && matches_availability
                    && matches_capability
                    && matches_quant
                    && matches_run_mode
                    && matches_params_bucket
                    && matches_tp
                    && matches_license
                    && matches_runtime
                    && matches_params_range
                    && matches_mem_range
            })
            .map(|(i, _)| i)
            .collect();

        // Clamp selection
        if self.filtered_fits.is_empty() {
            self.selected_row = 0;
        } else if self.selected_row >= self.filtered_fits.len() {
            self.selected_row = self.filtered_fits.len() - 1;
        }
        self.enqueue_capability_probes_for_visible(24);
    }

    pub fn selected_fit(&self) -> Option<&ModelFit> {
        self.filtered_fits
            .get(self.selected_row)
            .map(|&idx| &self.all_fits[idx])
    }

    pub fn move_up(&mut self) {
        self.confirm_download = false;
        if self.selected_row > 0 {
            self.selected_row -= 1;
        }
        self.enqueue_capability_probes_for_visible(24);
    }

    pub fn move_down(&mut self) {
        self.confirm_download = false;
        if !self.filtered_fits.is_empty() && self.selected_row < self.filtered_fits.len() - 1 {
            self.selected_row += 1;
        }
        self.enqueue_capability_probes_for_visible(24);
    }

    pub fn page_up(&mut self) {
        self.confirm_download = false;
        self.selected_row = self.selected_row.saturating_sub(10);
        self.enqueue_capability_probes_for_visible(24);
    }

    pub fn page_down(&mut self) {
        self.confirm_download = false;
        if !self.filtered_fits.is_empty() {
            self.selected_row = (self.selected_row + 10).min(self.filtered_fits.len() - 1);
        }
        self.enqueue_capability_probes_for_visible(24);
    }

    pub fn half_page_up(&mut self) {
        self.selected_row = self.selected_row.saturating_sub(5);
        self.enqueue_capability_probes_for_visible(24);
    }

    pub fn half_page_down(&mut self) {
        if !self.filtered_fits.is_empty() {
            self.selected_row = (self.selected_row + 5).min(self.filtered_fits.len() - 1);
        }
        self.enqueue_capability_probes_for_visible(24);
    }

    pub fn cycle_top_bottom(&mut self) {
        if !self.filtered_fits.is_empty() && self.selected_row == self.filtered_fits.len() - 1 {
            self.selected_row = 0;
        } else {
            self.selected_row = self.filtered_fits.len().saturating_sub(1);
        }
        self.enqueue_capability_probes_for_visible(24);
    }

    pub fn cycle_fit_filter(&mut self) {
        self.fit_filter = self.fit_filter.next();
        self.apply_filters();
    }

    pub fn cycle_availability_filter(&mut self) {
        self.availability_filter = self.availability_filter.next();
        self.apply_filters();
    }

    pub fn cycle_tp_filter(&mut self) {
        self.tp_filter = self.tp_filter.next();
        self.apply_filters();
    }

    /// Returns true when any filter beyond the fit-level filter is active
    /// (range filters, sub-selection popups, search, etc.).
    pub fn has_advanced_filters_active(&self) -> bool {
        let has_range = !self.filter_params_min_input.is_empty()
            || !self.filter_params_max_input.is_empty()
            || !self.filter_mem_pct_min_input.is_empty()
            || !self.filter_mem_pct_max_input.is_empty();
        let has_search = !self.search_query.is_empty();
        let has_provider_filter = !self.selected_providers.iter().all(|&s| s);
        let has_use_case_filter = !self.selected_use_cases.iter().all(|&s| s);
        let has_capability_filter = !self.selected_capabilities.iter().all(|&s| s);
        let has_quant_filter = !self.selected_quants.iter().all(|&s| s);
        let has_run_mode_filter = !self.selected_run_modes.iter().all(|&s| s);
        let has_params_bucket_filter = !self.selected_params_buckets.iter().all(|&s| s);
        let has_license_filter = !self.selected_licenses.iter().all(|&s| s);
        let has_runtime_filter = !self.selected_runtimes.iter().all(|&s| s);
        has_range
            || has_search
            || has_provider_filter
            || has_use_case_filter
            || has_capability_filter
            || has_quant_filter
            || has_run_mode_filter
            || has_params_bucket_filter
            || has_license_filter
            || has_runtime_filter
    }

    pub fn cycle_sort_column(&mut self) {
        self.sort_column = self.sort_column.next();
        self.sort_ascending = false;
        self.re_sort();
    }

    pub fn cycle_theme(&mut self) {
        self.theme = self.theme.next();
        self.theme.save();
    }

    pub fn enter_search(&mut self) {
        self.input_mode = InputMode::Search;
    }

    pub fn exit_search(&mut self) {
        self.input_mode = InputMode::Normal;
    }

    pub fn search_input(&mut self, c: char) {
        self.search_query.insert(self.cursor_position, c);
        self.cursor_position += 1;
        self.apply_filters();
    }

    pub fn search_backspace(&mut self) {
        if self.cursor_position > 0 {
            self.cursor_position -= 1;
            self.search_query.remove(self.cursor_position);
            self.apply_filters();
        }
    }

    pub fn search_delete(&mut self) {
        if self.cursor_position < self.search_query.len() {
            self.search_query.remove(self.cursor_position);
            self.apply_filters();
        }
    }

    pub fn clear_search(&mut self) {
        self.search_query.clear();
        self.cursor_position = 0;
        self.apply_filters();
    }

    pub fn toggle_downloads(&mut self) {
        self.show_plan = false;
        self.show_compare = false;
        self.show_multi_compare = false;
        self.show_detail = false;
        self.show_benchmarks = false;
        self.show_downloads = !self.show_downloads;
        if self.show_downloads {
            self.input_mode = InputMode::DownloadManager;
            self.dm_focus = DownloadManagerFocus::History;
            self.dm_confirm_delete = false;
            self.dm_editing_dir = false;
        } else {
            self.input_mode = InputMode::Normal;
        }
    }

    pub fn close_downloads(&mut self) {
        self.show_downloads = false;
        self.dm_confirm_delete = false;
        self.dm_editing_dir = false;
        self.input_mode = InputMode::Normal;
    }

    pub fn llamacpp_models_dir(&self) -> &std::path::Path {
        self.llamacpp.models_dir()
    }

    pub fn start_editing_download_dir(&mut self) {
        self.dm_dir_input = self.llamacpp.models_dir().display().to_string();
        self.dm_dir_cursor = self.dm_dir_input.len();
        self.dm_editing_dir = true;
    }

    pub fn apply_download_dir(&mut self) {
        let path = std::path::PathBuf::from(&self.dm_dir_input);
        if let Err(e) = std::fs::create_dir_all(&path) {
            self.pull_status = Some(format!("Invalid path: {}", e));
            return;
        }
        self.llamacpp.set_models_dir(path);
        self.pull_status = Some(format!("Models dir set to: {}", self.dm_dir_input));
        // Persist via FilterConfig
        let mut saved = FilterConfig::load();
        saved.download_dir = Some(self.dm_dir_input.clone());
        saved.save();
        self.refresh_installed();
    }

    pub fn delete_selected_download(&mut self) {
        let len = self.download_history.records.len();
        if len == 0 || self.dm_history_cursor >= len {
            return;
        }
        // History is displayed newest-first, so map display index to records index
        let actual_idx = len - 1 - self.dm_history_cursor;
        let record = &self.download_history.records[actual_idx];
        let provider_name = record.provider.clone();
        let model_name = record.model_name.clone();
        let file_path = record.file_path.clone();
        let was_error = matches!(record.result, DownloadResult::Error(_));

        // For failed downloads, just remove the history entry — there's nothing
        // on disk or in the provider to clean up.
        if was_error {
            self.pull_status = Some(format!("Removed {} from history", model_name));
            self.download_history.remove(actual_idx);
            self.clamp_dm_cursor();
            return;
        }

        // For successful downloads, attempt provider-level deletion
        let result = match provider_name.as_str() {
            "Ollama" => self.ollama.delete_model(&model_name),
            "llama.cpp" => {
                if let Some(ref path) = file_path {
                    let p = std::path::Path::new(path);
                    if p.exists() {
                        std::fs::remove_file(p).map_err(|e| format!("Failed to delete file: {}", e))
                    } else {
                        Err("File not found on disk".to_string())
                    }
                } else {
                    // Try matching by name in the models dir
                    self.llamacpp.delete_model(&model_name)
                }
            }
            _ => Err(format!("Deletion not supported for {}", provider_name)),
        };

        match result {
            Ok(()) => {
                self.pull_status = Some(format!("Deleted {}", model_name));
                self.download_history.remove(actual_idx);
                self.clamp_dm_cursor();
                self.refresh_installed();
            }
            Err(e) => {
                self.pull_status = Some(format!("Delete failed: {}", e));
            }
        }
    }

    fn clamp_dm_cursor(&mut self) {
        let len = self.download_history.records.len();
        if len == 0 {
            self.dm_history_cursor = 0;
        } else if self.dm_history_cursor >= len {
            self.dm_history_cursor = len - 1;
        }
    }

    pub fn toggle_detail(&mut self) {
        self.show_plan = false;
        self.show_compare = false;
        self.show_downloads = false;
        self.show_benchmarks = false;
        self.show_detail = !self.show_detail;
    }

    pub fn mark_selected_for_compare(&mut self) {
        let Some(model_name) = self.selected_fit().map(|fit| fit.model.name.clone()) else {
            self.pull_status = Some("No selected model to mark".to_string());
            return;
        };
        self.compare_mark_model = Some(model_name.clone());
        self.pull_status = Some(format!("Marked '{}' for compare", model_name));
    }

    pub fn clear_compare_mark(&mut self) {
        self.compare_mark_model = None;
        self.show_compare = false;
        self.pull_status = Some("Cleared compare mark".to_string());
    }

    pub fn copy_selected_model_name(&mut self) {
        let Some(fit) = self.selected_fit() else {
            self.pull_status = Some("No model selected".to_string());
            return;
        };
        let name = fit.model.name.clone();
        match arboard::Clipboard::new() {
            Ok(mut clipboard) => match clipboard.set_text(&name) {
                Ok(()) => self.pull_status = Some(format!("Copied '{}' to clipboard", name)),
                Err(e) => self.pull_status = Some(format!("Clipboard error: {}", e)),
            },
            Err(e) => self.pull_status = Some(format!("Clipboard error: {}", e)),
        }
    }

    pub fn selected_compare_pair(&self) -> Option<(&ModelFit, &ModelFit)> {
        let selected = self.selected_fit()?;
        let mark_name = self.compare_mark_model.as_deref()?;
        let marked = self.all_fits.iter().find(|f| f.model.name == mark_name)?;
        if marked.model.name == selected.model.name {
            return None;
        }
        Some((marked, selected))
    }

    pub fn toggle_compare_view(&mut self) {
        if self.show_compare {
            self.show_compare = false;
            return;
        }
        if self.compare_mark_model.is_none() {
            self.pull_status = Some("No marked model. Press m to mark one first".to_string());
            return;
        }
        if self.selected_compare_pair().is_none() {
            self.pull_status =
                Some("Select a different model than the marked one to compare".to_string());
            return;
        }
        self.show_detail = false;
        self.show_plan = false;
        self.show_downloads = false;
        self.show_benchmarks = false;
        self.show_compare = true;
    }

    pub fn open_plan_mode(&mut self) {
        let Some(&fit_idx) = self.filtered_fits.get(self.selected_row) else {
            return;
        };
        let fit = &self.all_fits[fit_idx];

        self.show_detail = false;
        self.show_compare = false;
        self.show_downloads = false;
        self.show_benchmarks = false;
        self.show_plan = true;
        self.input_mode = InputMode::Plan;
        self.plan_model_idx = Some(fit_idx);
        self.plan_field = PlanField::Context;
        self.plan_context_input = fit.model.context_length.min(8192).to_string();
        self.plan_quant_input = fit.model.quantization.clone();
        self.plan_kv_quant_input.clear();
        self.plan_target_tps_input.clear();
        self.plan_cursor_position = self.plan_context_input.len();
        self.refresh_plan_estimate();
    }

    pub fn close_plan_mode(&mut self) {
        self.show_plan = false;
        self.plan_model_idx = None;
        self.plan_estimate = None;
        self.plan_error = None;
        self.input_mode = InputMode::Normal;
    }

    // ── Benchmarks view ──────────────────────────────────────────────

    pub fn open_benchmarks(&mut self) {
        self.show_detail = false;
        self.show_compare = false;
        self.show_multi_compare = false;
        self.show_plan = false;
        self.show_downloads = false;
        self.show_benchmarks = true;
        self.input_mode = InputMode::Benchmarks;
        self.bench_cursor = 0;
        self.bench_scroll = 0;

        // Fetch if we don't have data yet
        if self.bench_entries.is_empty() && !self.bench_loading {
            self.fetch_benchmarks();
        }
    }

    pub fn close_benchmarks(&mut self) {
        self.show_benchmarks = false;
        self.input_mode = InputMode::Normal;
    }

    pub fn fetch_benchmarks(&mut self) {
        self.bench_loading = true;
        self.bench_error = None;

        let key = self.bench_api_key.as_deref();
        match llmfit_core::benchmarks::fetch_leaderboard(&self.specs, key, 100) {
            Ok(resp) => {
                self.bench_total = resp.total;
                self.bench_entries = resp.rows;
                self.bench_loading = false;
            }
            Err(_) => {
                // API failed — try embedded cache fallback
                self.bench_loading = false;
                if let Some(cached) = self.find_cached_for_specs() {
                    self.bench_total = cached.total;
                    self.bench_entries = cached.rows;
                    self.bench_error = Some("Using cached data (API unreachable)".to_string());
                } else {
                    self.bench_error =
                        Some("API unreachable and no cached data for this hardware".to_string());
                }
            }
        }
    }

    pub fn bench_move_up(&mut self) {
        if self.bench_cursor > 0 {
            self.bench_cursor -= 1;
        }
    }

    pub fn bench_move_down(&mut self) {
        if !self.bench_entries.is_empty() && self.bench_cursor < self.bench_entries.len() - 1 {
            self.bench_cursor += 1;
        }
    }

    pub fn bench_refresh(&mut self) {
        self.bench_entries.clear();
        self.bench_error = None;
        self.fetch_benchmarks();
    }

    /// Try to find cached benchmark data matching the user's detected hardware.
    fn find_cached_for_specs(&self) -> Option<llmfit_core::benchmarks::LeaderboardResponse> {
        let gpu_name = self.specs.gpu_name.as_deref().unwrap_or("");
        let lower = gpu_name.to_lowercase();

        // Try each preset and see if the GPU name matches
        for preset in llmfit_core::benchmarks::HardwarePreset::all() {
            if let Some(hw_name) = preset.hardware_name {
                if lower.contains(&hw_name.to_lowercase()) {
                    if let Some(cached) =
                        llmfit_core::benchmarks::cached_leaderboard_for_preset(preset.label)
                    {
                        if !cached.rows.is_empty() {
                            return Some(cached);
                        }
                    }
                }
            }
        }
        None
    }

    pub fn open_bench_hw_picker(&mut self) {
        self.bench_hw_picker_open = true;
        self.bench_hw_picker_cursor = 0;
    }

    pub fn close_bench_hw_picker(&mut self) {
        self.bench_hw_picker_open = false;
    }

    pub fn bench_hw_picker_up(&mut self) {
        if self.bench_hw_picker_cursor > 0 {
            self.bench_hw_picker_cursor -= 1;
        }
    }

    pub fn bench_hw_picker_down(&mut self) {
        let presets = llmfit_core::benchmarks::HardwarePreset::all();
        // +1 for the "My Hardware (auto)" entry at position 0
        let max = presets.len(); // last valid index = presets.len() (0..=presets.len())
        if self.bench_hw_picker_cursor < max {
            self.bench_hw_picker_cursor += 1;
        }
    }

    pub fn bench_hw_picker_select(&mut self) {
        let presets = llmfit_core::benchmarks::HardwarePreset::all();
        self.bench_hw_picker_open = false;

        if self.bench_hw_picker_cursor == 0 {
            // "My Hardware (auto)" — reset to detected hardware
            self.bench_hw_label = None;
            self.bench_entries.clear();
            self.fetch_benchmarks();
        } else {
            let preset = &presets[self.bench_hw_picker_cursor - 1];
            self.bench_hw_label = Some(preset.label.to_string());
            self.bench_entries.clear();
            self.bench_loading = true;
            self.bench_error = None;

            let key = self.bench_api_key.as_deref();
            match llmfit_core::benchmarks::fetch_leaderboard_for_preset(preset, key, 100) {
                Ok(resp) => {
                    self.bench_total = resp.total;
                    self.bench_entries = resp.rows;
                    self.bench_loading = false;
                }
                Err(_) => {
                    // API failed — try embedded cache
                    self.bench_loading = false;
                    if let Some(cached) =
                        llmfit_core::benchmarks::cached_leaderboard_for_preset(preset.label)
                    {
                        self.bench_total = cached.total;
                        self.bench_entries = cached.rows;
                        self.bench_error = Some("Using cached data (API unreachable)".to_string());
                    } else {
                        self.bench_error = Some(
                            "API unreachable and no cached data for this hardware".to_string(),
                        );
                    }
                }
            }
        }

        self.bench_cursor = 0;
        self.bench_scroll = 0;
    }

    pub fn plan_next_field(&mut self) {
        self.plan_field = self.plan_field.next();
        self.plan_cursor_position = self.active_plan_input().len();
    }

    pub fn plan_prev_field(&mut self) {
        self.plan_field = self.plan_field.prev();
        self.plan_cursor_position = self.active_plan_input().len();
    }

    pub fn plan_cursor_left(&mut self) {
        if self.plan_cursor_position > 0 {
            self.plan_cursor_position -= 1;
        }
    }

    pub fn plan_cursor_right(&mut self) {
        let len = self.active_plan_input().len();
        if self.plan_cursor_position < len {
            self.plan_cursor_position += 1;
        }
    }

    pub fn plan_input(&mut self, c: char) {
        match self.plan_field {
            PlanField::Context => {
                if !c.is_ascii_digit() {
                    return;
                }
            }
            PlanField::Quant => {
                if !(c.is_ascii_alphanumeric() || c == '_' || c == '-') {
                    return;
                }
            }
            PlanField::KvQuant => {
                if !(c.is_ascii_alphanumeric() || c == '_') {
                    return;
                }
            }
            PlanField::TargetTps => {
                if !(c.is_ascii_digit() || c == '.') {
                    return;
                }
                if c == '.' && self.plan_target_tps_input.contains('.') {
                    return;
                }
            }
        }

        let cursor = self.plan_cursor_position;
        let input = self.active_plan_input_mut();
        if cursor <= input.len() {
            input.insert(cursor, c);
            self.plan_cursor_position = cursor + 1;
            self.refresh_plan_estimate();
        }
    }

    pub fn plan_backspace(&mut self) {
        if self.plan_cursor_position == 0 {
            return;
        }
        let cursor = self.plan_cursor_position;
        let input = self.active_plan_input_mut();
        if cursor <= input.len() {
            input.remove(cursor - 1);
            self.plan_cursor_position = cursor - 1;
            self.refresh_plan_estimate();
        }
    }

    pub fn plan_delete(&mut self) {
        let cursor = self.plan_cursor_position;
        let input = self.active_plan_input_mut();
        if cursor < input.len() {
            input.remove(cursor);
            self.refresh_plan_estimate();
        }
    }

    pub fn plan_clear_field(&mut self) {
        self.active_plan_input_mut().clear();
        self.plan_cursor_position = 0;
        self.refresh_plan_estimate();
    }

    pub fn refresh_plan_estimate(&mut self) {
        let Some(model_idx) = self.plan_model_idx else {
            self.plan_estimate = None;
            self.plan_error = Some("No model selected for plan".to_string());
            return;
        };
        let Some(fit) = self.all_fits.get(model_idx) else {
            self.plan_estimate = None;
            self.plan_error = Some("Selected model is no longer available".to_string());
            return;
        };

        let context = match self.plan_context_input.trim().parse::<u32>() {
            Ok(v) if v > 0 => v,
            _ => {
                self.plan_estimate = None;
                self.plan_error = Some("Context must be a positive integer".to_string());
                return;
            }
        };

        let quant = if self.plan_quant_input.trim().is_empty() {
            None
        } else {
            Some(self.plan_quant_input.trim().to_string())
        };

        let target_tps = if self.plan_target_tps_input.trim().is_empty() {
            None
        } else {
            match self.plan_target_tps_input.trim().parse::<f64>() {
                Ok(v) if v > 0.0 => Some(v),
                _ => {
                    self.plan_estimate = None;
                    self.plan_error = Some("Target TPS must be a positive number".to_string());
                    return;
                }
            }
        };

        let kv_quant = if self.plan_kv_quant_input.trim().is_empty() {
            None
        } else {
            match llmfit_core::models::KvQuant::parse(self.plan_kv_quant_input.trim()) {
                Some(k) => Some(k),
                None => {
                    self.plan_estimate = None;
                    self.plan_error =
                        Some("KV quant must be one of fp16, fp8, q8_0, q4_0, tq".to_string());
                    return;
                }
            }
        };

        let request = PlanRequest {
            context,
            quant,
            target_tps,
            kv_quant,
        };

        match estimate_model_plan(&fit.model, &request, &self.specs) {
            Ok(plan) => {
                self.plan_estimate = Some(plan);
                self.plan_error = None;
            }
            Err(e) => {
                self.plan_estimate = None;
                self.plan_error = Some(e);
            }
        }
    }

    pub fn plan_model_name(&self) -> Option<&str> {
        self.plan_model_idx
            .and_then(|idx| self.all_fits.get(idx))
            .map(|fit| fit.model.name.as_str())
    }

    pub fn open_provider_popup(&mut self) {
        self.input_mode = InputMode::ProviderPopup;
        // Don't reset cursor -- keep it where it was last time
    }

    pub fn close_provider_popup(&mut self) {
        self.input_mode = InputMode::Normal;
    }

    pub fn open_use_case_popup(&mut self) {
        self.input_mode = InputMode::UseCasePopup;
        // Don't reset cursor -- keep it where it was last time
    }

    pub fn close_use_case_popup(&mut self) {
        self.input_mode = InputMode::Normal;
    }

    pub fn provider_popup_up(&mut self) {
        if self.provider_cursor > 0 {
            self.provider_cursor -= 1;
        }
    }

    pub fn provider_popup_down(&mut self) {
        if self.provider_cursor + 1 < self.providers.len() {
            self.provider_cursor += 1;
        }
    }

    pub fn provider_popup_toggle(&mut self) {
        if self.provider_cursor < self.selected_providers.len() {
            self.selected_providers[self.provider_cursor] =
                !self.selected_providers[self.provider_cursor];
            self.apply_filters();
        }
    }

    pub fn provider_popup_select_all(&mut self) {
        let all_selected = self.selected_providers.iter().all(|&s| s);
        let new_val = !all_selected;
        for s in &mut self.selected_providers {
            *s = new_val;
        }
        self.apply_filters();
    }

    pub fn provider_popup_clear_all(&mut self) {
        for s in &mut self.selected_providers {
            *s = false;
        }
        self.apply_filters();
    }

    pub fn use_case_popup_up(&mut self) {
        if self.use_case_cursor > 0 {
            self.use_case_cursor -= 1;
        }
    }

    pub fn use_case_popup_down(&mut self) {
        if self.use_case_cursor + 1 < self.use_cases.len() {
            self.use_case_cursor += 1;
        }
    }

    pub fn use_case_popup_toggle(&mut self) {
        if self.use_case_cursor < self.selected_use_cases.len() {
            self.selected_use_cases[self.use_case_cursor] =
                !self.selected_use_cases[self.use_case_cursor];
            self.apply_filters();
        }
    }

    pub fn use_case_popup_select_all(&mut self) {
        let all_selected = self.selected_use_cases.iter().all(|&s| s);
        let new_val = !all_selected;
        for s in &mut self.selected_use_cases {
            *s = new_val;
        }
        self.apply_filters();
    }

    pub fn open_capability_popup(&mut self) {
        self.input_mode = InputMode::CapabilityPopup;
    }

    pub fn close_capability_popup(&mut self) {
        self.input_mode = InputMode::Normal;
    }

    pub fn capability_popup_up(&mut self) {
        if self.capability_cursor > 0 {
            self.capability_cursor -= 1;
        }
    }

    pub fn capability_popup_down(&mut self) {
        if self.capability_cursor + 1 < self.capabilities.len() {
            self.capability_cursor += 1;
        }
    }

    pub fn capability_popup_toggle(&mut self) {
        if self.capability_cursor < self.selected_capabilities.len() {
            self.selected_capabilities[self.capability_cursor] =
                !self.selected_capabilities[self.capability_cursor];
            self.apply_filters();
        }
    }

    pub fn capability_popup_select_all(&mut self) {
        let all_selected = self.selected_capabilities.iter().all(|&s| s);
        let new_val = !all_selected;
        for s in &mut self.selected_capabilities {
            *s = new_val;
        }
        self.apply_filters();
    }

    // ── Visual mode ──────────────────────────────────────────────

    pub fn enter_visual_mode(&mut self) {
        self.visual_anchor = Some(self.selected_row);
        self.input_mode = InputMode::Visual;
    }

    pub fn exit_visual_mode(&mut self) {
        self.visual_anchor = None;
        self.input_mode = InputMode::Normal;
    }

    pub fn visual_range(&self) -> Option<std::ops::RangeInclusive<usize>> {
        let anchor = self.visual_anchor?;
        let lo = anchor.min(self.selected_row);
        let hi = anchor.max(self.selected_row);
        Some(lo..=hi)
    }

    pub fn visual_selection_count(&self) -> usize {
        self.visual_range()
            .map(|r| r.end() - r.start() + 1)
            .unwrap_or(0)
    }

    /// In visual mode, compare all selected models.
    pub fn visual_compare(&mut self) {
        let Some(range) = self.visual_range() else {
            return;
        };
        let lo = *range.start();
        let hi = *range.end();
        if lo == hi {
            self.pull_status = Some("Select at least 2 models to compare".to_string());
            return;
        }
        // Collect all filtered_fits indices in the visual range
        self.compare_models = (lo..=hi)
            .filter_map(|row| self.filtered_fits.get(row).copied())
            .collect();
        self.compare_scroll = 0;
        self.exit_visual_mode();
        self.show_detail = false;
        self.show_plan = false;
        self.show_compare = false;
        self.show_downloads = false;
        self.show_benchmarks = false;
        self.show_multi_compare = true;
    }

    pub fn close_multi_compare(&mut self) {
        self.show_multi_compare = false;
        self.compare_models.clear();
    }

    pub fn multi_compare_scroll_left(&mut self) {
        if self.compare_scroll > 0 {
            self.compare_scroll -= 1;
        }
    }

    pub fn multi_compare_scroll_right(&mut self) {
        if !self.compare_models.is_empty()
            && self.compare_scroll < self.compare_models.len().saturating_sub(1)
        {
            self.compare_scroll += 1;
        }
    }

    // ── Select mode ─────────────────────────────────────────────

    pub fn enter_select_mode(&mut self) {
        self.input_mode = InputMode::Select;
    }

    pub fn exit_select_mode(&mut self) {
        self.input_mode = InputMode::Normal;
    }

    pub fn select_column_left(&mut self) {
        if self.select_column > 1 {
            self.select_column -= 1;
        }
    }

    pub fn select_column_right(&mut self) {
        if self.select_column < 14 {
            self.select_column += 1;
        }
    }

    /// Activate the filter for the currently focused column in Select mode.
    pub fn activate_select_column_filter(&mut self) {
        match self.select_column {
            1 => self.cycle_availability_filter(), // Inst
            2 => {
                self.input_mode = InputMode::Search;
            } // Model → search
            3 => {
                self.input_mode = InputMode::ProviderPopup;
            } // Provider
            4 => {
                self.input_mode = InputMode::ParamsBucketPopup;
            } // Params
            5 => self.set_or_toggle_sort(SortColumn::Score), // Score
            6 => self.set_or_toggle_sort(SortColumn::Tps), // tok/s
            7 => {
                self.input_mode = InputMode::QuantPopup;
            } // Quant
            8 => {}                                // Disk (no filter/sort)
            9 => {
                self.input_mode = InputMode::RunModePopup;
            } // Mode
            10 => self.set_or_toggle_sort(SortColumn::MemPct), // Mem%
            11 => self.set_or_toggle_sort(SortColumn::Ctx), // Ctx
            12 => self.set_or_toggle_sort(SortColumn::ReleaseDate), // Date
            13 => self.cycle_fit_filter(),         // Fit
            14 => {
                self.input_mode = InputMode::UseCasePopup;
            } // Use Case
            _ => {}
        }
    }

    /// Set sort column, or toggle ascending/descending if already on that column.
    fn set_or_toggle_sort(&mut self, col: SortColumn) {
        if self.sort_column == col {
            self.sort_ascending = !self.sort_ascending;
        } else {
            self.sort_column = col;
            self.sort_ascending = false;
        }
        self.re_sort();
    }

    // ── Quant popup ─────────────────────────────────────────────

    pub fn close_quant_popup(&mut self) {
        self.input_mode = InputMode::Normal;
    }

    pub fn quant_popup_up(&mut self) {
        if self.quant_cursor > 0 {
            self.quant_cursor -= 1;
        }
    }

    pub fn quant_popup_down(&mut self) {
        if self.quant_cursor + 1 < self.quants.len() {
            self.quant_cursor += 1;
        }
    }

    pub fn quant_popup_toggle(&mut self) {
        if self.quant_cursor < self.selected_quants.len() {
            self.selected_quants[self.quant_cursor] = !self.selected_quants[self.quant_cursor];
            self.apply_filters();
        }
    }

    pub fn quant_popup_select_all(&mut self) {
        let all_selected = self.selected_quants.iter().all(|&s| s);
        let new_val = !all_selected;
        for s in &mut self.selected_quants {
            *s = new_val;
        }
        self.apply_filters();
    }

    // ── RunMode popup ───────────────────────────────────────────

    pub fn close_run_mode_popup(&mut self) {
        self.input_mode = InputMode::Normal;
    }

    pub fn run_mode_popup_up(&mut self) {
        if self.run_mode_cursor > 0 {
            self.run_mode_cursor -= 1;
        }
    }

    pub fn run_mode_popup_down(&mut self) {
        if self.run_mode_cursor + 1 < self.run_modes.len() {
            self.run_mode_cursor += 1;
        }
    }

    pub fn run_mode_popup_toggle(&mut self) {
        if self.run_mode_cursor < self.selected_run_modes.len() {
            self.selected_run_modes[self.run_mode_cursor] =
                !self.selected_run_modes[self.run_mode_cursor];
            self.apply_filters();
        }
    }

    pub fn run_mode_popup_select_all(&mut self) {
        let all_selected = self.selected_run_modes.iter().all(|&s| s);
        let new_val = !all_selected;
        for s in &mut self.selected_run_modes {
            *s = new_val;
        }
        self.apply_filters();
    }

    // ── Params bucket popup ─────────────────────────────────────

    pub fn close_params_bucket_popup(&mut self) {
        self.input_mode = InputMode::Normal;
    }

    pub fn params_bucket_popup_up(&mut self) {
        if self.params_bucket_cursor > 0 {
            self.params_bucket_cursor -= 1;
        }
    }

    pub fn params_bucket_popup_down(&mut self) {
        if self.params_bucket_cursor + 1 < self.params_buckets.len() {
            self.params_bucket_cursor += 1;
        }
    }

    pub fn params_bucket_popup_toggle(&mut self) {
        if self.params_bucket_cursor < self.selected_params_buckets.len() {
            self.selected_params_buckets[self.params_bucket_cursor] =
                !self.selected_params_buckets[self.params_bucket_cursor];
            self.apply_filters();
        }
    }

    pub fn params_bucket_popup_select_all(&mut self) {
        let all_selected = self.selected_params_buckets.iter().all(|&s| s);
        let new_val = !all_selected;
        for s in &mut self.selected_params_buckets {
            *s = new_val;
        }
        self.apply_filters();
    }

    // ── License popup ───────────────────────────────────────────

    pub fn open_license_popup(&mut self) {
        self.input_mode = InputMode::LicensePopup;
    }

    pub fn close_license_popup(&mut self) {
        self.input_mode = InputMode::Normal;
    }

    pub fn license_popup_up(&mut self) {
        if self.license_cursor > 0 {
            self.license_cursor -= 1;
        }
    }

    pub fn license_popup_down(&mut self) {
        if self.license_cursor + 1 < self.licenses.len() {
            self.license_cursor += 1;
        }
    }

    pub fn license_popup_toggle(&mut self) {
        if self.license_cursor < self.selected_licenses.len() {
            self.selected_licenses[self.license_cursor] =
                !self.selected_licenses[self.license_cursor];
            self.apply_filters();
        }
    }

    pub fn license_popup_select_all(&mut self) {
        let all_selected = self.selected_licenses.iter().all(|&s| s);
        let new_val = !all_selected;
        for s in &mut self.selected_licenses {
            *s = new_val;
        }
        self.apply_filters();
    }

    pub fn open_runtime_popup(&mut self) {
        self.input_mode = InputMode::RuntimePopup;
    }

    pub fn close_runtime_popup(&mut self) {
        self.input_mode = InputMode::Normal;
    }

    pub fn runtime_popup_up(&mut self) {
        if self.runtime_cursor > 0 {
            self.runtime_cursor -= 1;
        }
    }

    pub fn runtime_popup_down(&mut self) {
        if self.runtime_cursor + 1 < self.runtimes.len() {
            self.runtime_cursor += 1;
        }
    }

    pub fn runtime_popup_toggle(&mut self) {
        if self.runtime_cursor < self.selected_runtimes.len() {
            self.selected_runtimes[self.runtime_cursor] =
                !self.selected_runtimes[self.runtime_cursor];
            self.apply_filters();
        }
    }

    pub fn runtime_popup_select_all(&mut self) {
        let all_selected = self.selected_runtimes.iter().all(|&s| s);
        let new_val = !all_selected;
        for s in &mut self.selected_runtimes {
            *s = new_val;
        }
        self.apply_filters();
    }

    pub fn open_help_popup(&mut self) {
        self.help_scroll = 0;
        self.input_mode = InputMode::HelpPopup;
    }

    pub fn close_help_popup(&mut self) {
        self.input_mode = InputMode::Normal;
    }

    // ── Hardware simulation ──────────────────────────────────────────

    pub fn open_simulation_popup(&mut self) {
        self.sim_ram_input = format!("{:.1}", self.specs.total_ram_gb);
        self.sim_vram_input = format!("{:.1}", self.specs.gpu_vram_gb.unwrap_or(0.0));
        self.sim_cpu_input = format!("{}", self.specs.total_cpu_cores);
        self.sim_field = SimulationField::Ram;
        self.sim_cursor_position = self.sim_ram_input.len();
        self.input_mode = InputMode::Simulation;
    }

    pub fn close_simulation_popup(&mut self) {
        self.input_mode = InputMode::Normal;
    }

    pub fn apply_simulation(&mut self) {
        let ram: f64 = self
            .sim_ram_input
            .parse()
            .unwrap_or(self.real_specs.total_ram_gb);
        let vram: f64 = self
            .sim_vram_input
            .parse()
            .unwrap_or(self.real_specs.gpu_vram_gb.unwrap_or(0.0));
        let cores: usize = self
            .sim_cpu_input
            .parse()
            .unwrap_or(self.real_specs.total_cpu_cores);

        // Start from real specs, apply overrides (RAM first, then VRAM so it wins on unified)
        let mut specs = self.real_specs.clone();
        specs = specs.with_ram_override(ram);
        specs = specs.with_gpu_memory_override(vram);
        specs = specs.with_cpu_core_override(cores);

        self.specs = specs;
        self.sim_active = true;
        self.rebuild_fits();
        self.input_mode = InputMode::Normal;
    }

    pub fn reset_simulation(&mut self) {
        self.specs = self.real_specs.clone();
        self.sim_active = false;
        self.rebuild_fits();
    }

    /// Re-evaluate all model fits against current `self.specs`, preserving
    /// installed status and filter selections.
    fn rebuild_fits(&mut self) {
        let db = ModelDatabase::new();

        self.backend_hidden_count = db
            .get_all_models()
            .iter()
            .filter(|m| !backend_compatible(m, &self.specs))
            .count();

        self.all_fits = db
            .get_all_models()
            .iter()
            .filter(|m| backend_compatible(m, &self.specs))
            .map(|m| {
                let mut fit =
                    ModelFit::analyze_with_context_limit(m, &self.specs, self.context_limit);
                fit.installed = providers::is_model_installed(&m.name, &self.ollama_installed)
                    || providers::is_model_installed_mlx(&m.name, &self.mlx_installed)
                    || providers::is_model_installed_llamacpp(&m.name, &self.llamacpp_installed)
                    || providers::is_model_installed_docker_mr(&m.name, &self.docker_mr_installed)
                    || providers::is_model_installed_lmstudio(&m.name, &self.lmstudio_installed);
                fit
            })
            .collect();

        self.all_fits = llmfit_core::fit::rank_models_by_fit(self.all_fits.drain(..).collect());
        self.selected_row = 0;
        self.compare_models.clear();
        self.compare_mark_model = None;
        self.apply_filters();
    }

    fn active_sim_input(&self) -> &str {
        match self.sim_field {
            SimulationField::Ram => &self.sim_ram_input,
            SimulationField::Vram => &self.sim_vram_input,
            SimulationField::CpuCores => &self.sim_cpu_input,
        }
    }

    fn active_sim_input_mut(&mut self) -> &mut String {
        match self.sim_field {
            SimulationField::Ram => &mut self.sim_ram_input,
            SimulationField::Vram => &mut self.sim_vram_input,
            SimulationField::CpuCores => &mut self.sim_cpu_input,
        }
    }

    pub fn sim_next_field(&mut self) {
        self.sim_field = self.sim_field.next();
        self.sim_cursor_position = self.active_sim_input().len();
    }

    pub fn sim_prev_field(&mut self) {
        self.sim_field = self.sim_field.prev();
        self.sim_cursor_position = self.active_sim_input().len();
    }

    pub fn sim_input(&mut self, c: char) {
        // Only allow digits and '.' for RAM/VRAM, only digits for CPU
        let allow = match self.sim_field {
            SimulationField::Ram | SimulationField::Vram => c.is_ascii_digit() || c == '.',
            SimulationField::CpuCores => c.is_ascii_digit(),
        };
        if !allow {
            return;
        }
        let pos = self.sim_cursor_position;
        self.active_sim_input_mut().insert(pos, c);
        self.sim_cursor_position += 1;
    }

    pub fn sim_backspace(&mut self) {
        if self.sim_cursor_position > 0 {
            self.sim_cursor_position -= 1;
            let pos = self.sim_cursor_position;
            self.active_sim_input_mut().remove(pos);
        }
    }

    pub fn sim_delete(&mut self) {
        let len = self.active_sim_input().len();
        if self.sim_cursor_position < len {
            let pos = self.sim_cursor_position;
            self.active_sim_input_mut().remove(pos);
        }
    }

    pub fn sim_clear_field(&mut self) {
        self.active_sim_input_mut().clear();
        self.sim_cursor_position = 0;
    }

    pub fn sim_cursor_left(&mut self) {
        if self.sim_cursor_position > 0 {
            self.sim_cursor_position -= 1;
        }
    }

    pub fn sim_cursor_right(&mut self) {
        if self.sim_cursor_position < self.active_sim_input().len() {
            self.sim_cursor_position += 1;
        }
    }

    // ── Advanced Config Popup ──────────────────────────────────────────

    pub fn open_advanced_config_popup(&mut self) {
        self.adv_config_efficiency_input = format!("{:.2}", self.calc_config.efficiency);
        self.adv_config_eff_factor_gpu = format!("{:.2}", self.calc_config.run_mode_factors.gpu);
        self.adv_config_eff_factor_cpu_offload =
            format!("{:.2}", self.calc_config.run_mode_factors.cpu_offload);
        self.adv_config_eff_factor_moe =
            format!("{:.2}", self.calc_config.run_mode_factors.moe_offload);
        self.adv_config_eff_factor_tp =
            format!("{:.2}", self.calc_config.run_mode_factors.tensor_parallel);
        self.adv_config_eff_factor_cpu_only =
            format!("{:.2}", self.calc_config.run_mode_factors.cpu_only);
        self.adv_config_context_cap_input = match self.calc_config.context_cap {
            Some(cap) => cap.to_string(),
            None => String::new(),
        };
        self.adv_config_field = AdvConfigField::Efficiency;
        self.adv_config_cursor_position = self.adv_config_efficiency_input.len();
        self.adv_config_dirty = false;
        self.input_mode = InputMode::AdvancedConfig;
    }

    pub fn close_advanced_config_popup(&mut self) {
        self.input_mode = InputMode::Normal;
    }

    // ── Filter Popup ─────────────────────────────────────────────

    pub fn open_filter_popup(&mut self) {
        self.filter_snapshot = Some(FilterSnapshot {
            params_min: self.filter_params_min_input.clone(),
            params_max: self.filter_params_max_input.clone(),
            mem_pct_min: self.filter_mem_pct_min_input.clone(),
            mem_pct_max: self.filter_mem_pct_max_input.clone(),
            sort_ascending: self.sort_ascending,
            fit_filter: self.fit_filter,
        });
        self.filter_field = FilterPopupField::ParamsMin;
        self.filter_cursor_position = self.filter_params_min_input.len();
        self.filter_sort_ascending = self.sort_ascending;
        self.input_mode = InputMode::FilterPopup;
    }

    pub fn close_filter_popup(&mut self) {
        if let Some(snap) = self.filter_snapshot.take() {
            self.filter_params_min_input = snap.params_min;
            self.filter_params_max_input = snap.params_max;
            self.filter_mem_pct_min_input = snap.mem_pct_min;
            self.filter_mem_pct_max_input = snap.mem_pct_max;
            self.sort_ascending = snap.sort_ascending;
            self.fit_filter = snap.fit_filter;
        }
        self.input_mode = InputMode::Normal;
    }

    pub fn filter_next_field(&mut self) {
        self.filter_field = self.filter_field.next();
        self.filter_cursor_position = self.active_filter_input_len();
    }

    pub fn filter_prev_field(&mut self) {
        self.filter_field = self.filter_field.prev();
        self.filter_cursor_position = self.active_filter_input_len();
    }

    pub fn filter_input(&mut self, c: char) {
        match self.filter_field {
            FilterPopupField::ParamsMin => {
                if c == '.' && self.filter_params_min_input.contains('.') {
                    return;
                }
                if !c.is_ascii_digit() && c != '.' {
                    return;
                }
            }
            FilterPopupField::ParamsMax => {
                if c == '.' && self.filter_params_max_input.contains('.') {
                    return;
                }
                if !c.is_ascii_digit() && c != '.' {
                    return;
                }
            }
            FilterPopupField::MemPctMin | FilterPopupField::MemPctMax => {
                if !c.is_ascii_digit() {
                    return;
                }
            }
            _ => return,
        }
        let pos = self.filter_cursor_position;
        self.active_filter_input_mut().insert(pos, c);
        self.filter_cursor_position += 1;
    }

    pub fn filter_backspace(&mut self) {
        if self.filter_cursor_position == 0 {
            return;
        }
        self.filter_cursor_position -= 1;
        let pos = self.filter_cursor_position;
        self.active_filter_input_mut().remove(pos);
    }

    pub fn filter_delete(&mut self) {
        let len = self.active_filter_input_len();
        if self.filter_cursor_position < len {
            let pos = self.filter_cursor_position;
            self.active_filter_input_mut().remove(pos);
        }
    }

    fn active_filter_input_len(&self) -> usize {
        match self.filter_field {
            FilterPopupField::ParamsMin => self.filter_params_min_input.len(),
            FilterPopupField::ParamsMax => self.filter_params_max_input.len(),
            FilterPopupField::MemPctMin => self.filter_mem_pct_min_input.len(),
            FilterPopupField::MemPctMax => self.filter_mem_pct_max_input.len(),
            FilterPopupField::SortDirection | FilterPopupField::FitFilter => 0,
        }
    }

    fn active_filter_input_mut(&mut self) -> &mut String {
        match self.filter_field {
            FilterPopupField::ParamsMin => &mut self.filter_params_min_input,
            FilterPopupField::ParamsMax => &mut self.filter_params_max_input,
            FilterPopupField::MemPctMin => &mut self.filter_mem_pct_min_input,
            FilterPopupField::MemPctMax => &mut self.filter_mem_pct_max_input,
            FilterPopupField::SortDirection | FilterPopupField::FitFilter => {
                unreachable!("no text input for toggle fields")
            }
        }
    }

    pub fn filter_clear_active_input(&mut self) {
        self.active_filter_input_mut().clear();
        self.filter_cursor_position = 0;
    }

    pub fn filter_cursor_left(&mut self) {
        if self.filter_cursor_position > 0 {
            self.filter_cursor_position -= 1;
        }
    }

    pub fn filter_cursor_right(&mut self) {
        let len = self.active_filter_input_len();
        if self.filter_cursor_position < len {
            self.filter_cursor_position += 1;
        }
    }

    pub fn filter_toggle_sort_direction(&mut self) {
        self.filter_sort_ascending = !self.filter_sort_ascending;
    }

    pub fn cycle_filter_fit(&mut self) {
        self.fit_filter = self.fit_filter.next();
    }

    pub fn apply_filter_popup(&mut self) {
        self.filter_snapshot = None;
        self.sort_ascending = self.filter_sort_ascending;
        self.apply_filters();
        self.re_sort();
        self.save_filters();
        self.input_mode = InputMode::Normal;
    }

    fn active_adv_config_input(&self) -> &str {
        match self.adv_config_field {
            AdvConfigField::Efficiency => &self.adv_config_efficiency_input,
            AdvConfigField::FactorGpu => &self.adv_config_eff_factor_gpu,
            AdvConfigField::FactorCpuOffload => &self.adv_config_eff_factor_cpu_offload,
            AdvConfigField::FactorMoe => &self.adv_config_eff_factor_moe,
            AdvConfigField::FactorTp => &self.adv_config_eff_factor_tp,
            AdvConfigField::FactorCpuOnly => &self.adv_config_eff_factor_cpu_only,
            AdvConfigField::ContextCap => &self.adv_config_context_cap_input,
        }
    }

    fn active_adv_config_input_mut(&mut self) -> &mut String {
        match self.adv_config_field {
            AdvConfigField::Efficiency => &mut self.adv_config_efficiency_input,
            AdvConfigField::FactorGpu => &mut self.adv_config_eff_factor_gpu,
            AdvConfigField::FactorCpuOffload => &mut self.adv_config_eff_factor_cpu_offload,
            AdvConfigField::FactorMoe => &mut self.adv_config_eff_factor_moe,
            AdvConfigField::FactorTp => &mut self.adv_config_eff_factor_tp,
            AdvConfigField::FactorCpuOnly => &mut self.adv_config_eff_factor_cpu_only,
            AdvConfigField::ContextCap => &mut self.adv_config_context_cap_input,
        }
    }

    pub fn adv_config_next_field(&mut self) {
        self.adv_config_field = self.adv_config_field.next();
        self.adv_config_cursor_position = self.active_adv_config_input().len();
    }

    pub fn adv_config_prev_field(&mut self) {
        self.adv_config_field = self.adv_config_field.prev();
        self.adv_config_cursor_position = self.active_adv_config_input().len();
    }

    pub fn reset_advanced_config(&mut self) {
        self.calc_config = CalcConfig::default();
        self.rebuild_fits_with_config();
        // Refresh input fields to show defaults
        self.open_advanced_config_popup();
    }

    pub fn adv_config_input(&mut self, c: char) {
        let allow = match self.adv_config_field {
            AdvConfigField::ContextCap => c.is_ascii_digit(),
            _ => {
                if c == '.' && self.active_adv_config_input().contains('.') {
                    false
                } else {
                    c.is_ascii_digit() || c == '.'
                }
            }
        };
        if !allow {
            return;
        }
        let pos = self.adv_config_cursor_position;
        self.active_adv_config_input_mut().insert(pos, c);
        self.adv_config_cursor_position += 1;
        self.adv_config_dirty = true;
    }

    pub fn adv_config_backspace(&mut self) {
        if self.adv_config_cursor_position > 0 {
            self.adv_config_cursor_position -= 1;
            let pos = self.adv_config_cursor_position;
            self.active_adv_config_input_mut().remove(pos);
            self.adv_config_dirty = true;
        }
    }

    pub fn adv_config_delete(&mut self) {
        let len = self.active_adv_config_input().len();
        if self.adv_config_cursor_position < len {
            let pos = self.adv_config_cursor_position;
            self.active_adv_config_input_mut().remove(pos);
            self.adv_config_dirty = true;
        }
    }

    pub fn adv_config_clear_field(&mut self) {
        self.active_adv_config_input_mut().clear();
        self.adv_config_cursor_position = 0;
        self.adv_config_dirty = true;
    }

    pub fn adv_config_cursor_left(&mut self) {
        if self.adv_config_cursor_position > 0 {
            self.adv_config_cursor_position -= 1;
        }
    }

    pub fn adv_config_cursor_right(&mut self) {
        if self.adv_config_cursor_position < self.active_adv_config_input().len() {
            self.adv_config_cursor_position += 1;
        }
    }

    pub fn apply_advanced_config(&mut self) {
        // Parse all fields with fallbacks to current values
        let efficiency: f64 = self
            .adv_config_efficiency_input
            .parse()
            .unwrap_or(self.calc_config.efficiency);
        let gpu: f64 = self
            .adv_config_eff_factor_gpu
            .parse()
            .unwrap_or(self.calc_config.run_mode_factors.gpu);
        let cpu_offload: f64 = self
            .adv_config_eff_factor_cpu_offload
            .parse()
            .unwrap_or(self.calc_config.run_mode_factors.cpu_offload);
        let moe: f64 = self
            .adv_config_eff_factor_moe
            .parse()
            .unwrap_or(self.calc_config.run_mode_factors.moe_offload);
        let tp: f64 = self
            .adv_config_eff_factor_tp
            .parse()
            .unwrap_or(self.calc_config.run_mode_factors.tensor_parallel);
        let cpu_only: f64 = self
            .adv_config_eff_factor_cpu_only
            .parse()
            .unwrap_or(self.calc_config.run_mode_factors.cpu_only);
        let context_cap: Option<u32> = if self.adv_config_context_cap_input.is_empty() {
            None
        } else {
            self.adv_config_context_cap_input.parse().ok()
        };

        // Update the config
        self.calc_config = CalcConfig {
            efficiency,
            run_mode_factors: llmfit_core::fit::RunModeFactors {
                gpu,
                cpu_offload,
                moe_offload: moe,
                tensor_parallel: tp,
                cpu_only,
            },
            context_cap,
            ..self.calc_config
        };

        // Re-run analysis with new config
        self.rebuild_fits_with_config();
        self.input_mode = InputMode::Normal;
    }

    /// Rebuild fits using the custom calc_config
    fn rebuild_fits_with_config(&mut self) {
        let db = ModelDatabase::new();

        self.backend_hidden_count = db
            .get_all_models()
            .iter()
            .filter(|m| !backend_compatible(m, &self.specs))
            .count();

        self.all_fits = db
            .get_all_models()
            .iter()
            .filter(|m| backend_compatible(m, &self.specs))
            .map(|m| {
                let mut fit =
                    ModelFit::analyze_with_config(m, &self.specs, self.calc_config.clone());
                fit.installed = providers::is_model_installed(&m.name, &self.ollama_installed)
                    || providers::is_model_installed_mlx(&m.name, &self.mlx_installed)
                    || providers::is_model_installed_llamacpp(&m.name, &self.llamacpp_installed)
                    || providers::is_model_installed_docker_mr(&m.name, &self.docker_mr_installed)
                    || providers::is_model_installed_lmstudio(&m.name, &self.lmstudio_installed);
                fit
            })
            .collect();

        self.all_fits = llmfit_core::fit::rank_models_by_fit(self.all_fits.drain(..).collect());
        self.selected_row = 0;
        self.compare_models.clear();
        self.compare_mark_model = None;
        self.apply_filters();
    }

    pub fn toggle_installed_first(&mut self) {
        self.installed_first = !self.installed_first;
        self.re_sort();
    }

    /// Re-sort all_fits using current sort column and installed_first preference, then refilter.
    fn re_sort(&mut self) {
        let fits = std::mem::take(&mut self.all_fits);
        let mut sorted = llmfit_core::fit::rank_models_by_fit_opts_col(
            fits,
            self.installed_first,
            self.sort_column,
        );
        if self.sort_ascending {
            sorted.reverse();
        }
        self.all_fits = sorted;
        self.apply_filters();
    }

    /// Start pulling the currently selected model via the best available provider.
    pub fn start_download(&mut self) {
        let any_available = self.ollama_available
            || self.mlx_available
            || self.llamacpp_available
            || self.docker_mr_available
            || self.lmstudio_available
            || self.vllm_available;
        if !any_available {
            self.pull_status = Some(
                "No runtime available — install Ollama, llama.cpp, Docker, LM Studio, or vLLM"
                    .to_string(),
            );
            return;
        }
        if self.pull_active.is_some() {
            return; // already pulling
        }
        let Some(fit) = self.selected_fit() else {
            return;
        };
        if fit.installed {
            self.pull_status = Some("Already installed".to_string());
            return;
        }
        let model_name = fit.model.name.clone();
        let model_format = fit.model.format;
        let is_mlx_model = fit.model.is_mlx_model();
        let has_catalog_gguf = !fit.model.gguf_sources.is_empty();

        let download_options = self.available_download_providers(&model_name, has_catalog_gguf);
        if !download_options.is_empty() {
            self.open_download_provider_popup(model_name, download_options);
        } else {
            let any_runtime = self.ollama_available
                || self.ollama_binary_available
                || self.llamacpp_available
                || self.mlx_available
                || self.docker_mr_available
                || self.lmstudio_available
                || self.vllm_available;
            self.pull_status = Some(if any_runtime {
                Self::format_no_download_message(model_format, is_mlx_model)
            } else {
                "No runtime available — install Ollama, llama.cpp, Docker, LM Studio, or vLLM"
                    .to_string()
            });
        }
    }

    /// Build a user-friendly message explaining why no download is available,
    /// based on the model's weight format.
    fn format_no_download_message(
        format: llmfit_core::models::ModelFormat,
        is_mlx_model: bool,
    ) -> String {
        use llmfit_core::models::ModelFormat;
        if is_mlx_model {
            "MLX model — requires Apple Silicon with MLX installed".to_string()
        } else {
            match format {
                ModelFormat::Awq => {
                    "AWQ model — requires vLLM or a CUDA/ROCm GPU; no GGUF conversion available"
                        .to_string()
                }
                ModelFormat::Gptq => {
                    "GPTQ model — requires vLLM or a CUDA/ROCm GPU; no GGUF conversion available"
                        .to_string()
                }
                _ => "No downloadable format found for this model".to_string(),
            }
        }
    }

    fn start_mlx_download(&mut self, model_name: String) {
        let tag = providers::mlx_pull_tag(&model_name);
        match self.mlx.start_pull(&tag) {
            Ok(handle) => {
                self.pull_model_name = Some(model_name);
                let repo_display = if tag.contains('/') {
                    tag
                } else {
                    format!("mlx-community/{}", tag)
                };
                self.pull_status = Some(format!("Pulling {}...", repo_display));
                self.pull_percent = None;
                self.pull_provider = Some(ActivePullProvider::Mlx);
                self.pull_active = Some(handle);
            }
            Err(e) => {
                self.pull_status = Some(format!("MLX pull failed: {}", e));
            }
        }
    }

    fn start_download_with_provider(&mut self, model_name: String, provider: DownloadProvider) {
        match provider {
            DownloadProvider::Ollama => self.start_ollama_download(model_name),
            DownloadProvider::Mlx => self.start_mlx_download(model_name),
            DownloadProvider::LlamaCpp => self.start_llamacpp_download_for_model(model_name),
            DownloadProvider::DockerModelRunner => self.start_docker_mr_download(model_name),
            DownloadProvider::LmStudio => self.start_lmstudio_download(model_name),
            DownloadProvider::Vllm => self.start_vllm_download(model_name),
        }
    }

    fn start_ollama_download(&mut self, model_name: String) {
        let Some(tag) = providers::ollama_pull_tag(&model_name) else {
            self.pull_status = Some("Not available in Ollama registry".to_string());
            return;
        };
        match self.ollama.start_pull(&tag) {
            Ok(handle) => {
                self.pull_model_name = Some(model_name);
                self.pull_status = Some(format!("Pulling {}...", tag));
                self.pull_percent = Some(0.0);
                self.pull_provider = Some(ActivePullProvider::Ollama);
                self.pull_active = Some(handle);
            }
            Err(e) => {
                self.pull_status = Some(format!("Pull failed: {}", e));
            }
        }
    }

    /// Start downloading a GGUF model via the llama.cpp provider.
    fn start_llamacpp_download_for_model(&mut self, model_name: String) {
        // Check catalog gguf_sources first (instant), then fall back to HTTP probe
        let catalog_repo = self
            .all_fits
            .iter()
            .find(|f| f.model.name == model_name)
            .and_then(|f| f.model.gguf_sources.first())
            .map(|s| s.repo.clone());
        let Some(repo) = catalog_repo.or_else(|| providers::first_existing_gguf_repo(&model_name))
        else {
            self.pull_status = Some("No GGUF repo found in remote registry".to_string());
            return;
        };

        match self.llamacpp.start_pull(&repo) {
            Ok(handle) => {
                self.pull_model_name = Some(model_name);
                self.pull_status = Some(format!("Downloading GGUF from {}...", repo));
                self.pull_percent = Some(0.0);
                self.pull_provider = Some(ActivePullProvider::LlamaCpp);
                self.pull_active = Some(handle);
            }
            Err(e) => {
                self.pull_status = Some(format!("GGUF download failed: {}", e));
            }
        }
    }

    fn start_docker_mr_download(&mut self, model_name: String) {
        let Some(docker_tag) = providers::docker_mr_pull_tag(&model_name) else {
            self.pull_status = Some("Not available for Docker Model Runner".to_string());
            return;
        };
        match self.docker_mr.start_pull(&docker_tag) {
            Ok(handle) => {
                self.pull_model_name = Some(model_name);
                self.pull_status = Some(format!("Pulling {} via Docker...", docker_tag));
                self.pull_percent = None;
                self.pull_provider = Some(ActivePullProvider::DockerModelRunner);
                self.pull_active = Some(handle);
            }
            Err(e) => {
                self.pull_status = Some(format!("Docker pull failed: {}", e));
            }
        }
    }

    fn start_lmstudio_download(&mut self, model_name: String) {
        let Some(tag) = providers::lmstudio_pull_tag(&model_name) else {
            self.pull_status = Some("Not available for LM Studio".to_string());
            return;
        };
        match self.lmstudio.start_pull(&tag) {
            Ok(handle) => {
                self.pull_model_name = Some(model_name);
                self.pull_status = Some(format!("Downloading {} via LM Studio...", tag));
                self.pull_percent = Some(0.0);
                self.pull_provider = Some(ActivePullProvider::LmStudio);
                self.pull_active = Some(handle);
            }
            Err(e) => {
                self.pull_status = Some(format!("LM Studio download failed: {}", e));
            }
        }
    }

    fn start_vllm_download(&mut self, model_name: String) {
        let Some(tag) = providers::vllm_pull_tag(&model_name) else {
            self.pull_status = Some("Not available for vLLM".to_string());
            return;
        };
        match self.vllm.start_pull(&tag) {
            Ok(handle) => {
                self.pull_model_name = Some(model_name);
                self.pull_status = Some(format!("Downloading {} via vLLM...", tag));
                self.pull_percent = Some(0.0);
                self.pull_provider = Some(ActivePullProvider::Vllm);
                self.pull_active = Some(handle);
            }
            Err(e) => {
                self.pull_status = Some(format!("vLLM: {}", e));
            }
        }
    }

    /// Poll the active pull for progress. Called each TUI tick.
    pub fn tick_pull(&mut self) {
        self.tick_provider_detection();
        self.enqueue_capability_probes_for_visible(24);
        self.tick_download_capability();
        self.tick_count = self.tick_count.wrapping_add(1);
        let Some(handle) = &self.pull_active else {
            return;
        };
        // Drain all available events
        loop {
            match handle.receiver.try_recv() {
                Ok(PullEvent::Progress { status, percent }) => {
                    if let Some(p) = percent {
                        self.pull_percent = Some(p);
                    }
                    self.pull_status = Some(status);
                }
                Ok(PullEvent::Done) => {
                    let provider_label = self
                        .pull_provider
                        .map(|p| p.label().to_string())
                        .unwrap_or_else(|| "unknown".to_string());
                    let done_msg = format!("Download complete via {}!", provider_label);
                    self.pull_status = Some(done_msg);

                    // Record in download history
                    self.download_history.add_record(DownloadRecord {
                        model_name: self
                            .pull_model_name
                            .clone()
                            .unwrap_or_else(|| "unknown".to_string()),
                        provider: provider_label,
                        result: DownloadResult::Success,
                        timestamp: DownloadHistory::epoch_now(),
                        file_path: None,
                    });

                    self.pull_percent = None;
                    self.pull_active = None;
                    self.pull_provider = None;
                    self.refresh_installed();
                    return;
                }
                Ok(PullEvent::Error(e)) => {
                    let provider_label = self
                        .pull_provider
                        .map(|p| p.label().to_string())
                        .unwrap_or_else(|| "unknown".to_string());
                    self.pull_status = Some(format!("Error: {}", e));

                    // Record failure in download history
                    self.download_history.add_record(DownloadRecord {
                        model_name: self
                            .pull_model_name
                            .clone()
                            .unwrap_or_else(|| "unknown".to_string()),
                        provider: provider_label,
                        result: DownloadResult::Error(e),
                        timestamp: DownloadHistory::epoch_now(),
                        file_path: None,
                    });

                    self.pull_percent = None;
                    self.pull_active = None;
                    self.pull_provider = None;
                    return;
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.pull_status = Some("Pull ended".to_string());
                    self.pull_percent = None;
                    self.pull_active = None;
                    self.pull_provider = None;
                    self.refresh_installed();
                    return;
                }
            }
        }
    }

    fn available_download_providers(
        &self,
        model_name: &str,
        has_catalog_gguf: bool,
    ) -> Vec<DownloadProvider> {
        let mut providers_for_model = Vec::new();
        if providers::has_ollama_mapping(model_name)
            && (self.ollama_available || self.ollama_binary_available)
        {
            providers_for_model.push(DownloadProvider::Ollama);
        }
        if self.mlx_available {
            providers_for_model.push(DownloadProvider::Mlx);
        }
        // Check catalog gguf_sources first (no HTTP probe needed), then
        // fall back to the heuristic repo lookup. Cache the result so
        // both llama.cpp and LM Studio can use it without a double probe.
        let has_gguf =
            has_catalog_gguf || providers::first_existing_gguf_repo(model_name).is_some();
        if self.llamacpp_available && has_gguf {
            providers_for_model.push(DownloadProvider::LlamaCpp);
        }
        if self.docker_mr_available && providers::has_docker_mr_mapping(model_name) {
            providers_for_model.push(DownloadProvider::DockerModelRunner);
        }
        if self.lmstudio_available && has_gguf {
            providers_for_model.push(DownloadProvider::LmStudio);
        }
        if self.vllm_available && providers::has_vllm_mapping(model_name) {
            providers_for_model.push(DownloadProvider::Vllm);
        }
        providers_for_model
    }

    fn open_download_provider_popup(&mut self, model_name: String, options: Vec<DownloadProvider>) {
        self.download_provider_model = Some(model_name);
        self.download_provider_options = options;
        self.download_provider_cursor = 0;
        self.input_mode = InputMode::DownloadProviderPopup;
        self.pull_status = Some("Choose download runtime and press Enter".to_string());
    }

    pub fn close_download_provider_popup(&mut self) {
        self.download_provider_model = None;
        self.download_provider_options.clear();
        self.download_provider_cursor = 0;
        self.input_mode = InputMode::Normal;
        self.pull_status = Some("Download cancelled".to_string());
    }

    pub fn download_provider_popup_up(&mut self) {
        if self.download_provider_cursor > 0 {
            self.download_provider_cursor -= 1;
        }
    }

    pub fn download_provider_popup_down(&mut self) {
        if self.download_provider_cursor + 1 < self.download_provider_options.len() {
            self.download_provider_cursor += 1;
        }
    }

    pub fn confirm_download_provider_selection(&mut self) {
        let Some(model_name) = self.download_provider_model.clone() else {
            self.input_mode = InputMode::Normal;
            return;
        };
        let Some(provider) = self
            .download_provider_options
            .get(self.download_provider_cursor)
            .copied()
        else {
            self.close_download_provider_popup();
            return;
        };

        self.download_provider_model = None;
        self.download_provider_options.clear();
        self.download_provider_cursor = 0;
        self.input_mode = InputMode::Normal;
        self.start_download_with_provider(model_name, provider);
    }

    /// Re-query all providers for installed models and update all_fits.
    pub fn refresh_installed(&mut self) {
        let (ollama_set, ollama_count) = self.ollama.installed_models_counted();
        self.ollama_installed = ollama_set;
        self.ollama_installed_count = ollama_count;
        self.mlx_installed = self.mlx.installed_models();
        let (llamacpp_set, llamacpp_count) = self.llamacpp.installed_models_counted();
        self.llamacpp_installed = llamacpp_set;
        self.llamacpp_installed_count = llamacpp_count;
        let (docker_mr_set, docker_mr_count) = self.docker_mr.installed_models_counted();
        self.docker_mr_installed = docker_mr_set;
        self.docker_mr_installed_count = docker_mr_count;
        let (lmstudio_set, lmstudio_count) = self.lmstudio.installed_models_counted();
        self.lmstudio_installed = lmstudio_set;
        self.lmstudio_installed_count = lmstudio_count;
        let (vllm_set, vllm_count) = self.vllm.installed_models_counted();
        self.vllm_installed = vllm_set;
        self.vllm_installed_count = vllm_count;
        for fit in &mut self.all_fits {
            fit.installed = providers::is_model_installed(&fit.model.name, &self.ollama_installed)
                || providers::is_model_installed_mlx(&fit.model.name, &self.mlx_installed)
                || providers::is_model_installed_llamacpp(
                    &fit.model.name,
                    &self.llamacpp_installed,
                )
                || providers::is_model_installed_docker_mr(
                    &fit.model.name,
                    &self.docker_mr_installed,
                )
                || providers::is_model_installed_lmstudio(
                    &fit.model.name,
                    &self.lmstudio_installed,
                )
                || providers::is_model_installed_vllm(&fit.model.name, &self.vllm_installed);
        }
        self.re_sort();
        self.enqueue_capability_probes_for_visible(24);
    }

    pub fn download_capability_for(&self, model_name: &str) -> DownloadCapability {
        self.download_capabilities
            .get(model_name)
            .copied()
            .unwrap_or(DownloadCapability::Unknown)
    }

    pub fn enqueue_capability_probes_for_visible(&mut self, window: usize) {
        if self.filtered_fits.is_empty() {
            return;
        }
        let start = self.selected_row.saturating_sub(window / 2);
        let end = (start + window).min(self.filtered_fits.len());
        for idx in start..end {
            if let Some(&fit_idx) = self.filtered_fits.get(idx) {
                let model_name = self.all_fits[fit_idx].model.name.clone();
                let has_catalog_gguf = !self.all_fits[fit_idx].model.gguf_sources.is_empty();
                self.enqueue_capability_probe(model_name, has_catalog_gguf);
            }
        }
    }

    fn enqueue_capability_probe(&mut self, model_name: String, has_catalog_gguf: bool) {
        if self.download_capabilities.contains_key(&model_name)
            || self.download_capability_inflight.contains(&model_name)
            || self.download_capability_inflight.len() >= 12
        {
            return;
        }
        self.download_capability_inflight.insert(model_name.clone());

        let tx = self.download_capability_tx.clone();
        let ollama_runtime_available = self.ollama_available || self.ollama_binary_available;
        let llamacpp_available = self.llamacpp_available;
        let docker_mr_available = self.docker_mr_available;
        let lmstudio_available = self.lmstudio_available;
        let vllm_available = self.vllm_available;
        std::thread::spawn(move || {
            let has_ollama = ollama_runtime_available && providers::has_ollama_mapping(&model_name);
            let has_llamacpp = if llamacpp_available {
                // Use catalog data when available to skip slow HTTP probes
                has_catalog_gguf || providers::first_existing_gguf_repo(&model_name).is_some()
            } else {
                false
            };
            let has_docker = docker_mr_available && providers::has_docker_mr_mapping(&model_name);
            let has_lmstudio = lmstudio_available && providers::has_lmstudio_mapping(&model_name);
            let has_vllm = vllm_available && providers::has_vllm_mapping(&model_name);

            let mut flags = 0u8;
            if has_ollama {
                flags |= DL_OLLAMA;
            }
            if has_llamacpp {
                flags |= DL_LLAMACPP;
            }
            if has_docker {
                flags |= DL_DOCKER;
            }
            if has_lmstudio {
                flags |= DL_LMSTUDIO;
            }
            if has_vllm {
                flags |= DL_VLLM;
            }
            let _ = tx.send((model_name, DownloadCapability::Known(flags)));
        });
    }

    fn tick_download_capability(&mut self) {
        loop {
            match self.download_capability_rx.try_recv() {
                Ok((name, capability)) => {
                    self.download_capability_inflight.remove(&name);
                    self.download_capabilities.insert(name, capability);
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => break,
            }
        }
    }

    /// Poll background provider detection threads and merge results.
    fn tick_provider_detection(&mut self) {
        let mut got_any = false;
        loop {
            match self.provider_detection_rx.try_recv() {
                Ok(msg) => {
                    got_any = true;
                    match msg {
                        ProviderDetectionMsg::Ollama {
                            available,
                            binary_available,
                            installed,
                            installed_count,
                            provider,
                        } => {
                            self.ollama_available = available;
                            self.ollama_binary_available = binary_available;
                            self.ollama_installed = installed;
                            self.ollama_installed_count = installed_count;
                            self.ollama = provider;
                        }
                        ProviderDetectionMsg::Mlx {
                            available,
                            installed,
                        } => {
                            self.mlx_available = available;
                            self.mlx_installed = installed;
                        }
                        ProviderDetectionMsg::DockerMr {
                            available,
                            installed,
                            installed_count,
                        } => {
                            self.docker_mr_available = available;
                            self.docker_mr_installed = installed;
                            self.docker_mr_installed_count = installed_count;
                        }
                        ProviderDetectionMsg::LmStudio {
                            available,
                            installed,
                            installed_count,
                        } => {
                            self.lmstudio_available = available;
                            self.lmstudio_installed = installed;
                            self.lmstudio_installed_count = installed_count;
                        }
                        ProviderDetectionMsg::Vllm {
                            available,
                            installed,
                            installed_count,
                        } => {
                            self.vllm_available = available;
                            self.vllm_installed = installed;
                            self.vllm_installed_count = installed_count;
                        }
                    }
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.providers_loading = false;
                    break;
                }
            }
        }
        if got_any {
            // Re-mark installed status for all models
            for fit in &mut self.all_fits {
                fit.installed =
                    providers::is_model_installed(&fit.model.name, &self.ollama_installed)
                        || providers::is_model_installed_mlx(&fit.model.name, &self.mlx_installed)
                        || providers::is_model_installed_llamacpp(
                            &fit.model.name,
                            &self.llamacpp_installed,
                        )
                        || providers::is_model_installed_docker_mr(
                            &fit.model.name,
                            &self.docker_mr_installed,
                        )
                        || providers::is_model_installed_lmstudio(
                            &fit.model.name,
                            &self.lmstudio_installed,
                        )
                        || providers::is_model_installed_vllm(
                            &fit.model.name,
                            &self.vllm_installed,
                        );
            }
            self.re_sort();
        }
    }

    fn active_plan_input(&self) -> &String {
        match self.plan_field {
            PlanField::Context => &self.plan_context_input,
            PlanField::Quant => &self.plan_quant_input,
            PlanField::KvQuant => &self.plan_kv_quant_input,
            PlanField::TargetTps => &self.plan_target_tps_input,
        }
    }

    fn active_plan_input_mut(&mut self) -> &mut String {
        match self.plan_field {
            PlanField::Context => &mut self.plan_context_input,
            PlanField::Quant => &mut self.plan_quant_input,
            PlanField::KvQuant => &mut self.plan_kv_quant_input,
            PlanField::TargetTps => &mut self.plan_target_tps_input,
        }
    }

    // ── Live inference-bench view ─────────────────────────────────────────

    /// Open the live bench view. Loads cache if available, otherwise starts
    /// a fresh benchmark run against all installed Ollama models.
    pub fn open_bench(&mut self) {
        self.show_bench = true;
        self.live_bench_scroll = 0;
        self.bench_view_mode = BenchViewMode::Results;

        if !self.bench_running && self.ollama_available {
            if !self.load_bench_cache() {
                self.start_bench();
            }
        }
    }

    /// Force re-run benchmarks (Shift-B keybinding).
    pub fn rerun_bench(&mut self) {
        if !self.bench_running && self.ollama_available {
            self.start_bench();
        }
    }

    pub fn close_bench(&mut self) {
        self.show_bench = false;
    }

    fn bench_cache_path() -> Option<std::path::PathBuf> {
        dirs::home_dir().map(|h| h.join(".config").join("llmfit").join("bench-cache.json"))
    }

    fn save_bench_cache(&self) {
        let Some(path) = Self::bench_cache_path() else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let mut installed: Vec<String> = self
            .ollama_installed
            .iter()
            .map(|m| m.strip_suffix(":latest").unwrap_or(m).to_string())
            .collect();
        installed.sort();
        let cache = serde_json::json!({
            "models_hash": format!("{:?}", installed),
            "results": self.bench_results,
            "routing": self.bench_routing,
            "runner_ups": self.bench_runner_ups,
        });
        let _ = std::fs::write(
            &path,
            serde_json::to_string_pretty(&cache).unwrap_or_default(),
        );
    }

    /// Load cached results. Returns `true` if cache was valid and loaded.
    fn load_bench_cache(&mut self) -> bool {
        let Some(path) = Self::bench_cache_path() else {
            return false;
        };
        let Ok(data) = std::fs::read_to_string(&path) else {
            return false;
        };
        let Ok(cache) = serde_json::from_str::<serde_json::Value>(&data) else {
            return false;
        };

        // Check if installed models match
        let mut installed: Vec<String> = self
            .ollama_installed
            .iter()
            .map(|m| m.strip_suffix(":latest").unwrap_or(m).to_string())
            .collect();
        installed.sort();
        let current_hash = format!("{:?}", installed);
        let cached_hash = cache
            .get("models_hash")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if current_hash != cached_hash {
            self.bench_progress = "Models changed — re-running benchmarks...".to_string();
            return false;
        }

        if let Some(results) = cache.get("results") {
            if let Ok(r) =
                serde_json::from_value::<Vec<quality::ModelQualityResult>>(results.clone())
            {
                self.bench_results = r;
            }
        }
        if let Some(routing) = cache.get("routing") {
            if let Ok(r) =
                serde_json::from_value::<Vec<quality::RoutingRecommendation>>(routing.clone())
            {
                self.bench_routing = r;
            }
        }
        if let Some(ru) = cache.get("runner_ups") {
            if let Ok(r) = serde_json::from_value::<Vec<quality::RoutingRecommendation>>(ru.clone())
            {
                self.bench_runner_ups = r;
            }
        }

        // Rebuild model status from cached results
        let config = quality::default_quality_config();
        self.bench_model_status = installed
            .iter()
            .map(|name| {
                let is_done = self.bench_results.iter().any(|r| r.model == *name);
                BenchModelStatus {
                    name: name.clone(),
                    state: if is_done {
                        BenchModelState::Complete
                    } else {
                        BenchModelState::Pending
                    },
                    roles_done: if is_done { config.roles.len() } else { 0 },
                    roles_total: config.roles.len(),
                    current_role: if is_done {
                        "done".to_string()
                    } else {
                        String::new()
                    },
                    last_quality: 0.0,
                    last_speed: 0.0,
                }
            })
            .collect();

        self.bench_progress = "Loaded from cache (Shift-B to re-run)".to_string();
        self.bench_running = false;
        true
    }

    pub fn toggle_bench_view(&mut self) {
        self.bench_view_mode = match self.bench_view_mode {
            BenchViewMode::Results => BenchViewMode::Routing,
            BenchViewMode::Routing => BenchViewMode::Results,
        };
        self.live_bench_scroll = 0;
    }

    fn start_bench(&mut self) {
        self.bench_running = true;
        self.bench_confirm_quit = false;
        self.bench_progress = "Starting benchmarks...".to_string();
        self.bench_results.clear();
        self.bench_routing.clear();
        self.bench_runner_ups.clear();
        self.bench_tests_done = 0;

        let (tx, rx) = mpsc::channel();
        self.bench_rx = Some(rx);

        let ollama_url =
            std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string());
        // Deduplicate: strip ":latest" suffix and remove dupes
        let mut seen = std::collections::HashSet::new();
        let mut models: Vec<String> = self
            .ollama_installed
            .iter()
            .map(|m| m.strip_suffix(":latest").unwrap_or(m).to_string())
            .filter(|m| seen.insert(m.clone()))
            .collect();
        models.sort();
        let config = quality::default_quality_config();
        let roles_total = config.roles.len();

        // Compute total tests across all models
        let tests_per_model: usize = config.roles.values().map(|r| r.tests.len()).sum();
        self.bench_tests_total = tests_per_model * models.len();

        // Build initial status list
        self.bench_model_status = models
            .iter()
            .map(|name| BenchModelStatus {
                name: name.clone(),
                state: BenchModelState::Pending,
                roles_done: 0,
                roles_total,
                current_role: String::new(),
                last_quality: 0.0,
                last_speed: 0.0,
            })
            .collect();

        std::thread::spawn(move || {
            for model in &models {
                let _ = tx.send(BenchMsg::ModelStarted(model.clone()));

                let mut all_roles = Vec::new();
                let mut all_test_results = Vec::new();

                for (role_name, role_def) in &config.roles {
                    let url = format!("{}/api/generate", ollama_url.trim_end_matches('/'));
                    let mut role_results = Vec::new();

                    for test_def in &role_def.tests {
                        let max_tok = test_def.max_tokens.unwrap_or(1024);
                        let temp = test_def.temperature.unwrap_or(0.3);
                        let result = match quality::quality_ollama_generate(
                            &url,
                            model,
                            &test_def.prompt,
                            max_tok,
                            temp,
                        ) {
                            Ok(resp) => {
                                let q = quality::evaluate_response(&resp.text, &test_def.rules);
                                let sw = test_def.speed_weight.unwrap_or(1.0);
                                let sn = (resp.tok_per_sec / 3.0).min(10.0);
                                let comp = (q * 2.0 + sn * sw) / (2.0 + sw);
                                quality::QualityResult {
                                    test_name: test_def.name.clone(),
                                    role: role_name.clone(),
                                    quality: q,
                                    tok_per_sec: resp.tok_per_sec,
                                    eval_tokens: resp.eval_count,
                                    wall_time_sec: resp.wall_time_sec,
                                    ttft_ms: resp.ttft_ms,
                                    composite: comp,
                                    response_preview: resp.text.chars().take(150).collect(),
                                    error: None,
                                }
                            }
                            Err(e) => quality::QualityResult {
                                test_name: test_def.name.clone(),
                                role: role_name.clone(),
                                quality: 0.0,
                                tok_per_sec: 0.0,
                                eval_tokens: 0,
                                wall_time_sec: 0.0,
                                ttft_ms: None,
                                composite: 0.0,
                                response_preview: String::new(),
                                error: Some(e),
                            },
                        };
                        let _ = tx.send(BenchMsg::TestDone {
                            model: model.clone(),
                            role: role_name.clone(),
                            test_name: test_def.name.clone(),
                        });
                        role_results.push(result);
                    }

                    let n = role_results.len().max(1) as f64;
                    let avg_q = role_results.iter().map(|r| r.quality).sum::<f64>() / n;
                    let avg_s = role_results.iter().map(|r| r.tok_per_sec).sum::<f64>() / n;
                    let avg_c = role_results.iter().map(|r| r.composite).sum::<f64>() / n;
                    let tests_in_role = role_results.len();

                    let _ = tx.send(BenchMsg::RoleDone {
                        model: model.clone(),
                        role: role_name.clone(),
                        quality: avg_q,
                        speed: avg_s,
                        tests_in_role,
                    });

                    all_test_results.extend(role_results);

                    all_roles.push(quality::RoleScore {
                        role: role_name.clone(),
                        quality: (avg_q * 10.0).round() / 10.0,
                        speed: (avg_s * 10.0).round() / 10.0,
                        composite: (avg_c * 10.0).round() / 10.0,
                        test_count: tests_in_role,
                    });
                }

                let overall_q = all_roles.iter().map(|r| r.quality).sum::<f64>()
                    / all_roles.len().max(1) as f64;
                let overall_s =
                    all_roles.iter().map(|r| r.speed).sum::<f64>() / all_roles.len().max(1) as f64;
                let overall_c = all_roles.iter().map(|r| r.composite).sum::<f64>()
                    / all_roles.len().max(1) as f64;

                let _ = tx.send(BenchMsg::ModelDone(quality::ModelQualityResult {
                    model: model.clone(),
                    provider: "ollama".to_string(),
                    roles: all_roles,
                    test_results: all_test_results,
                    overall_quality: (overall_q * 10.0).round() / 10.0,
                    overall_speed: (overall_s * 10.0).round() / 10.0,
                    overall_composite: (overall_c * 10.0).round() / 10.0,
                }));
            }
            let _ = tx.send(BenchMsg::AllDone);
        });
    }

    pub fn tick_bench(&mut self) {
        if self.show_bench {
            self.tick_count = self.tick_count.wrapping_add(1);
        }
        if let Some(rx) = &self.bench_rx {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    BenchMsg::Progress(s) => self.bench_progress = s,
                    BenchMsg::ModelStarted(name) => {
                        self.bench_progress = format!("Benchmarking {}...", name);
                        if let Some(ms) =
                            self.bench_model_status.iter_mut().find(|m| m.name == name)
                        {
                            ms.state = BenchModelState::Running;
                        }
                    }
                    BenchMsg::TestDone {
                        model: _,
                        role,
                        test_name,
                    } => {
                        self.bench_tests_done += 1;
                        self.bench_progress = format!(
                            "{}/{} tests [{} > {}]",
                            self.bench_tests_done, self.bench_tests_total, role, test_name
                        );
                    }
                    BenchMsg::RoleDone {
                        model,
                        role,
                        quality,
                        speed,
                        tests_in_role: _,
                    } => {
                        if let Some(ms) =
                            self.bench_model_status.iter_mut().find(|m| m.name == model)
                        {
                            ms.roles_done += 1;
                            ms.current_role = role;
                            ms.last_quality = quality;
                            ms.last_speed = speed;
                        }
                    }
                    BenchMsg::ModelDone(result) => {
                        self.bench_progress = format!("Completed: {}", result.model);
                        if let Some(ms) = self
                            .bench_model_status
                            .iter_mut()
                            .find(|m| m.name == result.model)
                        {
                            ms.state = BenchModelState::Complete;
                            ms.current_role = "done".to_string();
                        }
                        self.bench_results.push(result);
                    }
                    BenchMsg::AllDone => {
                        self.bench_running = false;
                        self.bench_progress = "Benchmark complete! (Shift-B to re-run)".to_string();
                        self.bench_routing = quality::compute_routing(&self.bench_results);
                        self.bench_runner_ups = quality::compute_runner_ups(&self.bench_results);
                        self.save_bench_cache();
                    }
                }
            }
        }
    }
}

fn command_exists(name: &str) -> bool {
    std::process::Command::new("which")
        .arg(name)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}
