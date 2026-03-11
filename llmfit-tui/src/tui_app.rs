use llmfit_core::fit::{FitLevel, ModelFit, SortColumn, backend_compatible};
use llmfit_core::hardware::SystemSpecs;
use llmfit_core::models::{Capability, ModelDatabase, UseCase};
use llmfit_core::plan::{PlanEstimate, PlanRequest, estimate_model_plan};
use llmfit_core::providers::{
    self, LlamaCppProvider, MlxProvider, ModelProvider, OllamaProvider, PullEvent, PullHandle,
};

use std::collections::{HashMap, HashSet};
use std::sync::mpsc;

use crate::theme::Theme;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputMode {
    Normal,
    Search,
    Plan,
    ProviderPopup,
    UseCasePopup,
    CapabilityPopup,
    DownloadProviderPopup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanField {
    Context,
    Quant,
    TargetTps,
}

impl PlanField {
    fn next(self) -> Self {
        match self {
            PlanField::Context => PlanField::Quant,
            PlanField::Quant => PlanField::TargetTps,
            PlanField::TargetTps => PlanField::Context,
        }
    }

    fn prev(self) -> Self {
        match self {
            PlanField::Context => PlanField::TargetTps,
            PlanField::Quant => PlanField::Context,
            PlanField::TargetTps => PlanField::Quant,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitFilter {
    All,
    Perfect,
    Good,
    Marginal,
    TooTight,
    Runnable, // Perfect + Good + Marginal (excludes TooTight)
}

impl FitFilter {
    pub fn label(&self) -> &str {
        match self {
            FitFilter::All => "All",
            FitFilter::Perfect => "Perfect",
            FitFilter::Good => "Good",
            FitFilter::Marginal => "Marginal",
            FitFilter::TooTight => "Too Tight",
            FitFilter::Runnable => "Runnable",
        }
    }

    pub fn next(&self) -> Self {
        match self {
            FitFilter::All => FitFilter::Runnable,
            FitFilter::Runnable => FitFilter::Perfect,
            FitFilter::Perfect => FitFilter::Good,
            FitFilter::Good => FitFilter::Marginal,
            FitFilter::Marginal => FitFilter::TooTight,
            FitFilter::TooTight => FitFilter::All,
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

    pub fn next(&self) -> Self {
        match self {
            AvailabilityFilter::All => AvailabilityFilter::HasGguf,
            AvailabilityFilter::HasGguf => AvailabilityFilter::Installed,
            AvailabilityFilter::Installed => AvailabilityFilter::All,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownloadProvider {
    Ollama,
    LlamaCpp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownloadCapability {
    Unknown,
    None,
    Ollama,
    LlamaCpp,
    Both,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActivePullProvider {
    Ollama,
    Mlx,
    LlamaCpp,
}

impl ActivePullProvider {
    fn label(self) -> &'static str {
        match self {
            ActivePullProvider::Ollama => "Ollama",
            ActivePullProvider::Mlx => "MLX",
            ActivePullProvider::LlamaCpp => "llama.cpp",
        }
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
    pub installed_first: bool,
    pub sort_column: SortColumn,

    // Table state
    pub selected_row: usize,

    // Detail view
    pub show_detail: bool,
    pub show_compare: bool,
    pub compare_mark_model: Option<String>,
    pub show_plan: bool,
    plan_model_idx: Option<usize>,
    pub plan_field: PlanField,
    pub plan_context_input: String,
    pub plan_quant_input: String,
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
    llamacpp: LlamaCppProvider,

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

    // Theme
    pub theme: Theme,

    /// How many models we silently dropped because they can't run on this
    /// hardware — shown in the system bar so users aren't left wondering
    /// why the list looks shorter than expected.
    pub backend_hidden_count: usize,
}

impl App {
    pub fn with_specs_and_context(specs: SystemSpecs, context_limit: Option<u32>) -> Self {
        let db = ModelDatabase::new();

        // Detect Ollama
        let ollama = OllamaProvider::new();
        let (ollama_available, ollama_installed, ollama_installed_count) =
            ollama.detect_with_installed();
        let ollama_binary_available = command_exists("ollama");

        // Detect MLX
        let mlx = MlxProvider::new();
        let (mlx_available, mlx_installed) = mlx.detect_with_installed();

        // Detect llama.cpp
        let llamacpp = LlamaCppProvider::new();
        let llamacpp_available = llamacpp.is_available();
        let (llamacpp_installed, llamacpp_installed_count) = llamacpp.installed_models_counted();

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
                    || providers::is_model_installed_llamacpp(&m.name, &llamacpp_installed);
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

        let selected_providers = vec![true; model_providers.len()];
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
        let selected_use_cases = vec![true; model_use_cases.len()];

        let model_capabilities = Capability::all().to_vec();
        let selected_capabilities = vec![true; model_capabilities.len()];

        let filtered_count = all_fits.len();

        let (download_capability_tx, download_capability_rx) = mpsc::channel();

        let mut app = App {
            should_quit: false,
            input_mode: InputMode::Normal,
            search_query: String::new(),
            cursor_position: 0,
            specs,
            all_fits,
            filtered_fits: (0..filtered_count).collect(),
            providers: model_providers,
            selected_providers,
            use_cases: model_use_cases,
            selected_use_cases,
            capabilities: model_capabilities,
            selected_capabilities,
            fit_filter: FitFilter::All,
            availability_filter: AvailabilityFilter::All,
            installed_first: false,
            sort_column: SortColumn::Score,
            selected_row: 0,
            show_detail: false,
            show_compare: false,
            compare_mark_model: None,
            show_plan: false,
            plan_model_idx: None,
            plan_field: PlanField::Context,
            plan_context_input: String::new(),
            plan_quant_input: String::new(),
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
            llamacpp,
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
            theme: Theme::load(),
            backend_hidden_count,
        };

        app.apply_filters();
        app.enqueue_capability_probes_for_visible(24);
        app
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
                    let searchable = format!(
                        "{} {} {} {} {} {}",
                        fit.model.name.to_lowercase(),
                        fit.model.provider.to_lowercase(),
                        fit.model.parameter_count.to_lowercase(),
                        fit.model.use_case.to_lowercase(),
                        fit.use_case.label().to_lowercase(),
                        caps_text
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

                matches_search
                    && matches_provider
                    && matches_use_case
                    && matches_fit
                    && matches_availability
                    && matches_capability
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

    pub fn home(&mut self) {
        self.selected_row = 0;
        self.enqueue_capability_probes_for_visible(24);
    }

    pub fn end(&mut self) {
        if !self.filtered_fits.is_empty() {
            self.selected_row = self.filtered_fits.len() - 1;
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

    pub fn cycle_sort_column(&mut self) {
        self.sort_column = self.sort_column.next();
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

    pub fn toggle_detail(&mut self) {
        self.show_plan = false;
        self.show_compare = false;
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
        self.show_compare = true;
    }

    pub fn open_plan_mode(&mut self) {
        let Some(&fit_idx) = self.filtered_fits.get(self.selected_row) else {
            return;
        };
        let fit = &self.all_fits[fit_idx];

        self.show_detail = false;
        self.show_compare = false;
        self.show_plan = true;
        self.input_mode = InputMode::Plan;
        self.plan_model_idx = Some(fit_idx);
        self.plan_field = PlanField::Context;
        self.plan_context_input = fit.model.context_length.min(8192).to_string();
        self.plan_quant_input = fit.model.quantization.clone();
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

        let request = PlanRequest {
            context,
            quant,
            target_tps,
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

    pub fn toggle_installed_first(&mut self) {
        self.installed_first = !self.installed_first;
        self.re_sort();
    }

    /// Re-sort all_fits using current sort column and installed_first preference, then refilter.
    fn re_sort(&mut self) {
        let fits = std::mem::take(&mut self.all_fits);
        self.all_fits = llmfit_core::fit::rank_models_by_fit_opts_col(
            fits,
            self.installed_first,
            self.sort_column,
        );
        self.apply_filters();
    }

    /// Start pulling the currently selected model via the best available provider.
    pub fn start_download(&mut self) {
        let any_available = self.ollama_available || self.mlx_available || self.llamacpp_available;
        if !any_available {
            self.pull_status = Some("No provider available (Ollama/MLX/llama.cpp)".to_string());
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

        // Choose provider based on runtime
        let use_mlx = fit.runtime == llmfit_core::fit::InferenceRuntime::Mlx && self.mlx_available;

        if use_mlx {
            self.start_mlx_download(model_name);
            return;
        }

        let download_options = self.available_download_providers(&model_name);
        if !download_options.is_empty() {
            self.open_download_provider_popup(model_name, download_options);
        } else {
            self.pull_status =
                Some("No compatible runtime available — install Ollama or llama.cpp".to_string());
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
            DownloadProvider::LlamaCpp => self.start_llamacpp_download_for_model(model_name),
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
        let Some(repo) = providers::first_existing_gguf_repo(&model_name) else {
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

    /// Poll the active pull for progress. Called each TUI tick.
    pub fn tick_pull(&mut self) {
        self.enqueue_capability_probes_for_visible(24);
        self.tick_download_capability();
        if self.pull_active.is_some() {
            self.tick_count = self.tick_count.wrapping_add(1);
        }
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
                    let done_msg = if let Some(provider) = self.pull_provider {
                        format!("Download complete via {}!", provider.label())
                    } else {
                        "Download complete!".to_string()
                    };
                    self.pull_status = Some(done_msg);
                    self.pull_percent = None;
                    self.pull_active = None;
                    self.pull_provider = None;
                    // Refresh installed models
                    self.refresh_installed();
                    return;
                }
                Ok(PullEvent::Error(e)) => {
                    self.pull_status = Some(format!("Error: {}", e));
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

    fn available_download_providers(&self, model_name: &str) -> Vec<DownloadProvider> {
        let mut providers_for_model = Vec::new();
        if providers::has_ollama_mapping(model_name)
            && (self.ollama_available || self.ollama_binary_available)
        {
            providers_for_model.push(DownloadProvider::Ollama);
        }
        if self.llamacpp_available && providers::first_existing_gguf_repo(model_name).is_some() {
            providers_for_model.push(DownloadProvider::LlamaCpp);
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
        for fit in &mut self.all_fits {
            fit.installed = providers::is_model_installed(&fit.model.name, &self.ollama_installed)
                || providers::is_model_installed_mlx(&fit.model.name, &self.mlx_installed)
                || providers::is_model_installed_llamacpp(
                    &fit.model.name,
                    &self.llamacpp_installed,
                );
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
                self.enqueue_capability_probe(model_name);
            }
        }
    }

    fn enqueue_capability_probe(&mut self, model_name: String) {
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
        std::thread::spawn(move || {
            let has_ollama = ollama_runtime_available && providers::has_ollama_mapping(&model_name);
            let mut has_llamacpp = false;
            if llamacpp_available {
                has_llamacpp = providers::first_existing_gguf_repo(&model_name).is_some();
            }

            let capability = match (has_ollama, has_llamacpp) {
                (true, true) => DownloadCapability::Both,
                (true, false) => DownloadCapability::Ollama,
                (false, true) => DownloadCapability::LlamaCpp,
                (false, false) => DownloadCapability::None,
            };
            let _ = tx.send((model_name, capability));
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

    fn active_plan_input(&self) -> &String {
        match self.plan_field {
            PlanField::Context => &self.plan_context_input,
            PlanField::Quant => &self.plan_quant_input,
            PlanField::TargetTps => &self.plan_target_tps_input,
        }
    }

    fn active_plan_input_mut(&mut self) -> &mut String {
        match self.plan_field {
            PlanField::Context => &mut self.plan_context_input,
            PlanField::Quant => &mut self.plan_quant_input,
            PlanField::TargetTps => &mut self.plan_target_tps_input,
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
