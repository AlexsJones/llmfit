use llmfit_core::fit::{FitLevel, ModelFit, RunMode, SortColumn};
use llmfit_core::hardware::SystemSpecs;
use llmfit_core::models::{ModelDatabase, UseCase};
use llmfit_core::providers::{
    self, MlxProvider, ModelProvider, OllamaProvider, PullEvent, PullHandle,
};

use std::collections::HashSet;
use std::sync::mpsc;

use crate::theme::Theme;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputMode {
    Normal,
    Search,
    ProviderPopup,
    FilterPopup,
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

// ── Column Filters ───────────────────────────────────────────────────

const SCORE_STEPS: &[f64] = &[0.0, 30.0, 50.0, 60.0, 70.0, 80.0, 90.0];
const TPS_STEPS: &[f64] = &[0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0];
const PARAMS_STEPS: &[f64] = &[0.0, 1.0, 3.0, 7.0, 13.0, 30.0, 70.0];
const MEM_PCT_STEPS: &[f64] = &[200.0, 100.0, 90.0, 80.0, 70.0, 50.0];
const CTX_STEPS: &[f64] = &[0.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0];

#[derive(Debug, Clone, Copy)]
pub struct NumericFilter {
    steps: &'static [f64],
    step_index: usize,
    max_direction: bool,
    unit: &'static str,
}

impl NumericFilter {
    fn new(steps: &'static [f64], max_direction: bool, unit: &'static str) -> Self {
        Self {
            steps,
            step_index: 0,
            max_direction,
            unit,
        }
    }

    pub fn is_active(&self) -> bool {
        self.step_index > 0
    }

    pub fn label(&self) -> String {
        if self.step_index == 0 {
            "Any".to_string()
        } else {
            let v = self.steps[self.step_index];
            let dir = if self.max_direction { "≤" } else { "≥" };
            if v == v.floor() {
                format!("{} {}{}", dir, v as i64, self.unit)
            } else {
                format!("{} {}{}", dir, v, self.unit)
            }
        }
    }

    pub fn cycle_right(&mut self) {
        if self.step_index + 1 < self.steps.len() {
            self.step_index += 1;
        }
    }

    pub fn cycle_left(&mut self) {
        self.step_index = self.step_index.saturating_sub(1);
    }

    pub fn reset(&mut self) {
        self.step_index = 0;
    }

    pub fn matches(&self, value: f64) -> bool {
        if self.step_index == 0 {
            return true;
        }
        let t = self.steps[self.step_index];
        if self.max_direction {
            value <= t
        } else {
            value >= t
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DateFilter {
    Any,
    Last6Months,
    LastYear,
    Last2Years,
}

fn current_year_month() -> (i32, i32) {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let mut days = secs / 86400;
    let mut year = 1970i32;
    loop {
        let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
        let ydays: u64 = if leap { 366 } else { 365 };
        if days < ydays {
            break;
        }
        days -= ydays;
        year += 1;
    }
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let mdays: [u64; 12] = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut month = 1i32;
    for &md in &mdays {
        if days < md {
            break;
        }
        days -= md;
        month += 1;
    }
    (year, month)
}

impl DateFilter {
    pub fn is_active(&self) -> bool {
        *self != DateFilter::Any
    }

    pub fn label(&self) -> &str {
        match self {
            DateFilter::Any => "Any",
            DateFilter::Last6Months => "≤ 6mo",
            DateFilter::LastYear => "≤ 1yr",
            DateFilter::Last2Years => "≤ 2yr",
        }
    }

    pub fn cycle_right(&mut self) {
        *self = match self {
            DateFilter::Any => DateFilter::Last2Years,
            DateFilter::Last2Years => DateFilter::LastYear,
            DateFilter::LastYear => DateFilter::Last6Months,
            DateFilter::Last6Months => DateFilter::Last6Months,
        };
    }

    pub fn cycle_left(&mut self) {
        *self = match self {
            DateFilter::Last6Months => DateFilter::LastYear,
            DateFilter::LastYear => DateFilter::Last2Years,
            DateFilter::Last2Years => DateFilter::Any,
            DateFilter::Any => DateFilter::Any,
        };
    }

    pub fn reset(&mut self) {
        *self = DateFilter::Any;
    }

    pub fn matches(&self, release_date: &Option<String>) -> bool {
        if *self == DateFilter::Any {
            return true;
        }
        let Some(date) = release_date.as_deref().and_then(|d| d.get(..7)) else {
            return false;
        };
        let cutoff = self.cutoff_yyyy_mm();
        date >= cutoff.as_str()
    }

    fn cutoff_yyyy_mm(&self) -> String {
        let months_back: i32 = match self {
            DateFilter::Any => return "0000-00".to_string(),
            DateFilter::Last6Months => 6,
            DateFilter::LastYear => 12,
            DateFilter::Last2Years => 24,
        };
        let (year, month) = current_year_month();
        let total = year * 12 + (month - 1) - months_back;
        let y = total / 12;
        let m = total % 12 + 1;
        format!("{:04}-{:02}", y, m)
    }
}

const ALL_RUN_MODES: [RunMode; 4] = [
    RunMode::Gpu,
    RunMode::MoeOffload,
    RunMode::CpuOffload,
    RunMode::CpuOnly,
];

const RUN_MODE_LABELS: [&str; 4] = ["GPU", "MoE Off", "CPU Off", "CPU"];

#[derive(Debug, Clone, Copy)]
pub struct ModeFilter {
    selected: Option<usize>,
}

impl ModeFilter {
    fn new() -> Self {
        Self { selected: None }
    }
    pub fn is_active(&self) -> bool {
        self.selected.is_some()
    }
    pub fn label(&self) -> &str {
        match self.selected {
            None => "All",
            Some(i) => RUN_MODE_LABELS[i],
        }
    }
    pub fn cycle_right(&mut self) {
        self.selected = match self.selected {
            None => Some(0),
            Some(i) if i + 1 < ALL_RUN_MODES.len() => Some(i + 1),
            _ => None,
        };
    }
    pub fn cycle_left(&mut self) {
        self.selected = match self.selected {
            None => Some(ALL_RUN_MODES.len() - 1),
            Some(0) => None,
            Some(i) => Some(i - 1),
        };
    }
    pub fn reset(&mut self) {
        self.selected = None;
    }
    pub fn matches(&self, mode: RunMode) -> bool {
        match self.selected {
            None => true,
            Some(i) => mode == ALL_RUN_MODES[i],
        }
    }
}

const ALL_USE_CASES: [UseCase; 6] = [
    UseCase::General,
    UseCase::Coding,
    UseCase::Reasoning,
    UseCase::Chat,
    UseCase::Multimodal,
    UseCase::Embedding,
];

#[derive(Debug, Clone, Copy)]
pub struct UseCaseFilter {
    selected: Option<usize>,
}

impl UseCaseFilter {
    fn new() -> Self {
        Self { selected: None }
    }
    pub fn is_active(&self) -> bool {
        self.selected.is_some()
    }
    pub fn label(&self) -> &str {
        match self.selected {
            None => "All",
            Some(i) => ALL_USE_CASES[i].label(),
        }
    }
    pub fn cycle_right(&mut self) {
        self.selected = match self.selected {
            None => Some(0),
            Some(i) if i + 1 < ALL_USE_CASES.len() => Some(i + 1),
            _ => None,
        };
    }
    pub fn cycle_left(&mut self) {
        self.selected = match self.selected {
            None => Some(ALL_USE_CASES.len() - 1),
            Some(0) => None,
            Some(i) => Some(i - 1),
        };
    }
    pub fn reset(&mut self) {
        self.selected = None;
    }
    pub fn matches(&self, uc: UseCase) -> bool {
        match self.selected {
            None => true,
            Some(i) => uc == ALL_USE_CASES[i],
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantFilter {
    values: Vec<String>,
    selected: Option<usize>,
}

impl QuantFilter {
    fn new(values: Vec<String>) -> Self {
        Self {
            values,
            selected: None,
        }
    }
    pub fn is_active(&self) -> bool {
        self.selected.is_some()
    }
    pub fn label(&self) -> &str {
        match self.selected {
            None => "All",
            Some(i) => &self.values[i],
        }
    }
    pub fn cycle_right(&mut self) {
        self.selected = match self.selected {
            None if !self.values.is_empty() => Some(0),
            Some(i) if i + 1 < self.values.len() => Some(i + 1),
            Some(_) => None,
            None => None,
        };
    }
    pub fn cycle_left(&mut self) {
        self.selected = match self.selected {
            None if !self.values.is_empty() => Some(self.values.len() - 1),
            Some(0) => None,
            Some(i) => Some(i - 1),
            None => None,
        };
    }
    pub fn reset(&mut self) {
        self.selected = None;
    }
    pub fn matches(&self, quant: &str) -> bool {
        match self.selected {
            None => true,
            Some(i) => quant == self.values[i],
        }
    }
}

pub const FILTER_ROW_COUNT: usize = 9;
pub const FILTER_ROW_LABELS: [&str; FILTER_ROW_COUNT] = [
    "Score", "tok/s", "Params", "Mem %", "Ctx", "Date", "Mode", "Quant", "Use Case",
];

pub struct ColumnFilters {
    pub score: NumericFilter,
    pub tps: NumericFilter,
    pub params: NumericFilter,
    pub mem_pct: NumericFilter,
    pub ctx: NumericFilter,
    pub date: DateFilter,
    pub mode: ModeFilter,
    pub quant: QuantFilter,
    pub use_case: UseCaseFilter,
}

impl ColumnFilters {
    pub fn new(quant_values: Vec<String>) -> Self {
        Self {
            score: NumericFilter::new(SCORE_STEPS, false, ""),
            tps: NumericFilter::new(TPS_STEPS, false, ""),
            params: NumericFilter::new(PARAMS_STEPS, false, "B"),
            mem_pct: NumericFilter::new(MEM_PCT_STEPS, true, "%"),
            ctx: NumericFilter::new(CTX_STEPS, false, "k"),
            date: DateFilter::Any,
            mode: ModeFilter::new(),
            quant: QuantFilter::new(quant_values),
            use_case: UseCaseFilter::new(),
        }
    }

    pub fn active_count(&self) -> usize {
        [
            self.score.is_active(),
            self.tps.is_active(),
            self.params.is_active(),
            self.mem_pct.is_active(),
            self.ctx.is_active(),
            self.date.is_active(),
            self.mode.is_active(),
            self.quant.is_active(),
            self.use_case.is_active(),
        ]
        .iter()
        .filter(|&&a| a)
        .count()
    }

    pub fn reset_all(&mut self) {
        self.score.reset();
        self.tps.reset();
        self.params.reset();
        self.mem_pct.reset();
        self.ctx.reset();
        self.date.reset();
        self.mode.reset();
        self.quant.reset();
        self.use_case.reset();
    }

    pub fn matches(&self, fit: &ModelFit) -> bool {
        self.score.matches(fit.score)
            && self.tps.matches(fit.estimated_tps)
            && self.params.matches(fit.model.params_b())
            && self.mem_pct.matches(fit.utilization_pct)
            && self.ctx.matches(fit.model.context_length as f64 / 1000.0)
            && self.date.matches(&fit.model.release_date)
            && self.mode.matches(fit.run_mode)
            && self.quant.matches(&fit.best_quant)
            && self.use_case.matches(fit.use_case)
    }

    pub fn row_value_label(&self, row: usize) -> String {
        match row {
            0 => self.score.label(),
            1 => self.tps.label(),
            2 => self.params.label(),
            3 => self.mem_pct.label(),
            4 => self.ctx.label(),
            5 => self.date.label().to_string(),
            6 => self.mode.label().to_string(),
            7 => self.quant.label().to_string(),
            8 => self.use_case.label().to_string(),
            _ => String::new(),
        }
    }

    pub fn row_is_active(&self, row: usize) -> bool {
        match row {
            0 => self.score.is_active(),
            1 => self.tps.is_active(),
            2 => self.params.is_active(),
            3 => self.mem_pct.is_active(),
            4 => self.ctx.is_active(),
            5 => self.date.is_active(),
            6 => self.mode.is_active(),
            7 => self.quant.is_active(),
            8 => self.use_case.is_active(),
            _ => false,
        }
    }

    pub fn adjust_right(&mut self, row: usize) {
        match row {
            0 => self.score.cycle_right(),
            1 => self.tps.cycle_right(),
            2 => self.params.cycle_right(),
            3 => self.mem_pct.cycle_right(),
            4 => self.ctx.cycle_right(),
            5 => self.date.cycle_right(),
            6 => self.mode.cycle_right(),
            7 => self.quant.cycle_right(),
            8 => self.use_case.cycle_right(),
            _ => {}
        }
    }

    pub fn adjust_left(&mut self, row: usize) {
        match row {
            0 => self.score.cycle_left(),
            1 => self.tps.cycle_left(),
            2 => self.params.cycle_left(),
            3 => self.mem_pct.cycle_left(),
            4 => self.ctx.cycle_left(),
            5 => self.date.cycle_left(),
            6 => self.mode.cycle_left(),
            7 => self.quant.cycle_left(),
            8 => self.use_case.cycle_left(),
            _ => {}
        }
    }

    pub fn reset_row(&mut self, row: usize) {
        match row {
            0 => self.score.reset(),
            1 => self.tps.reset(),
            2 => self.params.reset(),
            3 => self.mem_pct.reset(),
            4 => self.ctx.reset(),
            5 => self.date.reset(),
            6 => self.mode.reset(),
            7 => self.quant.reset(),
            8 => self.use_case.reset(),
            _ => {}
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

    // Filters
    pub fit_filter: FitFilter,
    pub installed_first: bool,
    pub sort_column: SortColumn,

    // Table state
    pub selected_row: usize,

    // Detail view
    pub show_detail: bool,

    // Provider popup
    pub provider_cursor: usize,

    // Provider state
    pub ollama_available: bool,
    pub ollama_installed: HashSet<String>,
    ollama: OllamaProvider,
    pub mlx_available: bool,
    pub mlx_installed: HashSet<String>,
    mlx: MlxProvider,

    // Download state
    pub pull_active: Option<PullHandle>,
    pub pull_status: Option<String>,
    pub pull_percent: Option<f64>,
    pub pull_model_name: Option<String>,
    /// Animation frame counter, incremented every tick while pulling.
    pub tick_count: u64,
    /// When true, the next 'd' press will confirm and start the download.
    pub confirm_download: bool,

    // Column filters
    pub column_filters: ColumnFilters,
    pub filter_popup_cursor: usize,

    // Theme
    pub theme: Theme,
}

impl App {
    pub fn with_specs(specs: SystemSpecs) -> Self {
        Self::with_specs_and_context(specs, None)
    }

    pub fn with_specs_and_context(specs: SystemSpecs, context_limit: Option<u32>) -> Self {
        let db = ModelDatabase::new();

        // Detect Ollama
        let ollama = OllamaProvider::new();
        let ollama_available = ollama.is_available();
        let ollama_installed = if ollama_available {
            ollama.installed_models()
        } else {
            HashSet::new()
        };

        // Detect MLX
        let mlx = MlxProvider::new();
        let mlx_available = mlx.is_available();
        let mlx_installed = if mlx_available {
            mlx.installed_models()
        } else {
            // Still scan HF cache even if server/python isn't available
            mlx.installed_models()
        };

        // Analyze all models
        let mut all_fits: Vec<ModelFit> = db
            .get_all_models()
            .iter()
            .map(|m| {
                let mut fit = ModelFit::analyze_with_context_limit(m, &specs, context_limit);
                fit.installed = providers::is_model_installed(&m.name, &ollama_installed)
                    || providers::is_model_installed_mlx(&m.name, &mlx_installed);
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

        // Collect unique quantizations for column filter
        let quant_values: Vec<String> = all_fits
            .iter()
            .map(|f| f.best_quant.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();

        let filtered_count = all_fits.len();

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
            fit_filter: FitFilter::All,
            installed_first: false,
            sort_column: SortColumn::Score,
            selected_row: 0,
            show_detail: false,
            provider_cursor: 0,
            ollama_available,
            ollama_installed,
            ollama,
            mlx_available,
            mlx_installed,
            mlx,
            pull_active: None,
            pull_status: None,
            pull_percent: None,
            pull_model_name: None,
            tick_count: 0,
            confirm_download: false,
            column_filters: ColumnFilters::new(quant_values),
            filter_popup_cursor: 0,
            theme: Theme::load(),
        };

        app.apply_filters();
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
                    // Combine all searchable fields into one string
                    let searchable = format!(
                        "{} {} {} {}",
                        fit.model.name.to_lowercase(),
                        fit.model.provider.to_lowercase(),
                        fit.model.parameter_count.to_lowercase(),
                        fit.model.use_case.to_lowercase()
                    );
                    // All terms must be present (AND logic)
                    terms.iter().all(|term| searchable.contains(term))
                };

                // Provider filter
                let provider_idx = self.providers.iter().position(|p| p == &fit.model.provider);
                let matches_provider = provider_idx
                    .map(|idx| self.selected_providers[idx])
                    .unwrap_or(true);

                // Fit filter
                let matches_fit = match self.fit_filter {
                    FitFilter::All => true,
                    FitFilter::Perfect => fit.fit_level == FitLevel::Perfect,
                    FitFilter::Good => fit.fit_level == FitLevel::Good,
                    FitFilter::Marginal => fit.fit_level == FitLevel::Marginal,
                    FitFilter::TooTight => fit.fit_level == FitLevel::TooTight,
                    FitFilter::Runnable => fit.fit_level != FitLevel::TooTight,
                };

                let matches_columns = self.column_filters.matches(fit);

                matches_search && matches_provider && matches_fit && matches_columns
            })
            .map(|(i, _)| i)
            .collect();

        // Clamp selection
        if self.filtered_fits.is_empty() {
            self.selected_row = 0;
        } else if self.selected_row >= self.filtered_fits.len() {
            self.selected_row = self.filtered_fits.len() - 1;
        }
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
    }

    pub fn move_down(&mut self) {
        self.confirm_download = false;
        if !self.filtered_fits.is_empty() && self.selected_row < self.filtered_fits.len() - 1 {
            self.selected_row += 1;
        }
    }

    pub fn page_up(&mut self) {
        self.confirm_download = false;
        self.selected_row = self.selected_row.saturating_sub(10);
    }

    pub fn page_down(&mut self) {
        self.confirm_download = false;
        if !self.filtered_fits.is_empty() {
            self.selected_row = (self.selected_row + 10).min(self.filtered_fits.len() - 1);
        }
    }

    pub fn half_page_up(&mut self) {
        self.selected_row = self.selected_row.saturating_sub(5);
    }

    pub fn half_page_down(&mut self) {
        if !self.filtered_fits.is_empty() {
            self.selected_row = (self.selected_row + 5).min(self.filtered_fits.len() - 1);
        }
    }

    pub fn home(&mut self) {
        self.selected_row = 0;
    }

    pub fn end(&mut self) {
        if !self.filtered_fits.is_empty() {
            self.selected_row = self.filtered_fits.len() - 1;
        }
    }

    pub fn cycle_fit_filter(&mut self) {
        self.fit_filter = self.fit_filter.next();
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
        self.show_detail = !self.show_detail;
    }

    // ── Filter Popup ──────────────────────────────────────────────

    pub fn open_filter_popup(&mut self) {
        self.input_mode = InputMode::FilterPopup;
    }

    pub fn close_filter_popup(&mut self) {
        self.input_mode = InputMode::Normal;
    }

    pub fn filter_popup_up(&mut self) {
        if self.filter_popup_cursor > 0 {
            self.filter_popup_cursor -= 1;
        }
    }

    pub fn filter_popup_down(&mut self) {
        if self.filter_popup_cursor + 1 < FILTER_ROW_COUNT {
            self.filter_popup_cursor += 1;
        }
    }

    pub fn filter_popup_adjust_right(&mut self) {
        self.column_filters.adjust_right(self.filter_popup_cursor);
        self.apply_filters();
    }

    pub fn filter_popup_adjust_left(&mut self) {
        self.column_filters.adjust_left(self.filter_popup_cursor);
        self.apply_filters();
    }

    pub fn filter_popup_reset_current(&mut self) {
        self.column_filters.reset_row(self.filter_popup_cursor);
        self.apply_filters();
    }

    pub fn filter_popup_reset_all(&mut self) {
        self.column_filters.reset_all();
        self.apply_filters();
    }

    // ── Provider Popup ───────────────────────────────────────────

    pub fn open_provider_popup(&mut self) {
        self.input_mode = InputMode::ProviderPopup;
        // Don't reset cursor -- keep it where it was last time
    }

    pub fn close_provider_popup(&mut self) {
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
        let any_available = self.ollama_available || self.mlx_available;
        if !any_available {
            self.pull_status = Some("No provider available (Ollama/MLX)".to_string());
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

        // Choose provider based on runtime
        let use_mlx = fit.runtime == llmfit_core::fit::InferenceRuntime::Mlx && self.mlx_available;

        if use_mlx {
            let tag = providers::mlx_pull_tag(&fit.model.name);
            let model_name = fit.model.name.clone();
            match self.mlx.start_pull(&tag) {
                Ok(handle) => {
                    self.pull_model_name = Some(model_name);
                    self.pull_status = Some(format!("Pulling mlx-community/{}...", tag));
                    self.pull_percent = None;
                    self.pull_active = Some(handle);
                }
                Err(e) => {
                    self.pull_status = Some(format!("MLX pull failed: {}", e));
                }
            }
        } else if self.ollama_available {
            let Some(tag) = providers::ollama_pull_tag(&fit.model.name) else {
                self.pull_status = Some("Not available in Ollama".to_string());
                return;
            };
            let model_name = fit.model.name.clone();
            match self.ollama.start_pull(&tag) {
                Ok(handle) => {
                    self.pull_model_name = Some(model_name);
                    self.pull_status = Some(format!("Pulling {}...", tag));
                    self.pull_percent = Some(0.0);
                    self.pull_active = Some(handle);
                }
                Err(e) => {
                    self.pull_status = Some(format!("Pull failed: {}", e));
                }
            }
        } else {
            self.pull_status = Some("No provider available".to_string());
        }
    }

    /// Poll the active pull for progress. Called each TUI tick.
    pub fn tick_pull(&mut self) {
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
                    self.pull_status = Some("Download complete!".to_string());
                    self.pull_percent = None;
                    self.pull_active = None;
                    // Refresh installed models
                    self.refresh_installed();
                    return;
                }
                Ok(PullEvent::Error(e)) => {
                    self.pull_status = Some(format!("Error: {}", e));
                    self.pull_percent = None;
                    self.pull_active = None;
                    return;
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.pull_status = Some("Pull ended".to_string());
                    self.pull_percent = None;
                    self.pull_active = None;
                    self.refresh_installed();
                    return;
                }
            }
        }
    }

    /// Re-query all providers for installed models and update all_fits.
    pub fn refresh_installed(&mut self) {
        self.ollama_installed = self.ollama.installed_models();
        self.mlx_installed = self.mlx.installed_models();
        for fit in &mut self.all_fits {
            fit.installed = providers::is_model_installed(&fit.model.name, &self.ollama_installed)
                || providers::is_model_installed_mlx(&fit.model.name, &self.mlx_installed);
        }
        self.re_sort();
    }
}
