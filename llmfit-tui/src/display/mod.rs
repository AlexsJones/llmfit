pub mod json_mode;
pub mod table_mode;

use llmfit_core::fit::ModelFit;
use llmfit_core::hardware::SystemSpecs;
use llmfit_core::models::LlmModel;
use llmfit_core::plan::PlanEstimate;

/// Common interface for all display backends (table, JSON, …).
pub trait DisplayMode {
    fn display_models(&self, models: &[LlmModel]);
    fn display_model_fits(&self, specs: &SystemSpecs, fits: &[ModelFit]);
    fn display_model_detail(&self, specs: &SystemSpecs, fit: &ModelFit);
    fn display_system(&self, specs: &SystemSpecs);
    fn display_search_results(&self, models: &[&LlmModel], query: &str);
    fn display_plan(&self, specs: &SystemSpecs, plan: &PlanEstimate);
    /// Diff view: compare 2+ models (table or JSON). sort_label used by table backend for header.
    fn display_diff(&self, specs: &SystemSpecs, fits: &[ModelFit], sort_label: &str);
}

/// Construct the right display backend from the `--json` flag.
pub fn new(json: bool) -> Box<dyn DisplayMode> {
    if json {
        Box::new(json_mode::JsonDisplay)
    } else {
        Box::new(table_mode::TableDisplay)
    }
}

// ────────────────────────────────────────────────────────────────────
// Shared helpers used by both backends
// ────────────────────────────────────────────────────────────────────

pub(crate) fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

pub(crate) fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}
