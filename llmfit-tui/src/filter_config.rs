use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

/// Persisted filter state, saved to `~/.config/llmfit/filters.json`.
///
/// Every field is optional so the file degrades gracefully when new filters are
/// added or the model database changes between runs.  Multi-select filters are
/// stored as `name -> selected` maps so additions/removals in the model list
/// don't corrupt saved state.
#[derive(Debug, Serialize, Deserialize, Default)]
pub struct FilterConfig {
    pub fit_filter: Option<String>,
    pub availability_filter: Option<String>,
    pub tp_filter: Option<String>,
    pub sort_column: Option<String>,
    pub sort_ascending: Option<bool>,
    pub installed_first: Option<bool>,
    pub search_query: Option<String>,

    // Multi-select popup filters: name → selected
    pub providers: Option<HashMap<String, bool>>,
    pub use_cases: Option<HashMap<String, bool>>,
    pub capabilities: Option<HashMap<String, bool>>,
    pub quants: Option<HashMap<String, bool>>,
    pub run_modes: Option<HashMap<String, bool>>,
    pub params_buckets: Option<HashMap<String, bool>>,
    pub licenses: Option<HashMap<String, bool>>,
    pub runtimes: Option<HashMap<String, bool>>,

    // Range filters from the Filter Popup
    pub filter_params_min: Option<String>,
    pub filter_params_max: Option<String>,
    pub filter_mem_pct_min: Option<String>,
    pub filter_mem_pct_max: Option<String>,

    /// Custom download directory for GGUF models.
    pub download_dir: Option<String>,
}

impl FilterConfig {
    /// Path to the config file: `~/.config/llmfit/filters.json`
    fn config_path() -> Option<PathBuf> {
        Some(dirs::config_dir()?.join("llmfit").join("filters.json"))
    }

    /// Load the saved filter config from disk, falling back to defaults.
    pub fn load() -> Self {
        Self::config_path()
            .and_then(|path| fs::read_to_string(path).ok())
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    /// Save the current filter config to disk.
    pub fn save(&self) {
        if let Some(path) = Self::config_path() {
            if let Some(parent) = path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            if let Ok(json) = serde_json::to_string_pretty(self) {
                let _ = fs::write(&path, json);
            }
        }
    }

    /// Apply a saved name→selected map onto a positional `Vec<bool>`,
    /// matching by the corresponding names vector.  Entries not present
    /// in the saved map keep their current (default) value.
    pub fn apply_map(names: &[String], selected: &mut [bool], saved: &HashMap<String, bool>) {
        for (i, name) in names.iter().enumerate() {
            if let Some(&val) = saved.get(name) {
                selected[i] = val;
            }
        }
    }

    /// Build a name→selected map from parallel name and selected slices.
    pub fn build_map(names: &[String], selected: &[bool]) -> HashMap<String, bool> {
        names
            .iter()
            .zip(selected.iter())
            .map(|(name, &sel)| (name.clone(), sel))
            .collect()
    }
}
