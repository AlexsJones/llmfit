use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// Result of a completed download.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DownloadResult {
    Success,
    Error(String),
}

/// A single download history entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadRecord {
    pub model_name: String,
    pub provider: String,
    pub result: DownloadResult,
    pub timestamp: u64,
    /// File path on disk, for providers that store files directly (e.g. LlamaCpp).
    pub file_path: Option<String>,
}

/// Persistent download history, saved to `~/.config/llmfit/download_history.json`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DownloadHistory {
    pub records: Vec<DownloadRecord>,
}

const MAX_RECORDS: usize = 100;

impl DownloadHistory {
    fn config_path() -> Option<PathBuf> {
        Some(
            dirs::config_dir()?
                .join("llmfit")
                .join("download_history.json"),
        )
    }

    pub fn load() -> Self {
        Self::config_path()
            .and_then(|path| fs::read_to_string(path).ok())
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

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

    pub fn add_record(&mut self, record: DownloadRecord) {
        self.records.push(record);
        // Keep only the most recent entries.
        if self.records.len() > MAX_RECORDS {
            let excess = self.records.len() - MAX_RECORDS;
            self.records.drain(0..excess);
        }
        self.save();
    }

    pub fn remove(&mut self, index: usize) {
        if index < self.records.len() {
            self.records.remove(index);
            self.save();
        }
    }

    pub fn epoch_now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }
}
