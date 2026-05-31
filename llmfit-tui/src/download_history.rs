use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
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
    /// All file paths written by the download when a model is split across shards.
    /// When empty, `file_path` is used as the legacy single-file path.
    #[serde(default)]
    pub download_paths: Vec<String>,
    /// Expected total size of the downloaded files in bytes, when known.
    #[serde(default)]
    pub expected_size_bytes: Option<u64>,
}

impl DownloadRecord {
    pub(crate) fn recorded_paths(&self) -> Vec<String> {
        if !self.download_paths.is_empty() {
            self.download_paths.clone()
        } else {
            self.file_path.iter().cloned().collect()
        }
    }

    pub(crate) fn resolved_paths(&self) -> Vec<PathBuf> {
        self.recorded_paths()
            .into_iter()
            .filter_map(|raw| resolve_recorded_path(&raw))
            .collect()
    }
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

    /// Return the number of validated llama.cpp downloads currently present on disk.
    pub fn valid_llamacpp_downloads(&self) -> (HashSet<String>, usize) {
        let mut installed = HashSet::new();
        for record in &self.records {
            if record.provider != "llama.cpp" {
                continue;
            }
            if !matches!(record.result, DownloadResult::Success) {
                continue;
            }

            let paths = record.resolved_paths();
            if paths.is_empty() {
                continue;
            }

            let mut total_size = 0u64;
            let mut complete = true;
            for path in &paths {
                let Ok(meta) = fs::metadata(path) else {
                    complete = false;
                    break;
                };
                let size = meta.len();
                if size == 0 {
                    complete = false;
                    break;
                }
                total_size += size;
            }
            if !complete {
                continue;
            }

            if let Some(expected) = record.expected_size_bytes
                && total_size != expected
            {
                continue;
            }

            installed.insert(record.model_name.to_lowercase());
        }
        let count = installed.len();
        (installed, count)
    }

    pub fn epoch_now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }
}

fn resolve_recorded_path(raw: &str) -> Option<PathBuf> {
    let path = Path::new(raw);
    if path.is_absolute() {
        Some(path.to_path_buf())
    } else {
        Some(dirs::cache_dir()?.join("llmfit").join("models").join(path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dir(name: &str) -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("llmfit-download-history-{name}-{unique}"));
        fs::create_dir_all(&dir).expect("create test dir");
        dir
    }

    #[test]
    fn valid_llamacpp_downloads_requires_existing_files_and_matching_size() {
        let dir = test_dir("matching-size");
        let file = dir.join("model.gguf");
        fs::write(&file, [1u8, 2, 3, 4]).expect("write test file");

        let record = DownloadRecord {
            model_name: "Foo/Bar".to_string(),
            provider: "llama.cpp".to_string(),
            result: DownloadResult::Success,
            timestamp: 1,
            file_path: Some(file.display().to_string()),
            download_paths: vec![file.display().to_string()],
            expected_size_bytes: Some(4),
        };
        let history = DownloadHistory {
            records: vec![record.clone()],
        };

        let (installed, count) = history.valid_llamacpp_downloads();
        assert_eq!(count, 1);
        assert!(installed.contains("foo/bar"));

        let mut partial_history = history.clone();
        partial_history.records[0].expected_size_bytes = Some(5);
        let (_, partial_count) = partial_history.valid_llamacpp_downloads();
        assert_eq!(partial_count, 0);

        let mut missing_history = history;
        let missing_path = dir.join("missing.gguf");
        missing_history.records[0].file_path = Some(missing_path.display().to_string());
        missing_history.records[0].download_paths = vec![missing_path.display().to_string()];
        let (_, missing_count) = missing_history.valid_llamacpp_downloads();
        assert_eq!(missing_count, 0);

        let _ = fs::remove_dir_all(&dir);
    }
}
