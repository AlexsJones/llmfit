//! Runtime model providers (Ollama, llama.cpp, MLX).
//!
//! Each provider can list locally installed models and pull new ones.
//! The trait is designed to be extended for vLLM, etc.

use std::collections::HashSet;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Provider trait
// ---------------------------------------------------------------------------

/// A runtime provider that can serve LLM models locally.
pub trait ModelProvider {
    /// Human-readable name shown in the UI.
    fn name(&self) -> &str;

    /// Whether the provider service is reachable right now.
    fn is_available(&self) -> bool;

    /// Return the set of model name stems that are currently installed.
    /// Names are normalised lowercase, e.g. "llama3.1:8b".
    fn installed_models(&self) -> HashSet<String>;

    /// Start pulling a model. Returns immediately; progress is polled
    /// via `pull_progress()`.
    fn start_pull(&self, model_tag: &str) -> Result<PullHandle, String>;
}

/// Handle returned by `start_pull`. The TUI polls this in a background
/// thread and reads status/progress.
pub struct PullHandle {
    pub model_tag: String,
    pub receiver: std::sync::mpsc::Receiver<PullEvent>,
}

#[derive(Debug, Clone)]
pub enum PullEvent {
    Progress {
        status: String,
        percent: Option<f64>,
    },
    Done,
    Error(String),
}

// ---------------------------------------------------------------------------
// Ollama provider
// ---------------------------------------------------------------------------

pub struct OllamaProvider {
    base_url: String,
}

impl Default for OllamaProvider {
    fn default() -> Self {
        let base_url = std::env::var("OLLAMA_HOST")
            .ok()
            .and_then(|url| {
                if url.starts_with("http://") || url.starts_with("https://") {
                    Some(url)
                } else {
                    eprintln!(
                        "Warning: OLLAMA_HOST must start with http:// or https://, ignoring: {}",
                        url
                    );
                    None
                }
            })
            .unwrap_or_else(|| "http://localhost:11434".to_string());
        Self { base_url }
    }
}

impl OllamaProvider {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build the full API URL for a given endpoint path.
    fn api_url(&self, path: &str) -> String {
        format!("{}/api/{}", self.base_url.trim_end_matches('/'), path)
    }

    /// Single-pass startup probe to avoid duplicate `/api/tags` calls.
    /// Returns `(available, installed_models)`.
    pub fn detect_with_installed(&self) -> (bool, HashSet<String>) {
        let mut set = HashSet::new();
        let Ok(resp) = ureq::get(&self.api_url("tags"))
            .config()
            .timeout_global(Some(std::time::Duration::from_millis(800)))
            .build()
            .call()
        else {
            return (false, set);
        };

        let Ok(tags): Result<TagsResponse, _> = resp.into_body().read_json() else {
            return (true, set);
        };
        for m in tags.models {
            let lower = m.name.to_lowercase();
            set.insert(lower.clone());
            if let Some(family) = lower.split(':').next() {
                set.insert(family.to_string());
            }
        }
        (true, set)
    }

    /// Best-effort check that a tag exists in Ollama's remote registry.
    /// Uses the local Ollama daemon's `/api/show` resolution path.
    pub fn has_remote_tag(&self, model_tag: &str) -> bool {
        let body = serde_json::json!({ "model": model_tag });
        ureq::post(&self.api_url("show"))
            .config()
            .timeout_global(Some(std::time::Duration::from_millis(1200)))
            .build()
            .send_json(&body)
            .is_ok()
    }
}

// -- JSON response types for Ollama API --

#[derive(serde::Deserialize)]
struct TagsResponse {
    models: Vec<OllamaModel>,
}

#[derive(serde::Deserialize)]
struct OllamaModel {
    /// e.g. "llama3.1:8b-instruct-q4_K_M"
    name: String,
}

#[derive(serde::Deserialize)]
struct PullStreamLine {
    #[serde(default)]
    status: String,
    #[serde(default)]
    total: Option<u64>,
    #[serde(default)]
    completed: Option<u64>,
    #[serde(default)]
    error: Option<String>,
}

impl ModelProvider for OllamaProvider {
    fn name(&self) -> &str {
        "Ollama"
    }

    fn is_available(&self) -> bool {
        ureq::get(&self.api_url("tags"))
            .config()
            .timeout_global(Some(std::time::Duration::from_secs(2)))
            .build()
            .call()
            .is_ok()
    }

    fn installed_models(&self) -> HashSet<String> {
        let mut set = HashSet::new();
        let Ok(resp) = ureq::get(&self.api_url("tags"))
            .config()
            .timeout_global(Some(std::time::Duration::from_secs(5)))
            .build()
            .call()
        else {
            return set;
        };
        let Ok(tags): Result<TagsResponse, _> = resp.into_body().read_json() else {
            return set;
        };
        for m in tags.models {
            let lower = m.name.to_lowercase();
            // Store the full tag as-is (lowercased)
            set.insert(lower.clone());
            // Also store just the family (before the colon) so fuzzy matching works
            if let Some(family) = lower.split(':').next() {
                set.insert(family.to_string());
            }
        }
        set
    }

    fn start_pull(&self, model_tag: &str) -> Result<PullHandle, String> {
        let url = self.api_url("pull");
        let tag = model_tag.to_string();
        let (tx, rx) = std::sync::mpsc::channel();

        let body = serde_json::json!({
            "model": tag,
            "stream": true,
        });

        std::thread::spawn(move || {
            let resp = ureq::post(&url)
                .config()
                .timeout_global(Some(std::time::Duration::from_secs(3600)))
                .build()
                .send_json(&body);

            match resp {
                Ok(resp) => {
                    let reader = std::io::BufReader::new(resp.into_body().into_reader());
                    use std::io::BufRead;
                    for line in reader.lines() {
                        let Ok(line) = line else { break };
                        if line.is_empty() {
                            continue;
                        }
                        if let Ok(parsed) = serde_json::from_str::<PullStreamLine>(&line) {
                            // Check for error responses from Ollama
                            if let Some(ref err) = parsed.error {
                                let _ = tx.send(PullEvent::Error(err.clone()));
                                return;
                            }
                            let percent = match (parsed.completed, parsed.total) {
                                (Some(c), Some(t)) if t > 0 => Some(c as f64 / t as f64 * 100.0),
                                _ => None,
                            };
                            let _ = tx.send(PullEvent::Progress {
                                status: parsed.status.clone(),
                                percent,
                            });
                            if parsed.status == "success" {
                                let _ = tx.send(PullEvent::Done);
                                return;
                            }
                        }
                    }
                    // Stream ended without "success" — treat as error
                    let _ = tx.send(PullEvent::Error(
                        "Pull ended without success (model may not exist in Ollama registry)"
                            .to_string(),
                    ));
                }
                Err(e) => {
                    let _ = tx.send(PullEvent::Error(format!("{e}")));
                }
            }
        });

        Ok(PullHandle {
            model_tag: model_tag.to_string(),
            receiver: rx,
        })
    }
}

// ---------------------------------------------------------------------------
// MLX provider (Apple MLX framework via HuggingFace cache)
// ---------------------------------------------------------------------------

pub struct MlxProvider {
    server_url: String,
}

impl Default for MlxProvider {
    fn default() -> Self {
        let server_url = std::env::var("MLX_LM_HOST")
            .ok()
            .and_then(|url| {
                if url.starts_with("http://") || url.starts_with("https://") {
                    Some(url)
                } else {
                    eprintln!(
                        "Warning: MLX_LM_HOST must start with http:// or https://, ignoring: {}",
                        url
                    );
                    None
                }
            })
            .unwrap_or_else(|| "http://localhost:8080".to_string());
        Self { server_url }
    }
}

impl MlxProvider {
    pub fn new() -> Self {
        Self::default()
    }

    /// Single-pass startup probe for MLX.
    /// On non-macOS, skips network checks and reports `available=false`.
    pub fn detect_with_installed(&self) -> (bool, HashSet<String>) {
        let mut set = scan_hf_cache_for_mlx();
        if !cfg!(target_os = "macos") {
            return (false, set);
        }

        let url = format!("{}/v1/models", self.server_url.trim_end_matches('/'));
        if let Ok(resp) = ureq::get(&url)
            .config()
            .timeout_global(Some(std::time::Duration::from_millis(800)))
            .build()
            .call()
        {
            if let Ok(json) = resp.into_body().read_json::<serde_json::Value>()
                && let Some(data) = json.get("data").and_then(|d| d.as_array())
            {
                for model in data {
                    if let Some(id) = model.get("id").and_then(|i| i.as_str()) {
                        set.insert(id.to_lowercase());
                    }
                }
            }
            return (true, set);
        }

        (check_mlx_python(), set)
    }
}

/// Cache whether mlx_lm Python package is importable.
static MLX_PYTHON_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn check_mlx_python() -> bool {
    *MLX_PYTHON_AVAILABLE.get_or_init(|| {
        std::process::Command::new("python3")
            .args(["-c", "import mlx_lm"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    })
}

/// Scan ~/.cache/huggingface/hub/ for MLX model directories.
fn scan_hf_cache_for_mlx() -> HashSet<String> {
    let mut set = HashSet::new();
    let cache_dir = dirs_hf_cache();
    let Ok(entries) = std::fs::read_dir(&cache_dir) else {
        return set;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if let Some(rest) = name_str.strip_prefix("models--mlx-community--") {
            // Directory name: models--mlx-community--Llama-3.1-8B-Instruct-4bit
            // Normalize to lowercase
            let model_name = rest.replace("--", "/").to_lowercase();
            set.insert(model_name);
        }
    }
    set
}

fn dirs_hf_cache() -> std::path::PathBuf {
    if let Ok(cache) = std::env::var("HF_HOME") {
        std::path::PathBuf::from(cache).join("hub")
    } else if let Ok(home) = std::env::var("HOME") {
        std::path::PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub")
    } else {
        std::path::PathBuf::from("/tmp/.cache/huggingface/hub")
    }
}

impl ModelProvider for MlxProvider {
    fn name(&self) -> &str {
        "MLX"
    }

    fn is_available(&self) -> bool {
        if !cfg!(target_os = "macos") {
            return false;
        }
        // Try the MLX server first
        let url = format!("{}/v1/models", self.server_url.trim_end_matches('/'));
        if ureq::get(&url)
            .config()
            .timeout_global(Some(std::time::Duration::from_secs(2)))
            .build()
            .call()
            .is_ok()
        {
            return true;
        }
        // Fall back to checking if mlx_lm is installed
        check_mlx_python()
    }

    fn installed_models(&self) -> HashSet<String> {
        let mut set = scan_hf_cache_for_mlx();
        if !cfg!(target_os = "macos") {
            return set;
        }
        // Also try querying the MLX server if running
        let url = format!("{}/v1/models", self.server_url.trim_end_matches('/'));
        if let Ok(resp) = ureq::get(&url)
            .config()
            .timeout_global(Some(std::time::Duration::from_secs(2)))
            .build()
            .call()
            && let Ok(json) = resp.into_body().read_json::<serde_json::Value>()
            && let Some(data) = json.get("data").and_then(|d| d.as_array())
        {
            for model in data {
                if let Some(id) = model.get("id").and_then(|i| i.as_str()) {
                    set.insert(id.to_lowercase());
                }
            }
        }
        set
    }

    fn start_pull(&self, model_tag: &str) -> Result<PullHandle, String> {
        let tag = model_tag.to_string();
        let (tx, rx) = std::sync::mpsc::channel();

        std::thread::spawn(move || {
            let _ = tx.send(PullEvent::Progress {
                status: format!("Downloading mlx-community/{}...", tag),
                percent: None,
            });

            // Download from Hugging Face using their CLI tool
            let result = std::process::Command::new("hf")
                .args(["download", &format!("mlx-community/{}", tag)])
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .status();

            match result {
                Ok(status) if status.success() => {
                    let _ = tx.send(PullEvent::Done);
                }
                _ => {
                    let _ = tx.send(PullEvent::Error(
                        "hf not found. Install it with: uv tool install 'huggingface_hub[cli]'"
                            .to_string(),
                    ));
                }
            }
        });

        Ok(PullHandle {
            model_tag: model_tag.to_string(),
            receiver: rx,
        })
    }
}

// ---------------------------------------------------------------------------
// llama.cpp provider (direct GGUF download from HuggingFace)
// ---------------------------------------------------------------------------

/// A provider that downloads GGUF model files directly from HuggingFace
/// and uses llama.cpp binaries (`llama-cli`, `llama-server`) to run them.
///
/// Unlike Ollama, this doesn't require a running daemon — it downloads
/// GGUF files to a local cache directory and invokes llama.cpp directly.
pub struct LlamaCppProvider {
    /// Directory where GGUF models are stored.
    models_dir: PathBuf,
    /// Path to llama-cli binary, if found.
    llama_cli: Option<String>,
    /// Path to llama-server binary, if found.
    llama_server: Option<String>,
}

impl Default for LlamaCppProvider {
    fn default() -> Self {
        let models_dir = llamacpp_models_dir();
        let llama_cli = find_binary("llama-cli");
        let llama_server = find_binary("llama-server");
        Self {
            models_dir,
            llama_cli,
            llama_server,
        }
    }
}

impl LlamaCppProvider {
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the directory where GGUF models are cached.
    pub fn models_dir(&self) -> &std::path::Path {
        &self.models_dir
    }

    /// Path to `llama-cli` if detected.
    pub fn llama_cli_path(&self) -> Option<&str> {
        self.llama_cli.as_deref()
    }

    /// Path to `llama-server` if detected.
    pub fn llama_server_path(&self) -> Option<&str> {
        self.llama_server.as_deref()
    }

    /// List all `.gguf` files in the cache directory.
    pub fn list_gguf_files(&self) -> Vec<PathBuf> {
        let mut files = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&self.models_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
                    files.push(path);
                }
            }
        }
        files
    }

    /// Search HuggingFace for GGUF repositories matching a query.
    /// Returns a list of (repo_id, description) tuples.
    pub fn search_hf_gguf(query: &str) -> Vec<(String, String)> {
        let url = format!(
            "https://huggingface.co/api/models?library=gguf&search={}&sort=trending&limit=20",
            urlencoding::encode(query)
        );
        let Ok(resp) = ureq::get(&url)
            .config()
            .timeout_global(Some(std::time::Duration::from_secs(15)))
            .build()
            .call()
        else {
            return Vec::new();
        };
        let Ok(models) = resp.into_body().read_json::<Vec<serde_json::Value>>() else {
            return Vec::new();
        };
        models
            .into_iter()
            .filter_map(|m| {
                let id = m.get("id")?.as_str()?.to_string();
                let desc = m
                    .get("pipeline_tag")
                    .and_then(|v| v.as_str())
                    .unwrap_or("model")
                    .to_string();
                Some((id, desc))
            })
            .collect()
    }

    /// List GGUF files available in a HuggingFace repository.
    /// Returns a list of (filename, size_bytes) tuples.
    pub fn list_repo_gguf_files(repo_id: &str) -> Vec<(String, u64)> {
        let url = format!("https://huggingface.co/api/models/{}/tree/main", repo_id);
        let Ok(resp) = ureq::get(&url)
            .config()
            .timeout_global(Some(std::time::Duration::from_secs(15)))
            .build()
            .call()
        else {
            return Vec::new();
        };
        let Ok(entries) = resp.into_body().read_json::<Vec<serde_json::Value>>() else {
            return Vec::new();
        };
        entries
            .into_iter()
            .filter_map(|e| {
                let path = e.get("path")?.as_str()?.to_string();
                if !path.ends_with(".gguf") {
                    return None;
                }
                let size = e.get("size").and_then(|v| v.as_u64()).unwrap_or(0);
                // Skip split files (e.g., model-00001-of-00003.gguf) but not the
                // primary file. We look for files that look like quantized models.
                Some((path, size))
            })
            .collect()
    }

    /// Select the best GGUF file from a repo that fits within a memory budget.
    /// Prefers higher quality quantizations (Q8 > Q6 > Q5 > Q4 > Q3 > Q2).
    /// `budget_gb` is the available memory in gigabytes.
    pub fn select_best_gguf(files: &[(String, u64)], budget_gb: f64) -> Option<(String, u64)> {
        // Quant preference order (best quality first)
        let quant_order = [
            "Q8_0", "q8_0", "Q6_K", "q6_k", "Q6_K_L", "q6_k_l", "Q5_K_M", "q5_k_m", "Q5_K_S",
            "q5_k_s", "Q4_K_M", "q4_k_m", "Q4_K_S", "q4_k_s", "Q4_0", "q4_0", "Q3_K_M", "q3_k_m",
            "Q3_K_S", "q3_k_s", "Q2_K", "q2_k", "IQ4_XS", "iq4_xs", "IQ3_M", "iq3_m", "IQ2_M",
            "iq2_m", "IQ1_M", "iq1_m",
        ];
        let budget_bytes = (budget_gb * 1024.0 * 1024.0 * 1024.0) as u64;

        // Try each quant level in preference order
        for quant in &quant_order {
            for (filename, size) in files {
                if *size > 0
                    && *size <= budget_bytes
                    && filename.contains(quant)
                    && !is_split_file(filename)
                {
                    return Some((filename.clone(), *size));
                }
            }
        }

        // Fallback: smallest file that fits
        let mut fitting: Vec<_> = files
            .iter()
            .filter(|(f, s)| *s > 0 && *s <= budget_bytes && !is_split_file(f))
            .collect();
        fitting.sort_by_key(|(_, s)| *s);
        fitting.last().map(|(f, s)| (f.clone(), *s))
    }

    /// Download a GGUF file from a HuggingFace repository.
    /// `repo_id` is e.g. "bartowski/Llama-3.1-8B-Instruct-GGUF"
    /// `filename` is e.g. "Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    pub fn download_gguf(&self, repo_id: &str, filename: &str) -> Result<PullHandle, String> {
        let models_dir = self.models_dir.clone();
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            repo_id, filename
        );
        let dest_path = models_dir.join(filename);
        let tag = format!("{}/{}", repo_id, filename);
        let filename_owned = filename.to_string();
        let (tx, rx) = std::sync::mpsc::channel();

        // Ensure cache directory exists
        if let Err(e) = std::fs::create_dir_all(&models_dir) {
            return Err(format!("Failed to create models directory: {}", e));
        }

        std::thread::spawn(move || {
            let _ = tx.send(PullEvent::Progress {
                status: format!("Connecting to {}...", url),
                percent: Some(0.0),
            });

            let resp = ureq::get(&url)
                .config()
                .timeout_global(Some(std::time::Duration::from_secs(7200)))
                .build()
                .call();

            match resp {
                Ok(resp) => {
                    let total_size = resp
                        .headers()
                        .get("content-length")
                        .and_then(|v| v.to_str().ok())
                        .and_then(|s| s.parse::<u64>().ok())
                        .unwrap_or(0);

                    let _ = tx.send(PullEvent::Progress {
                        status: format!(
                            "Downloading {} ({:.1} GB)...",
                            filename_owned,
                            total_size as f64 / 1_073_741_824.0
                        ),
                        percent: Some(0.0),
                    });

                    // Write to a temp file, then rename to avoid partial files
                    let tmp_path = dest_path.with_extension("gguf.part");
                    let file = match std::fs::File::create(&tmp_path) {
                        Ok(f) => f,
                        Err(e) => {
                            let _ =
                                tx.send(PullEvent::Error(format!("Failed to create file: {}", e)));
                            return;
                        }
                    };

                    let mut writer = std::io::BufWriter::new(file);
                    let mut reader = resp.into_body().into_reader();
                    let mut downloaded: u64 = 0;
                    let mut buf = [0u8; 128 * 1024]; // 128 KB buffer
                    let mut last_report = std::time::Instant::now();

                    loop {
                        match std::io::Read::read(&mut reader, &mut buf) {
                            Ok(0) => break, // EOF
                            Ok(n) => {
                                if let Err(e) = std::io::Write::write_all(&mut writer, &buf[..n]) {
                                    let _ =
                                        tx.send(PullEvent::Error(format!("Write error: {}", e)));
                                    let _ = std::fs::remove_file(&tmp_path);
                                    return;
                                }
                                downloaded += n as u64;

                                // Report progress at most every 200ms
                                if last_report.elapsed() >= std::time::Duration::from_millis(200) {
                                    let pct = if total_size > 0 {
                                        downloaded as f64 / total_size as f64 * 100.0
                                    } else {
                                        0.0
                                    };
                                    let dl_gb = downloaded as f64 / 1_073_741_824.0;
                                    let total_gb = total_size as f64 / 1_073_741_824.0;
                                    let _ = tx.send(PullEvent::Progress {
                                        status: format!(
                                            "Downloading {:.1}/{:.1} GB",
                                            dl_gb, total_gb
                                        ),
                                        percent: Some(pct),
                                    });
                                    last_report = std::time::Instant::now();
                                }
                            }
                            Err(e) => {
                                let _ = tx.send(PullEvent::Error(format!("Download error: {}", e)));
                                let _ = std::fs::remove_file(&tmp_path);
                                return;
                            }
                        }
                    }

                    // Flush and rename
                    if let Err(e) = std::io::Write::flush(&mut writer) {
                        let _ = tx.send(PullEvent::Error(format!("Flush error: {}", e)));
                        let _ = std::fs::remove_file(&tmp_path);
                        return;
                    }
                    drop(writer);

                    if let Err(e) = std::fs::rename(&tmp_path, &dest_path) {
                        let _ = tx.send(PullEvent::Error(format!(
                            "Failed to finalize download: {}",
                            e
                        )));
                        let _ = std::fs::remove_file(&tmp_path);
                        return;
                    }

                    let _ = tx.send(PullEvent::Progress {
                        status: "Download complete!".to_string(),
                        percent: Some(100.0),
                    });
                    let _ = tx.send(PullEvent::Done);
                }
                Err(e) => {
                    let _ = tx.send(PullEvent::Error(format!("Download failed: {}", e)));
                }
            }
        });

        Ok(PullHandle {
            model_tag: tag,
            receiver: rx,
        })
    }
}

/// Check if a filename looks like a split GGUF shard (e.g., model-00001-of-00003.gguf).
fn is_split_file(filename: &str) -> bool {
    // Pattern: anything with "-NNNNN-of-NNNNN" before .gguf
    filename.contains("-of-")
}

/// Default directory for llama.cpp GGUF model cache.
fn llamacpp_models_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("LLMFIT_MODELS_DIR") {
        PathBuf::from(dir)
    } else if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home)
            .join(".cache")
            .join("llmfit")
            .join("models")
    } else {
        PathBuf::from("/tmp/.cache/llmfit/models")
    }
}

/// Find a binary in PATH using `which`.
fn find_binary(name: &str) -> Option<String> {
    std::process::Command::new("which")
        .arg(name)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()
        .and_then(|out| {
            if out.status.success() {
                String::from_utf8(out.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
            } else {
                None
            }
        })
}

/// Simple percent-encoding for URL query parameters.
mod urlencoding {
    pub fn encode(s: &str) -> String {
        let mut result = String::with_capacity(s.len() * 3);
        for byte in s.bytes() {
            match byte {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                    result.push(byte as char);
                }
                _ => {
                    result.push('%');
                    result.push_str(&format!("{:02X}", byte));
                }
            }
        }
        result
    }
}

impl ModelProvider for LlamaCppProvider {
    fn name(&self) -> &str {
        "llama.cpp"
    }

    fn is_available(&self) -> bool {
        self.llama_cli.is_some() || self.llama_server.is_some()
    }

    fn installed_models(&self) -> HashSet<String> {
        let mut set = HashSet::new();
        for path in self.list_gguf_files() {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                let lower = stem.to_lowercase();
                set.insert(lower.clone());
                // Also insert a normalized form: strip quantization suffix
                // e.g. "llama-3.1-8b-instruct-q4_k_m" → "llama-3.1-8b-instruct"
                if let Some(base) = strip_gguf_quant_suffix(&lower) {
                    set.insert(base);
                }
            }
        }
        set
    }

    fn start_pull(&self, model_tag: &str) -> Result<PullHandle, String> {
        // model_tag can be:
        // 1. A HuggingFace repo ID like "bartowski/Llama-3.1-8B-Instruct-GGUF"
        // 2. A repo_id/filename like "bartowski/Llama-3.1-8B-Instruct-GGUF/Q4_K_M.gguf"
        // 3. A short search term like "llama-3.1-8b"

        // If it contains a slash and ends with .gguf, treat as repo/file
        if model_tag.matches('/').count() >= 2 && model_tag.ends_with(".gguf") {
            let parts: Vec<&str> = model_tag.splitn(3, '/').collect();
            if parts.len() == 3 {
                let repo = format!("{}/{}", parts[0], parts[1]);
                let filename = parts[2];
                return self.download_gguf(&repo, filename);
            }
        }

        // If it looks like a repo (org/name), list files and pick the best
        if model_tag.contains('/') {
            let files = Self::list_repo_gguf_files(model_tag);
            if files.is_empty() {
                return Err(format!("No GGUF files found in repository '{}'", model_tag));
            }
            // Pick a reasonable default (Q4_K_M or similar)
            if let Some((filename, _)) = Self::select_best_gguf(&files, 999.0) {
                return self.download_gguf(model_tag, &filename);
            }
            // Fallback: just pick the first
            let (filename, _) = &files[0];
            return self.download_gguf(model_tag, filename);
        }

        // Otherwise, search HuggingFace for GGUF repos
        let results = Self::search_hf_gguf(model_tag);
        if results.is_empty() {
            return Err(format!(
                "No GGUF models found on HuggingFace for '{}'",
                model_tag
            ));
        }
        // Use the first result
        let (repo_id, _) = &results[0];
        let files = Self::list_repo_gguf_files(repo_id);
        if files.is_empty() {
            return Err(format!("No GGUF files found in repository '{}'", repo_id));
        }
        if let Some((filename, _)) = Self::select_best_gguf(&files, 999.0) {
            return self.download_gguf(repo_id, &filename);
        }
        let (filename, _) = &files[0];
        self.download_gguf(repo_id, filename)
    }
}

/// Strip quantization suffix from a GGUF file stem.
/// "llama-3.1-8b-instruct-q4_k_m" → "llama-3.1-8b-instruct"
fn strip_gguf_quant_suffix(stem: &str) -> Option<String> {
    let quant_patterns = [
        "-q8_0", "-q6_k", "-q6_k_l", "-q5_k_m", "-q5_k_s", "-q4_k_m", "-q4_k_s", "-q4_0",
        "-q3_k_m", "-q3_k_s", "-q2_k", "-iq4_xs", "-iq3_m", "-iq2_m", "-iq1_m", "-f16", "-f32",
        "-bf16", ".q8_0", ".q6_k", ".q5_k_m", ".q4_k_m", ".q4_0", ".q3_k_m", ".q2_k",
    ];
    for pat in &quant_patterns {
        if let Some(pos) = stem.rfind(pat) {
            return Some(stem[..pos].to_string());
        }
    }
    None
}

// ---------------------------------------------------------------------------
// llama.cpp name-matching helpers
// ---------------------------------------------------------------------------

/// Authoritative mapping from HF repo names to known GGUF repository IDs on HuggingFace.
/// Models not in this table fall back to a heuristic search.
const LLAMACPP_GGUF_MAPPINGS: &[(&str, &str)] = &[
    // Meta Llama
    (
        "llama-3.3-70b-instruct",
        "bartowski/Llama-3.3-70B-Instruct-GGUF",
    ),
    (
        "llama-3.2-3b-instruct",
        "bartowski/Llama-3.2-3B-Instruct-GGUF",
    ),
    (
        "llama-3.2-1b-instruct",
        "bartowski/Llama-3.2-1B-Instruct-GGUF",
    ),
    (
        "llama-3.1-8b-instruct",
        "bartowski/Llama-3.1-8B-Instruct-GGUF",
    ),
    (
        "llama-3.1-70b-instruct",
        "bartowski/Llama-3.1-70B-Instruct-GGUF",
    ),
    (
        "llama-3.1-405b-instruct",
        "bartowski/Meta-Llama-3.1-405B-Instruct-GGUF",
    ),
    (
        "meta-llama-3-8b-instruct",
        "bartowski/Meta-Llama-3-8B-Instruct-GGUF",
    ),
    // Qwen
    (
        "qwen2.5-72b-instruct",
        "bartowski/Qwen2.5-72B-Instruct-GGUF",
    ),
    (
        "qwen2.5-32b-instruct",
        "bartowski/Qwen2.5-32B-Instruct-GGUF",
    ),
    (
        "qwen2.5-14b-instruct",
        "bartowski/Qwen2.5-14B-Instruct-GGUF",
    ),
    ("qwen2.5-7b-instruct", "bartowski/Qwen2.5-7B-Instruct-GGUF"),
    ("qwen2.5-3b-instruct", "bartowski/Qwen2.5-3B-Instruct-GGUF"),
    (
        "qwen2.5-1.5b-instruct",
        "bartowski/Qwen2.5-1.5B-Instruct-GGUF",
    ),
    (
        "qwen2.5-0.5b-instruct",
        "bartowski/Qwen2.5-0.5B-Instruct-GGUF",
    ),
    (
        "qwen2.5-coder-32b-instruct",
        "bartowski/Qwen2.5-Coder-32B-Instruct-GGUF",
    ),
    (
        "qwen2.5-coder-14b-instruct",
        "bartowski/Qwen2.5-Coder-14B-Instruct-GGUF",
    ),
    (
        "qwen2.5-coder-7b-instruct",
        "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
    ),
    ("qwen3-32b", "bartowski/Qwen3-32B-GGUF"),
    ("qwen3-14b", "bartowski/Qwen3-14B-GGUF"),
    ("qwen3-8b", "bartowski/Qwen3-8B-GGUF"),
    ("qwen3-4b", "bartowski/Qwen3-4B-GGUF"),
    ("qwen3-0.6b", "bartowski/Qwen3-0.6B-GGUF"),
    // Mistral
    (
        "mistral-7b-instruct-v0.3",
        "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
    ),
    (
        "mistral-small-24b-instruct-2501",
        "bartowski/Mistral-Small-24B-Instruct-2501-GGUF",
    ),
    (
        "mixtral-8x7b-instruct-v0.1",
        "bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF",
    ),
    // Google Gemma
    ("gemma-3-12b-it", "bartowski/gemma-3-12b-it-GGUF"),
    ("gemma-2-27b-it", "bartowski/gemma-2-27b-it-GGUF"),
    ("gemma-2-9b-it", "bartowski/gemma-2-9b-it-GGUF"),
    ("gemma-2-2b-it", "bartowski/gemma-2-2b-it-GGUF"),
    // Microsoft Phi
    ("phi-4", "bartowski/phi-4-GGUF"),
    ("phi-4-mini-instruct", "bartowski/phi-4-mini-instruct-GGUF"),
    (
        "phi-3.5-mini-instruct",
        "bartowski/Phi-3.5-mini-instruct-GGUF",
    ),
    (
        "phi-3-mini-4k-instruct",
        "bartowski/Phi-3-mini-4k-instruct-GGUF",
    ),
    // DeepSeek
    ("deepseek-r1", "bartowski/DeepSeek-R1-GGUF"),
    (
        "deepseek-r1-distill-qwen-32b",
        "bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF",
    ),
    (
        "deepseek-r1-distill-qwen-14b",
        "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
    ),
    (
        "deepseek-r1-distill-qwen-7b",
        "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",
    ),
    ("deepseek-v3", "bartowski/DeepSeek-V3-GGUF"),
    // Community
    (
        "tinyllama-1.1b-chat-v1.0",
        "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    ),
    ("falcon-7b-instruct", "TheBloke/falcon-7b-instruct-GGUF"),
    (
        "smollm2-135m-instruct",
        "bartowski/SmolLM2-135M-Instruct-GGUF",
    ),
];

/// Look up a known GGUF repo for an HF model name.
fn lookup_gguf_repo(hf_name: &str) -> Option<&'static str> {
    let repo = hf_name
        .split('/')
        .next_back()
        .unwrap_or(hf_name)
        .to_lowercase();
    LLAMACPP_GGUF_MAPPINGS
        .iter()
        .find(|&&(hf_suffix, _)| repo == hf_suffix)
        .map(|&(_, gguf_repo)| gguf_repo)
}

/// Map a HuggingFace model name to candidate GGUF repo IDs.
pub fn hf_name_to_gguf_candidates(hf_name: &str) -> Vec<String> {
    if let Some(repo) = lookup_gguf_repo(hf_name) {
        return vec![repo.to_string()];
    }

    // Heuristic: try common GGUF repo naming patterns
    let base = hf_name.split('/').next_back().unwrap_or(hf_name);

    vec![
        format!("bartowski/{}-GGUF", base),
        format!("ggml-org/{}-GGUF", base),
        format!("TheBloke/{}-GGUF", base),
    ]
}

/// Returns `true` if this HF model has a known GGUF mapping.
pub fn has_gguf_mapping(hf_name: &str) -> bool {
    lookup_gguf_repo(hf_name).is_some()
}

/// Check if a model is installed in the llama.cpp cache.
pub fn is_model_installed_llamacpp(hf_name: &str, installed: &HashSet<String>) -> bool {
    let repo = hf_name
        .split('/')
        .next_back()
        .unwrap_or(hf_name)
        .to_lowercase();

    // Direct match on model name stem
    if installed.contains(&repo) {
        return true;
    }

    // Check with common suffixes stripped
    let stripped = repo
        .replace("-instruct", "")
        .replace("-chat", "")
        .replace("-hf", "")
        .replace("-it", "");

    installed.iter().any(|name| {
        name.contains(&repo) || name.contains(&stripped) || repo.contains(name.as_str())
    })
}

/// Given an HF model name, return the best GGUF repo to pull from.
pub fn gguf_pull_tag(hf_name: &str) -> Option<String> {
    lookup_gguf_repo(hf_name).map(|s| s.to_string())
}

/// Best-effort check that a Hugging Face model repository exists.
pub fn hf_repo_exists(repo_id: &str) -> bool {
    let url = format!("https://huggingface.co/api/models/{}", repo_id);
    ureq::get(&url)
        .config()
        .timeout_global(Some(std::time::Duration::from_millis(1200)))
        .build()
        .call()
        .is_ok()
}

/// Resolve the first GGUF repo that appears to exist remotely.
pub fn first_existing_gguf_repo(hf_name: &str) -> Option<String> {
    if let Some(repo) = gguf_pull_tag(hf_name)
        && hf_repo_exists(&repo)
    {
        return Some(repo);
    }
    let candidates = hf_name_to_gguf_candidates(hf_name);
    candidates.into_iter().find(|repo| hf_repo_exists(repo))
}

// ---------------------------------------------------------------------------
// MLX name-matching helpers
// ---------------------------------------------------------------------------

/// Map a HuggingFace model name to mlx-community repo name candidates.
/// Pattern: mlx-community/{RepoName}-{quant}bit
pub fn hf_name_to_mlx_candidates(hf_name: &str) -> Vec<String> {
    let repo = hf_name.split('/').next_back().unwrap_or(hf_name);

    // Explicit mappings: HF repo suffix → mlx-community repo name (without quant suffix)
    let mappings: &[(&str, &str)] = &[
        // Meta Llama
        ("Llama-3.3-70B-Instruct", "Llama-3.3-70B-Instruct"),
        ("Llama-3.2-3B-Instruct", "Llama-3.2-3B-Instruct"),
        ("Llama-3.2-1B-Instruct", "Llama-3.2-1B-Instruct"),
        ("Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct"),
        ("Llama-3.1-70B-Instruct", "Llama-3.1-70B-Instruct"),
        // Qwen
        ("Qwen2.5-72B-Instruct", "Qwen2.5-72B-Instruct"),
        ("Qwen2.5-32B-Instruct", "Qwen2.5-32B-Instruct"),
        ("Qwen2.5-14B-Instruct", "Qwen2.5-14B-Instruct"),
        ("Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct"),
        ("Qwen2.5-Coder-32B-Instruct", "Qwen2.5-Coder-32B-Instruct"),
        ("Qwen2.5-Coder-14B-Instruct", "Qwen2.5-Coder-14B-Instruct"),
        ("Qwen2.5-Coder-7B-Instruct", "Qwen2.5-Coder-7B-Instruct"),
        ("Qwen3-32B", "Qwen3-32B"),
        ("Qwen3-14B", "Qwen3-14B"),
        ("Qwen3-8B", "Qwen3-8B"),
        ("Qwen3-4B", "Qwen3-4B"),
        // Mistral
        ("Mistral-7B-Instruct-v0.3", "Mistral-7B-Instruct-v0.3"),
        (
            "Mistral-Small-24B-Instruct-2501",
            "Mistral-Small-24B-Instruct-2501",
        ),
        ("Mixtral-8x7B-Instruct-v0.1", "Mixtral-8x7B-Instruct-v0.1"),
        // DeepSeek
        (
            "DeepSeek-R1-Distill-Qwen-32B",
            "DeepSeek-R1-Distill-Qwen-32B",
        ),
        ("DeepSeek-R1-Distill-Qwen-7B", "DeepSeek-R1-Distill-Qwen-7B"),
        // Gemma
        ("gemma-3-12b-it", "gemma-3-12b-it"),
        ("gemma-2-27b-it", "gemma-2-27b-it"),
        ("gemma-2-9b-it", "gemma-2-9b-it"),
        ("gemma-2-2b-it", "gemma-2-2b-it"),
        // Phi
        ("Phi-4", "Phi-4"),
        ("Phi-3.5-mini-instruct", "Phi-3.5-mini-instruct"),
        ("Phi-3-mini-4k-instruct", "Phi-3-mini-4k-instruct"),
    ];

    let repo_lower = repo.to_lowercase();
    for &(hf_suffix, mlx_base) in mappings {
        if repo_lower == hf_suffix.to_lowercase() {
            let base_lower = mlx_base.to_lowercase();
            return vec![
                format!("{}-8bit", base_lower),
                format!("{}-4bit", base_lower),
                base_lower,
            ];
        }
    }

    // Fallback heuristic: strip common suffixes and generate candidates
    let stripped = repo_lower
        .replace("-instruct", "")
        .replace("-chat", "")
        .replace("-hf", "")
        .replace("-it", "");
    vec![
        format!("{}-8bit", repo_lower),
        format!("{}-4bit", repo_lower),
        format!("{}-8bit", stripped),
        format!("{}-4bit", stripped),
        repo_lower,
    ]
}

/// Check if any MLX candidates for an HF model appear in the installed set.
pub fn is_model_installed_mlx(hf_name: &str, installed: &HashSet<String>) -> bool {
    let candidates = hf_name_to_mlx_candidates(hf_name);
    candidates.iter().any(|c| installed.contains(c))
}

/// Given an HF model name, return the best MLX tag to use for pulling.
pub fn mlx_pull_tag(hf_name: &str) -> String {
    let candidates = hf_name_to_mlx_candidates(hf_name);
    // Prefer 4bit (smaller download) for pulling
    candidates
        .iter()
        .find(|c| c.ends_with("-4bit"))
        .cloned()
        .unwrap_or_else(|| {
            candidates.into_iter().next().unwrap_or_else(|| {
                hf_name
                    .split('/')
                    .next_back()
                    .unwrap_or(hf_name)
                    .to_lowercase()
            })
        })
}

// ---------------------------------------------------------------------------
// Ollama name-matching helpers
// ---------------------------------------------------------------------------

/// Authoritative mapping from HF repo name (lowercased, after slash) to Ollama tag.
/// Only models with a known Ollama registry entry are listed here.
/// If a model is not in this table, it cannot be pulled from Ollama.
const OLLAMA_MAPPINGS: &[(&str, &str)] = &[
    // Meta Llama family
    ("llama-3.3-70b-instruct", "llama3.3:70b"),
    ("llama-3.2-11b-vision-instruct", "llama3.2-vision:11b"),
    ("llama-3.2-3b-instruct", "llama3.2:3b"),
    ("llama-3.2-3b", "llama3.2:3b"),
    ("llama-3.2-1b-instruct", "llama3.2:1b"),
    ("llama-3.2-1b", "llama3.2:1b"),
    ("llama-3.1-405b-instruct", "llama3.1:405b"),
    ("llama-3.1-405b", "llama3.1:405b"),
    ("llama-3.1-70b-instruct", "llama3.1:70b"),
    ("llama-3.1-8b-instruct", "llama3.1:8b"),
    ("llama-3.1-8b", "llama3.1:8b"),
    ("meta-llama-3-8b-instruct", "llama3:8b"),
    ("meta-llama-3-8b", "llama3:8b"),
    ("llama-2-7b-hf", "llama2:7b"),
    ("codellama-34b-instruct-hf", "codellama:34b"),
    ("codellama-13b-instruct-hf", "codellama:13b"),
    ("codellama-7b-instruct-hf", "codellama:7b"),
    // Google Gemma
    ("gemma-3-12b-it", "gemma3:12b"),
    ("gemma-2-27b-it", "gemma2:27b"),
    ("gemma-2-9b-it", "gemma2:9b"),
    ("gemma-2-2b-it", "gemma2:2b"),
    // Microsoft Phi
    ("phi-4", "phi4"),
    ("phi-4-mini-instruct", "phi4-mini"),
    ("phi-3.5-mini-instruct", "phi3.5"),
    ("phi-3-mini-4k-instruct", "phi3"),
    ("phi-3-medium-14b-instruct", "phi3:14b"),
    ("phi-2", "phi"),
    ("orca-2-7b", "orca2:7b"),
    ("orca-2-13b", "orca2:13b"),
    // Mistral
    ("mistral-7b-instruct-v0.3", "mistral:7b"),
    ("mistral-7b-instruct-v0.2", "mistral:7b"),
    ("mistral-nemo-instruct-2407", "mistral-nemo"),
    ("mistral-small-24b-instruct-2501", "mistral-small:24b"),
    ("mistral-large-instruct-2407", "mistral-large"),
    ("mixtral-8x7b-instruct-v0.1", "mixtral:8x7b"),
    ("mixtral-8x22b-instruct-v0.1", "mixtral:8x22b"),
    // Qwen 2 / 2.5
    ("qwen2-1.5b-instruct", "qwen2:1.5b"),
    ("qwen2.5-72b-instruct", "qwen2.5:72b"),
    ("qwen2.5-32b-instruct", "qwen2.5:32b"),
    ("qwen2.5-14b-instruct", "qwen2.5:14b"),
    ("qwen2.5-7b-instruct", "qwen2.5:7b"),
    ("qwen2.5-7b", "qwen2.5:7b"),
    ("qwen2.5-3b-instruct", "qwen2.5:3b"),
    ("qwen2.5-1.5b-instruct", "qwen2.5:1.5b"),
    ("qwen2.5-1.5b", "qwen2.5:1.5b"),
    ("qwen2.5-0.5b-instruct", "qwen2.5:0.5b"),
    ("qwen2.5-0.5b", "qwen2.5:0.5b"),
    ("qwen2.5-coder-32b-instruct", "qwen2.5-coder:32b"),
    ("qwen2.5-coder-14b-instruct", "qwen2.5-coder:14b"),
    ("qwen2.5-coder-7b-instruct", "qwen2.5-coder:7b"),
    ("qwen2.5-coder-1.5b-instruct", "qwen2.5-coder:1.5b"),
    ("qwen2.5-coder-0.5b-instruct", "qwen2.5-coder:0.5b"),
    ("qwen2.5-vl-7b-instruct", "qwen2.5vl:7b"),
    ("qwen2.5-vl-3b-instruct", "qwen2.5vl:3b"),
    // Qwen 3
    ("qwen3-235b-a22b", "qwen3:235b"),
    ("qwen3-32b", "qwen3:32b"),
    ("qwen3-30b-a3b", "qwen3:30b-a3b"),
    ("qwen3-30b-a3b-instruct-2507", "qwen3:30b-a3b"),
    ("qwen3-14b", "qwen3:14b"),
    ("qwen3-8b", "qwen3:8b"),
    ("qwen3-4b", "qwen3:4b"),
    ("qwen3-4b-instruct-2507", "qwen3:4b"),
    ("qwen3-1.7b-base", "qwen3:1.7b"),
    ("qwen3-0.6b", "qwen3:0.6b"),
    ("qwen3-coder-30b-a3b-instruct", "qwen3-coder"),
    // DeepSeek
    ("deepseek-v3", "deepseek-v3"),
    ("deepseek-v3.2", "deepseek-v3"),
    ("deepseek-r1", "deepseek-r1"),
    ("deepseek-r1-0528", "deepseek-r1"),
    ("deepseek-r1-distill-qwen-32b", "deepseek-r1:32b"),
    ("deepseek-r1-distill-qwen-14b", "deepseek-r1:14b"),
    ("deepseek-r1-distill-qwen-7b", "deepseek-r1:7b"),
    ("deepseek-coder-v2-lite-instruct", "deepseek-coder-v2:16b"),
    // Community / other
    ("tinyllama-1.1b-chat-v1.0", "tinyllama"),
    ("stablelm-2-1_6b-chat", "stablelm2:1.6b"),
    ("yi-6b-chat", "yi:6b"),
    ("yi-34b-chat", "yi:34b"),
    ("starcoder2-7b", "starcoder2:7b"),
    ("starcoder2-15b", "starcoder2:15b"),
    ("falcon-7b-instruct", "falcon:7b"),
    ("falcon-40b-instruct", "falcon:40b"),
    ("falcon-180b-chat", "falcon:180b"),
    ("falcon3-7b-instruct", "falcon3:7b"),
    ("openchat-3.5-0106", "openchat:7b"),
    ("vicuna-7b-v1.5", "vicuna:7b"),
    ("vicuna-13b-v1.5", "vicuna:13b"),
    ("glm-4-9b-chat", "glm4:9b"),
    ("solar-10.7b-instruct-v1.0", "solar:10.7b"),
    ("zephyr-7b-beta", "zephyr:7b"),
    ("c4ai-command-r-v01", "command-r"),
    (
        "nous-hermes-2-mixtral-8x7b-dpo",
        "nous-hermes2-mixtral:8x7b",
    ),
    ("hermes-3-llama-3.1-8b", "hermes3:8b"),
    ("nomic-embed-text-v1.5", "nomic-embed-text"),
    ("bge-large-en-v1.5", "bge-large"),
    ("smollm2-135m-instruct", "smollm2:135m"),
    ("smollm2-135m", "smollm2:135m"),
];

/// Look up the Ollama tag for an HF repo name. Returns the first match
/// from `OLLAMA_MAPPINGS`, or `None` if the model has no known Ollama equivalent.
fn lookup_ollama_tag(hf_name: &str) -> Option<&'static str> {
    let repo = hf_name
        .split('/')
        .next_back()
        .unwrap_or(hf_name)
        .to_lowercase();
    OLLAMA_MAPPINGS
        .iter()
        .find(|&&(hf_suffix, _)| repo == hf_suffix)
        .map(|&(_, tag)| tag)
}

/// Map a HuggingFace model name to Ollama candidate tags for install checking.
/// Returns candidates from the authoritative mapping table only.
pub fn hf_name_to_ollama_candidates(hf_name: &str) -> Vec<String> {
    match lookup_ollama_tag(hf_name) {
        Some(tag) => vec![tag.to_string()],
        None => vec![],
    }
}

/// Returns `true` if this HF model has a known Ollama registry entry
/// and can be pulled.
pub fn has_ollama_mapping(hf_name: &str) -> bool {
    lookup_ollama_tag(hf_name).is_some()
}

/// Check if any of the Ollama candidates for an HF model appear in the
/// installed set.
pub fn is_model_installed(hf_name: &str, installed: &HashSet<String>) -> bool {
    let candidates = hf_name_to_ollama_candidates(hf_name);
    candidates.iter().any(|c| installed.contains(c))
}

/// Given an HF model name, return the Ollama tag to use for pulling.
/// Returns `None` if the model has no known Ollama mapping.
pub fn ollama_pull_tag(hf_name: &str) -> Option<String> {
    lookup_ollama_tag(hf_name).map(|s| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hf_name_to_mlx_candidates() {
        let candidates = hf_name_to_mlx_candidates("meta-llama/Llama-3.1-8B-Instruct");
        assert!(
            candidates
                .iter()
                .any(|c| c.contains("llama-3.1-8b-instruct"))
        );
        assert!(candidates.iter().any(|c| c.ends_with("-4bit")));
        assert!(candidates.iter().any(|c| c.ends_with("-8bit")));

        let qwen = hf_name_to_mlx_candidates("Qwen/Qwen2.5-Coder-14B-Instruct");
        assert!(
            qwen.iter()
                .any(|c| c.contains("qwen2.5-coder-14b-instruct"))
        );
    }

    #[test]
    fn test_mlx_cache_scan_parsing() {
        // Test that the candidate matching works with cache-style names
        let mut installed = HashSet::new();
        installed.insert("llama-3.1-8b-instruct-4bit".to_string());

        assert!(is_model_installed_mlx(
            "meta-llama/Llama-3.1-8B-Instruct",
            &installed
        ));
        // Should not match unrelated model
        assert!(!is_model_installed_mlx(
            "Qwen/Qwen2.5-7B-Instruct",
            &installed
        ));
    }

    #[test]
    fn test_is_model_installed_mlx() {
        let mut installed = HashSet::new();
        installed.insert("qwen2.5-coder-14b-instruct-8bit".to_string());

        assert!(is_model_installed_mlx(
            "Qwen/Qwen2.5-Coder-14B-Instruct",
            &installed
        ));
        assert!(!is_model_installed_mlx(
            "Qwen/Qwen2.5-14B-Instruct",
            &installed
        ));
    }

    #[test]
    fn test_qwen_coder_14b_matches_coder_entry() {
        // "qwen2.5-coder:14b" from `ollama list` should match
        // the HF entry "Qwen/Qwen2.5-Coder-14B-Instruct", NOT
        // the base "Qwen/Qwen2.5-14B-Instruct".
        let mut installed = HashSet::new();
        installed.insert("qwen2.5-coder:14b".to_string());
        installed.insert("qwen2.5-coder".to_string());

        assert!(is_model_installed(
            "Qwen/Qwen2.5-Coder-14B-Instruct",
            &installed
        ));
        // Must NOT match the non-coder model
        assert!(!is_model_installed("Qwen/Qwen2.5-14B-Instruct", &installed));
    }

    #[test]
    fn test_qwen_base_does_not_match_coder() {
        // "qwen2.5:14b" from `ollama list` should match the base model,
        // not the coder variant.
        let mut installed = HashSet::new();
        installed.insert("qwen2.5:14b".to_string());
        installed.insert("qwen2.5".to_string());

        assert!(is_model_installed("Qwen/Qwen2.5-14B-Instruct", &installed));
        assert!(!is_model_installed(
            "Qwen/Qwen2.5-Coder-14B-Instruct",
            &installed
        ));
    }

    #[test]
    fn test_candidates_for_coder_model() {
        let candidates = hf_name_to_ollama_candidates("Qwen/Qwen2.5-Coder-14B-Instruct");
        assert!(candidates.contains(&"qwen2.5-coder:14b".to_string()));
    }

    #[test]
    fn test_candidates_for_base_model() {
        let candidates = hf_name_to_ollama_candidates("Qwen/Qwen2.5-14B-Instruct");
        assert!(candidates.contains(&"qwen2.5:14b".to_string()));
    }

    #[test]
    fn test_llama_mapping() {
        let candidates = hf_name_to_ollama_candidates("meta-llama/Llama-3.1-8B-Instruct");
        assert!(candidates.contains(&"llama3.1:8b".to_string()));
    }

    #[test]
    fn test_deepseek_coder_mapping() {
        let candidates =
            hf_name_to_ollama_candidates("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct");
        assert!(candidates.contains(&"deepseek-coder-v2:16b".to_string()));
    }
}
