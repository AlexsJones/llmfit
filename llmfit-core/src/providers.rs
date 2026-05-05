//! Runtime model providers (Ollama, llama.cpp, MLX, Docker Model Runner, LM Studio, vLLM).
//!
//! Each provider can list locally installed models and pull new ones.

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
    /// Fallback URL to try when `base_url` is unreachable.
    /// Set when using the default `localhost` address so that systems where
    /// `localhost` resolves to `::1` (IPv6) can fall back to `127.0.0.1`.
    fallback_url: Option<String>,
}

fn normalize_ollama_host(raw: &str) -> Option<String> {
    let host = raw.trim();
    if host.is_empty() {
        return None;
    }

    if host.starts_with("http://") || host.starts_with("https://") {
        return Some(host.to_string());
    }

    if host.contains("://") {
        // Unsupported scheme (e.g. ftp://)
        return None;
    }

    // Port-only value like ":11434" — prepend localhost
    if host.starts_with(':') {
        return Some(format!("http://localhost{host}"));
    }

    Some(format!("http://{host}"))
}

/// Returns true if the URL's host is a wildcard bind address — `0.0.0.0`
/// (IPv4) or `[::]` (IPv6). Servers listen on these to accept traffic on
/// every interface, but they are never valid as a connect target. When
/// Ollama is started with `OLLAMA_HOST=0.0.0.0`, that value leaks into the
/// environment and we must not pass it to a client.
fn is_wildcard_bind_address(url: &str) -> bool {
    let after_scheme = url
        .strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))
        .unwrap_or(url);
    let host_port = after_scheme
        .split(['/', '?', '#'])
        .next()
        .unwrap_or(after_scheme);

    if let Some(rest) = host_port.strip_prefix('[') {
        if let Some(end_idx) = rest.find(']') {
            let host = &rest[..end_idx];
            return host == "::" || host == "0:0:0:0:0:0:0:0";
        }
        return false;
    }

    let host = host_port.split(':').next().unwrap_or("");
    host == "0.0.0.0"
}

impl Default for OllamaProvider {
    fn default() -> Self {
        let explicit = std::env::var("OLLAMA_HOST").ok().and_then(|raw| {
            let Some(normalized) = normalize_ollama_host(&raw) else {
                eprintln!(
                    "Warning: could not parse OLLAMA_HOST='{}'. Expected host:port or http(s)://host:port",
                    raw
                );
                return None;
            };
            if is_wildcard_bind_address(&normalized) {
                eprintln!(
                    "Warning: OLLAMA_HOST='{}' is a wildcard bind address; falling back to localhost.",
                    raw
                );
                return None;
            }
            Some(normalized)
        });

        if let Some(base_url) = explicit {
            // User supplied an explicit host — use it as-is, no fallback.
            Self {
                base_url,
                fallback_url: None,
            }
        } else {
            // Default: try `localhost` first; fall back to `127.0.0.1` for
            // systems where `localhost` resolves to the IPv6 loopback `::1`
            // while Ollama is only listening on the IPv4 `127.0.0.1`.
            Self {
                base_url: "http://localhost:11434".to_string(),
                fallback_url: Some("http://127.0.0.1:11434".to_string()),
            }
        }
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

    /// Delete a model from Ollama via its API.
    pub fn delete_model(&self, model_tag: &str) -> Result<(), String> {
        // Ollama DELETE /api/delete requires a JSON body.
        // ureq v3's delete() doesn't support request bodies, so we build a
        // raw http::Request and pass it to the agent's `run()` method.
        let body = serde_json::json!({ "name": model_tag }).to_string();
        let url = self.api_url("delete");
        let request = http::Request::builder()
            .method("DELETE")
            .uri(&url)
            .header("content-type", "application/json")
            .body(body)
            .map_err(|e| format!("Failed to build request: {}", e))?;
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(std::time::Duration::from_secs(10)))
            .build()
            .into();
        let resp = agent
            .run(request)
            .map_err(|e| format!("Ollama delete request failed: {}", e))?;
        if resp.status() == 200 {
            Ok(())
        } else {
            Err(format!("Ollama returned status {}", resp.status()))
        }
    }

    /// Single-pass startup probe to avoid duplicate `/api/tags` calls.
    /// Returns `(available, installed_models)`.
    /// When the primary URL (`localhost`) fails and a fallback (`127.0.0.1`)
    /// is configured, the fallback is tried and—if successful—adopted as the
    /// provider's base URL for all subsequent requests (pull, show, …).
    pub fn detect_with_installed(&mut self) -> (bool, HashSet<String>, usize) {
        let mut set = HashSet::new();

        let primary_ok = ureq::get(&self.api_url("tags"))
            .config()
            .timeout_global(Some(std::time::Duration::from_millis(800)))
            .build()
            .call();

        let resp = match primary_ok {
            Ok(r) => r,
            Err(_) => {
                // Primary URL failed — try the fallback if one is set.
                let Some(ref fallback) = self.fallback_url.clone() else {
                    return (false, set, 0);
                };
                let fallback_url = format!("{}/api/tags", fallback.trim_end_matches('/'));
                let Ok(r) = ureq::get(&fallback_url)
                    .config()
                    .timeout_global(Some(std::time::Duration::from_millis(800)))
                    .build()
                    .call()
                else {
                    return (false, set, 0);
                };
                // Fallback worked: adopt it so that pull/show use 127.0.0.1.
                self.base_url = fallback.clone();
                self.fallback_url = None;
                r
            }
        };

        let Ok(tags): Result<TagsResponse, _> = resp.into_body().read_json() else {
            return (true, set, 0);
        };
        let count = tags.models.len();
        for m in tags.models {
            let lower = m.name.to_lowercase();
            set.insert(lower.clone());
            if let Some(family) = lower.split(':').next() {
                set.insert(family.to_string());
            }
        }
        (true, set, count)
    }

    /// Like `installed_models`, but also returns the true model count.
    /// The HashSet may have fewer entries than 2*count due to family-name deduplication,
    /// so `len() / 2` is unreliable for counting models.
    pub fn installed_models_counted(&self) -> (HashSet<String>, usize) {
        let mut set = HashSet::new();
        let Ok(resp) = ureq::get(&self.api_url("tags"))
            .config()
            .timeout_global(Some(std::time::Duration::from_secs(5)))
            .build()
            .call()
        else {
            return (set, 0);
        };
        let Ok(tags): Result<TagsResponse, _> = resp.into_body().read_json() else {
            return (set, 0);
        };
        let count = tags.models.len();
        for m in tags.models {
            let lower = m.name.to_lowercase();
            set.insert(lower.clone());
            if let Some(family) = lower.split(':').next() {
                set.insert(family.to_string());
            }
        }
        (set, count)
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
        let (set, _) = self.installed_models_counted();
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

fn is_likely_mlx_repo(owner: &str, repo: &str) -> bool {
    let owner_lower = owner.to_lowercase();
    let repo_lower = repo.to_lowercase();
    owner_lower == "mlx-community"
        || repo_lower.contains("-mlx-")
        || repo_lower.ends_with("-mlx")
        || repo_lower.contains("mlx-")
        || repo_lower.ends_with("mlx")
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
        let Some(rest) = name_str.strip_prefix("models--") else {
            continue;
        };
        let mut parts = rest.splitn(2, "--");
        let Some(owner) = parts.next() else {
            continue;
        };
        let Some(repo) = parts.next() else {
            continue;
        };

        if !is_likely_mlx_repo(owner, repo) {
            continue;
        }

        let owner_lower = owner.to_lowercase();
        let repo_lower = repo.to_lowercase();
        set.insert(format!("{}/{}", owner_lower, repo_lower));
        set.insert(repo_lower);
    }
    set
}

fn dirs_hf_cache() -> std::path::PathBuf {
    if let Ok(cache) = std::env::var("HF_HOME") {
        std::path::PathBuf::from(cache).join("hub")
    } else if let Some(cache) = dirs::cache_dir() {
        cache.join("huggingface").join("hub")
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
        let repo_id = if model_tag.contains('/') {
            model_tag.to_string()
        } else {
            format!("mlx-community/{}", model_tag)
        };
        let repo_for_thread = repo_id.clone();
        let (tx, rx) = std::sync::mpsc::channel();

        // Resolve the hf binary path before spawning the thread so we can
        // give a clear "not found" error instead of a confusing OS error.
        let hf_bin = find_binary("hf").ok_or_else(|| {
            "hf not found in PATH. Install it with: uv tool install 'huggingface_hub[cli]'"
                .to_string()
        })?;

        std::thread::spawn(move || {
            let _ = tx.send(PullEvent::Progress {
                status: format!("Downloading {}...", repo_for_thread),
                percent: None,
            });

            // Download from Hugging Face using their CLI tool.
            // `--` terminates option parsing so a repo id beginning with `-`
            // (reachable via the unauthenticated localhost /api/v1/download
            // endpoint) cannot be misinterpreted as a flag like --local-dir.
            let result = std::process::Command::new(&hf_bin)
                .args(["download", "--", &repo_for_thread])
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .output();

            match result {
                Ok(output) if output.status.success() => {
                    let _ = tx.send(PullEvent::Done);
                }
                Ok(output) => {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    let _ = tx.send(PullEvent::Error(format!(
                        "hf download failed (exit {}): {}",
                        output.status.code().unwrap_or(-1),
                        stderr.trim()
                    )));
                }
                Err(e) => {
                    let _ = tx.send(PullEvent::Error(format!("failed to run hf: {e}")));
                }
            }
        });

        Ok(PullHandle {
            model_tag: repo_id,
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
    /// Whether a running llama-server was detected via health probe.
    server_running: bool,
}

impl Default for LlamaCppProvider {
    fn default() -> Self {
        let models_dir = llamacpp_models_dir();
        let llama_cli = find_binary("llama-cli");
        let llama_server = find_binary("llama-server");

        // If no binaries found, check if a server is already running
        let server_running = if llama_cli.is_none() && llama_server.is_none() {
            let port = std::env::var("LLAMA_SERVER_PORT").unwrap_or_else(|_| "8080".to_string());
            probe_llama_server(&format!("http://localhost:{}", port))
        } else {
            false
        };

        Self {
            models_dir,
            llama_cli,
            llama_server,
            server_running,
        }
    }
}

impl LlamaCppProvider {
    pub fn new() -> Self {
        Self::default()
    }

    /// Like `installed_models`, but also returns the true GGUF file count.
    /// The HashSet may have fewer entries than 2*count due to deduplication
    /// when stripping quantization suffixes, so `len() / 2` is unreliable.
    pub fn installed_models_counted(&self) -> (HashSet<String>, usize) {
        let mut set = HashSet::new();
        let mut count = 0usize;
        for path in self.list_gguf_files() {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                count += 1;
                let lower = stem.to_lowercase();
                set.insert(lower.clone());
                if let Some(base) = strip_gguf_quant_suffix(&lower) {
                    set.insert(base);
                }
            }
        }
        (set, count)
    }

    /// Return the directory where GGUF models are cached.
    pub fn models_dir(&self) -> &std::path::Path {
        &self.models_dir
    }

    /// Override the models directory at runtime.
    pub fn set_models_dir(&mut self, dir: PathBuf) {
        self.models_dir = dir;
    }

    /// Delete a GGUF model file by tag (file stem match).
    pub fn delete_model(&self, model_tag: &str) -> Result<(), String> {
        let tag_lower = model_tag.to_lowercase();
        for path in self.list_gguf_files() {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str())
                && stem.to_lowercase() == tag_lower
            {
                return std::fs::remove_file(&path)
                    .map_err(|e| format!("Failed to delete {}: {}", path.display(), e));
            }
        }
        Err(format!("Model file not found for '{}'", model_tag))
    }

    /// Path to `llama-cli` if detected.
    pub fn llama_cli_path(&self) -> Option<&str> {
        self.llama_cli.as_deref()
    }

    /// Path to `llama-server` if detected.
    pub fn llama_server_path(&self) -> Option<&str> {
        self.llama_server.as_deref()
    }

    /// Whether a running llama-server was detected via health probe.
    pub fn server_running(&self) -> bool {
        self.server_running
    }

    /// Return a short status hint describing how llama.cpp was (or wasn't) detected.
    pub fn detection_hint(&self) -> &'static str {
        if self.llama_cli.is_some() || self.llama_server.is_some() {
            ""
        } else if self.server_running {
            "server detected"
        } else {
            "not in PATH, set LLAMA_CPP_PATH"
        }
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
        let url = format!(
            "https://huggingface.co/api/models/{}/tree/main?recursive=true",
            repo_id
        );
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
        parse_repo_gguf_entries(entries)
    }

    /// Select the best GGUF file from a repo that fits within a memory budget.
    /// Prefers higher quality quantizations (Q8 > Q6 > Q5 > Q4 > Q3 > Q2).
    /// `budget_gb` is the available memory in gigabytes.
    ///
    /// Sharded models (e.g. `model-00001-of-00003.gguf`) are treated as a
    /// single candidate: the returned path is the first shard and the
    /// returned size is the sum of all shards in the set. The download path
    /// expands the first shard back into the full set.
    pub fn select_best_gguf(files: &[(String, u64)], budget_gb: f64) -> Option<(String, u64)> {
        // Quant preference order (best quality first)
        let quant_order = [
            "Q8_0", "q8_0", "Q6_K", "q6_k", "Q6_K_L", "q6_k_l", "Q5_K_M", "q5_k_m", "Q5_K_S",
            "q5_k_s", "Q4_K_M", "q4_k_m", "Q4_K_S", "q4_k_s", "Q4_0", "q4_0", "Q3_K_M", "q3_k_m",
            "Q3_K_S", "q3_k_s", "Q2_K", "q2_k", "IQ4_XS", "iq4_xs", "IQ3_M", "iq3_m", "IQ2_M",
            "iq2_m", "IQ1_M", "iq1_m",
        ];
        let budget_bytes = (budget_gb * 1024.0 * 1024.0 * 1024.0) as u64;
        let candidates = build_gguf_candidates(files);

        // Try each quant level in preference order
        for quant in &quant_order {
            for (filename, size) in &candidates {
                if *size > 0 && *size <= budget_bytes && filename.contains(quant) {
                    return Some((filename.clone(), *size));
                }
            }
        }

        // Fallback: largest candidate that still fits
        let mut fitting: Vec<_> = candidates
            .iter()
            .filter(|(_, s)| *s > 0 && *s <= budget_bytes)
            .collect();
        fitting.sort_by_key(|(_, s)| *s);
        fitting.last().map(|(f, s)| (f.clone(), *s))
    }

    /// Download a GGUF file from a HuggingFace repository.
    /// `repo_id` is e.g. "bartowski/Llama-3.1-8B-Instruct-GGUF"
    /// `filename` is e.g. "Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    ///
    /// If `filename` is one shard of a multi-part model
    /// (e.g. `...-00001-of-00003.gguf`), all sibling shards are fetched from
    /// the repo tree and downloaded sequentially.
    pub fn download_gguf(&self, repo_id: &str, filename: &str) -> Result<PullHandle, String> {
        // Validate the repo path (may include subdirectories like "Q4_K_M/model.gguf")
        validate_gguf_repo_path(filename)?;

        // If this looks like a shard, expand to the full set by listing the
        // repo tree. Fall through to a single-file download otherwise (or if
        // expansion fails, e.g. the listing is empty).
        let paths: Vec<String> = if parse_shard_info(filename).is_some() {
            let listing = Self::list_repo_gguf_files(repo_id);
            match collect_shard_set(&listing, filename) {
                Some(shards) if !shards.is_empty() => shards.into_iter().map(|(f, _)| f).collect(),
                _ => vec![filename.to_string()],
            }
        } else {
            vec![filename.to_string()]
        };

        self.download_gguf_paths(repo_id, paths)
    }

    /// Download one or more GGUF files from the same HuggingFace repository
    /// into the local cache. Used by `download_gguf` to handle shard sets.
    fn download_gguf_paths(&self, repo_id: &str, paths: Vec<String>) -> Result<PullHandle, String> {
        if paths.is_empty() {
            return Err("download_gguf_paths called with no paths".to_string());
        }

        let models_dir = self.models_dir.clone();

        // Validate every path and pre-compute (url, dest_path) pairs.
        let mut jobs: Vec<(String, PathBuf)> = Vec::with_capacity(paths.len());
        for path in &paths {
            validate_gguf_repo_path(path)?;
            let local_filename = std::path::Path::new(path)
                .file_name()
                .and_then(|n| n.to_str())
                .ok_or_else(|| format!("Invalid filename in path: {}", path))?;
            validate_gguf_filename(local_filename)?;
            let dest_path = models_dir.join(local_filename);

            // Final safety check: ensure resolved path stays within models_dir
            if let (Ok(canonical_dir), Ok(canonical_dest)) = (
                std::fs::create_dir_all(&models_dir).and_then(|_| models_dir.canonicalize()),
                dest_path
                    .parent()
                    .ok_or_else(|| std::io::Error::other("no parent"))
                    .and_then(|p| {
                        std::fs::create_dir_all(p)?;
                        p.canonicalize()
                    }),
            ) && !canonical_dest.starts_with(&canonical_dir)
            {
                return Err(format!(
                    "Security: download path escapes cache directory: {}",
                    dest_path.display()
                ));
            }

            let url = format!("https://huggingface.co/{}/resolve/main/{}", repo_id, path);
            jobs.push((url, dest_path));
        }

        let tag = format!("{}/{}", repo_id, paths[0]);
        let total_parts = jobs.len();
        let (tx, rx) = std::sync::mpsc::channel();

        std::thread::spawn(move || {
            for (idx, (url, dest_path)) in jobs.into_iter().enumerate() {
                let part_num = idx + 1;
                let part_label = if total_parts > 1 {
                    format!("[{}/{}] ", part_num, total_parts)
                } else {
                    String::new()
                };
                let display_name = dest_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_string();

                let _ = tx.send(PullEvent::Progress {
                    status: format!("{}Connecting to {}...", part_label, display_name),
                    percent: Some(0.0),
                });

                let resp = ureq::get(&url)
                    .config()
                    .timeout_global(Some(std::time::Duration::from_secs(7200)))
                    .build()
                    .call();

                let resp = match resp {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = tx.send(PullEvent::Error(format!(
                            "{}Download failed: {}",
                            part_label, e
                        )));
                        return;
                    }
                };

                let total_size = resp
                    .headers()
                    .get("content-length")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(0);

                let _ = tx.send(PullEvent::Progress {
                    status: format!(
                        "{}Downloading {} ({:.1} GB)...",
                        part_label,
                        display_name,
                        total_size as f64 / 1_073_741_824.0
                    ),
                    percent: Some(0.0),
                });

                // Write to a temp file, then rename to avoid partial files.
                // Remove any pre-existing entry and open with create_new
                // (O_EXCL) so a planted symlink at tmp_path cannot redirect
                // the write outside models_dir.
                let tmp_path = dest_path.with_extension("gguf.part");
                let _ = std::fs::remove_file(&tmp_path);
                let file = match std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&tmp_path)
                {
                    Ok(f) => f,
                    Err(e) => {
                        let _ = tx.send(PullEvent::Error(format!("Failed to create file: {}", e)));
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
                                let _ = tx.send(PullEvent::Error(format!("Write error: {}", e)));
                                let _ = std::fs::remove_file(&tmp_path);
                                return;
                            }
                            downloaded += n as u64;

                            if last_report.elapsed() >= std::time::Duration::from_millis(200) {
                                // Per-part percent (kept simple; aggregate progress
                                // across shards is shown via the [i/N] label).
                                let pct = if total_size > 0 {
                                    downloaded as f64 / total_size as f64 * 100.0
                                } else {
                                    0.0
                                };
                                let dl_gb = downloaded as f64 / 1_073_741_824.0;
                                let total_gb = total_size as f64 / 1_073_741_824.0;
                                let _ = tx.send(PullEvent::Progress {
                                    status: format!(
                                        "{}Downloading {:.1}/{:.1} GB",
                                        part_label, dl_gb, total_gb
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

                if let Err(e) = std::io::Write::flush(&mut writer) {
                    let _ = tx.send(PullEvent::Error(format!("Flush error: {}", e)));
                    let _ = std::fs::remove_file(&tmp_path);
                    return;
                }
                drop(writer);

                // Sanity check: refuse to keep an obviously bogus tiny file
                // when content-length advertised something larger. This
                // catches truncated transfers and HTML error responses.
                if total_size > 0 && downloaded < total_size {
                    let _ = std::fs::remove_file(&tmp_path);
                    let _ = tx.send(PullEvent::Error(format!(
                        "{}Truncated download: got {} bytes, expected {}",
                        part_label, downloaded, total_size
                    )));
                    return;
                }

                if let Err(e) = std::fs::rename(&tmp_path, &dest_path) {
                    let _ = tx.send(PullEvent::Error(format!(
                        "Failed to finalize download: {}",
                        e
                    )));
                    let _ = std::fs::remove_file(&tmp_path);
                    return;
                }

                let _ = tx.send(PullEvent::Progress {
                    status: format!("{}Saved {}", part_label, display_name),
                    percent: Some(100.0),
                });
            }

            let _ = tx.send(PullEvent::Progress {
                status: "Download complete!".to_string(),
                percent: Some(100.0),
            });
            let _ = tx.send(PullEvent::Done);
        });

        Ok(PullHandle {
            model_tag: tag,
            receiver: rx,
        })
    }
}

/// Validate a GGUF filename used for local cache writes.
fn validate_gguf_filename(filename: &str) -> Result<(), String> {
    if filename.is_empty() {
        return Err("GGUF filename must not be empty".to_string());
    }

    if filename.contains('/') || filename.contains('\\') {
        return Err(format!(
            "Security: path separators not allowed in GGUF filename: {}",
            filename
        ));
    }

    let path = std::path::Path::new(filename);

    if path.is_absolute() {
        return Err(format!(
            "Security: absolute paths not allowed in GGUF filename: {}",
            filename
        ));
    }

    if !filename.ends_with(".gguf") {
        return Err(format!(
            "GGUF filename must end in .gguf, got: {}",
            filename
        ));
    }

    if path.file_name().and_then(|n| n.to_str()) != Some(filename) {
        return Err(format!(
            "Security: GGUF filename must be a basename without path components: {}",
            filename
        ));
    }

    Ok(())
}

/// If `filename` ends with `-NNNNN-of-MMMMM.gguf`, return `(index, total)`.
/// Both numbers must be ASCII digits, `index >= 1`, and `index <= total`.
fn parse_shard_info(filename: &str) -> Option<(u32, u32)> {
    let stem = filename.strip_suffix(".gguf")?;
    let of_pos = stem.rfind("-of-")?;
    let total_str = &stem[of_pos + 4..];
    if total_str.is_empty() || !total_str.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let total: u32 = total_str.parse().ok()?;
    let before = &stem[..of_pos];
    let dash_pos = before.rfind('-')?;
    let index_str = &before[dash_pos + 1..];
    if index_str.is_empty() || !index_str.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let index: u32 = index_str.parse().ok()?;
    if index == 0 || index > total {
        return None;
    }
    Some((index, total))
}

/// Given a shard path and a listing of repo files, return all sibling shards
/// in the same set, sorted by index. Returns `None` if `path` isn't a shard.
/// The returned vec is empty only if no matching siblings exist (which
/// shouldn't normally happen since the path itself is a shard).
pub fn collect_shard_set(files: &[(String, u64)], path: &str) -> Option<Vec<(String, u64)>> {
    let (_, total) = parse_shard_info(path)?;
    let stem = path.strip_suffix(".gguf")?;
    let of_pos = stem.rfind("-of-")?;
    let before = &stem[..of_pos];
    let dash_pos = before.rfind('-')?;
    // `prefix` includes the trailing '-' that separates the shard index.
    let prefix = &path[..=dash_pos];
    // `suffix` is the "-of-MMMMM.gguf" tail (positions in `path` and `stem`
    // align since `stem` is just `path` minus the trailing ".gguf").
    let suffix = &path[of_pos..];

    let mut matches: Vec<(u32, String, u64)> = files
        .iter()
        .filter_map(|(f, s)| {
            if !f.starts_with(prefix) || !f.ends_with(suffix) {
                return None;
            }
            let (idx, t) = parse_shard_info(f)?;
            if t != total {
                return None;
            }
            Some((idx, f.clone(), *s))
        })
        .collect();
    matches.sort_by_key(|(i, _, _)| *i);
    if matches.is_empty() {
        return None;
    }
    Some(matches.into_iter().map(|(_, f, s)| (f, s)).collect())
}

/// Convert a flat repo file listing into selection candidates. Each shard
/// group is collapsed to a single entry whose path is the first shard and
/// whose size is the sum of all shards. Non-shard files are passed through
/// unchanged. Order is preserved relative to the first occurrence.
fn build_gguf_candidates(files: &[(String, u64)]) -> Vec<(String, u64)> {
    let mut seen_groups: HashSet<String> = HashSet::new();
    let mut out: Vec<(String, u64)> = Vec::new();
    for (f, s) in files {
        if parse_shard_info(f).is_some() {
            // Build a stable group key from prefix + suffix.
            let Some(stem) = f.strip_suffix(".gguf") else {
                continue;
            };
            let Some(of_pos) = stem.rfind("-of-") else {
                continue;
            };
            let before = &stem[..of_pos];
            let Some(dash_pos) = before.rfind('-') else {
                continue;
            };
            let key = format!("{}|{}", &f[..=dash_pos], &f[of_pos..]);
            if !seen_groups.insert(key) {
                continue;
            }
            if let Some(shards) = collect_shard_set(files, f) {
                let total: u64 = shards.iter().map(|(_, sz)| *sz).sum();
                let rep = shards[0].0.clone();
                out.push((rep, total));
            }
        } else {
            out.push((f.clone(), *s));
        }
    }
    out
}

/// Validate a GGUF path returned from the HuggingFace API.
/// Unlike `validate_gguf_filename`, this allows subdirectory paths (e.g.
/// `Q4_K_M/model.gguf`) but still rejects path traversal and non-GGUF files.
fn validate_gguf_repo_path(path: &str) -> Result<(), String> {
    if path.is_empty() {
        return Err("GGUF path must not be empty".to_string());
    }

    // Reject path-traversal components
    for component in path.split('/') {
        if component == ".." || component == "." {
            return Err(format!(
                "Security: path traversal not allowed in GGUF path: {}",
                path
            ));
        }
    }

    // Reject backslashes (Windows-style paths)
    if path.contains('\\') {
        return Err(format!(
            "Security: backslash not allowed in GGUF path: {}",
            path
        ));
    }

    // Reject absolute paths
    if path.starts_with('/') {
        return Err(format!(
            "Security: absolute paths not allowed in GGUF path: {}",
            path
        ));
    }

    if !path.ends_with(".gguf") {
        return Err(format!("GGUF path must end in .gguf, got: {}", path));
    }

    Ok(())
}

fn parse_repo_gguf_entries(entries: Vec<serde_json::Value>) -> Vec<(String, u64)> {
    entries
        .into_iter()
        .filter_map(|e| {
            let path = e.get("path")?.as_str()?.to_string();
            if validate_gguf_repo_path(&path).is_err() {
                return None;
            }
            let size = e.get("size").and_then(|v| v.as_u64()).unwrap_or(0);
            // Skip split files (e.g., model-00001-of-00003.gguf) but not the
            // primary file. We look for files that look like quantized models.
            Some((path, size))
        })
        .collect()
}

/// Default directory for llama.cpp GGUF model cache.
pub fn llamacpp_models_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("LLMFIT_MODELS_DIR") {
        PathBuf::from(dir)
    } else if let Some(cache) = dirs::cache_dir() {
        cache.join("llmfit").join("models")
    } else {
        PathBuf::from(".llmfit").join("models")
    }
}

/// Find a binary by checking `LLAMA_CPP_PATH` env var, common install
/// locations, and finally the system PATH via `which`.
fn find_binary(name: &str) -> Option<String> {
    // 1. Check LLAMA_CPP_PATH env var first
    if let Ok(dir) = std::env::var("LLAMA_CPP_PATH") {
        let candidate = PathBuf::from(&dir).join(name);
        if candidate.is_file() {
            return Some(candidate.to_string_lossy().to_string());
        }
    }

    // 2. Check common install locations
    let mut common_dirs: Vec<PathBuf> = vec![
        PathBuf::from("/usr/local/bin"),
        PathBuf::from("/opt/llama.cpp/build/bin"),
    ];
    if let Some(home) = dirs::home_dir() {
        common_dirs.push(home.join(".local").join("bin"));
    }
    for dir in common_dirs {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate.to_string_lossy().to_string());
        }
    }

    // 3. Fall back to PATH lookup
    which::which(name)
        .ok()
        .map(|p| p.to_string_lossy().to_string())
}

/// Check if a llama-server is reachable at the given URL by probing its
/// health endpoint. Returns `true` if the server responds.
fn probe_llama_server(base_url: &str) -> bool {
    let url = format!("{}/health", base_url.trim_end_matches('/'));
    std::process::Command::new("curl")
        .args(["-sf", "--max-time", "2", &url])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
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
        self.llama_cli.is_some() || self.llama_server.is_some() || self.server_running
    }

    fn installed_models(&self) -> HashSet<String> {
        let (set, _) = self.installed_models_counted();
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

// ---------------------------------------------------------------------------
// Docker Model Runner provider
// ---------------------------------------------------------------------------

/// Docker Model Runner — Docker Desktop's built-in model serving feature.
///
/// Exposes an OpenAI-compatible API at `http://localhost:12434` by default.
/// Models are listed via `GET /engines` and pulled via `docker model pull`.
pub struct DockerModelRunnerProvider {
    base_url: String,
}

fn normalize_docker_mr_host(raw: &str) -> Option<String> {
    let host = raw.trim();
    if host.is_empty() {
        return None;
    }

    if host.starts_with("http://") || host.starts_with("https://") {
        return Some(host.to_string());
    }

    if host.contains("://") {
        return None;
    }

    Some(format!("http://{host}"))
}

impl Default for DockerModelRunnerProvider {
    fn default() -> Self {
        let base_url = std::env::var("DOCKER_MODEL_RUNNER_HOST")
            .ok()
            .and_then(|raw| {
                let normalized = normalize_docker_mr_host(&raw);
                if normalized.is_none() {
                    eprintln!(
                        "Warning: could not parse DOCKER_MODEL_RUNNER_HOST='{}'. \
                         Expected host:port or http(s)://host:port",
                        raw
                    );
                }
                normalized
            })
            .unwrap_or_else(|| "http://localhost:12434".to_string());
        Self { base_url }
    }
}

impl DockerModelRunnerProvider {
    pub fn new() -> Self {
        Self::default()
    }

    fn models_url(&self) -> String {
        format!("{}/v1/models", self.base_url.trim_end_matches('/'))
    }

    /// Single-pass startup probe.
    /// Returns `(available, installed_models, count)`.
    pub fn detect_with_installed(&self) -> (bool, HashSet<String>, usize) {
        let mut set = HashSet::new();
        let Ok(resp) = ureq::get(&self.models_url())
            .config()
            .timeout_global(Some(std::time::Duration::from_millis(800)))
            .build()
            .call()
        else {
            return (false, set, 0);
        };

        let Ok(list) = resp.into_body().read_json::<DockerModelList>() else {
            return (true, set, 0);
        };
        let engines = list.data;
        let count = engines.len();
        for e in engines {
            let lower = e.id.to_lowercase();
            set.insert(lower.clone());
            // Also insert the model part after the namespace (e.g. "ai/llama3.1" → "llama3.1")
            if let Some(name) = lower.split('/').next_back()
                && name != lower
            {
                set.insert(name.to_string());
            }
            // Strip quantization tag if present (e.g. "llama3.1:8B-Q4_K_M" → "llama3.1:8b")
            if let Some(base) = lower.split(':').next() {
                set.insert(base.to_string());
            }
        }
        (true, set, count)
    }

    pub fn installed_models_counted(&self) -> (HashSet<String>, usize) {
        let (_, set, count) = self.detect_with_installed();
        (set, count)
    }
}

#[derive(serde::Deserialize)]
struct DockerModelList {
    data: Vec<DockerEngine>,
}

#[derive(serde::Deserialize)]
struct DockerEngine {
    /// Model ID, e.g. "ai/llama3.1:8B-Q4_K_M"
    id: String,
}

impl ModelProvider for DockerModelRunnerProvider {
    fn name(&self) -> &str {
        "Docker Model Runner"
    }

    fn is_available(&self) -> bool {
        ureq::get(&self.models_url())
            .config()
            .timeout_global(Some(std::time::Duration::from_secs(2)))
            .build()
            .call()
            .is_ok()
    }

    fn installed_models(&self) -> HashSet<String> {
        let (set, _) = self.installed_models_counted();
        set
    }

    fn start_pull(&self, model_tag: &str) -> Result<PullHandle, String> {
        let tag = model_tag.to_string();
        let (tx, rx) = std::sync::mpsc::channel();

        std::thread::spawn(move || {
            let _ = tx.send(PullEvent::Progress {
                status: format!("Pulling {} via docker model pull...", tag),
                percent: None,
            });

            // `--` terminates option parsing so a tag beginning with `-`
            // cannot inject docker CLI flags.
            let result = std::process::Command::new("docker")
                .args(["model", "pull", "--", &tag])
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .output();

            match result {
                Ok(output) if output.status.success() => {
                    let _ = tx.send(PullEvent::Done);
                }
                Ok(output) => {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    let _ = tx.send(PullEvent::Error(format!(
                        "docker model pull failed: {}",
                        stderr.trim()
                    )));
                }
                Err(e) => {
                    let _ = tx.send(PullEvent::Error(format!("Failed to run docker: {e}")));
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
// LM Studio provider
// ---------------------------------------------------------------------------

/// LM Studio — local model server with REST API for model management.
///
/// Exposes an OpenAI-compatible API plus management endpoints at
/// `http://127.0.0.1:1234` by default. Models are downloaded via
/// `POST /api/v1/models/download` and listed via `GET /v1/models`.
pub struct LmStudioProvider {
    base_url: String,
}

fn normalize_lmstudio_host(raw: &str) -> Option<String> {
    let host = raw.trim();
    if host.is_empty() {
        return None;
    }

    if host.starts_with("http://") || host.starts_with("https://") {
        return Some(host.to_string());
    }

    if host.contains("://") {
        return None;
    }

    Some(format!("http://{host}"))
}

impl Default for LmStudioProvider {
    fn default() -> Self {
        let base_url = std::env::var("LMSTUDIO_HOST")
            .ok()
            .and_then(|raw| {
                let normalized = normalize_lmstudio_host(&raw);
                if normalized.is_none() {
                    eprintln!(
                        "Warning: could not parse LMSTUDIO_HOST='{}'. \
                         Expected host:port or http(s)://host:port",
                        raw
                    );
                }
                normalized
            })
            .unwrap_or_else(|| "http://127.0.0.1:1234".to_string());
        Self { base_url }
    }
}

impl LmStudioProvider {
    pub fn new() -> Self {
        Self::default()
    }

    fn models_url(&self) -> String {
        format!("{}/v1/models", self.base_url.trim_end_matches('/'))
    }

    fn download_url(&self) -> String {
        format!(
            "{}/api/v1/models/download",
            self.base_url.trim_end_matches('/')
        )
    }

    /// Single-pass startup probe.
    /// Returns `(available, installed_models, count)`.
    pub fn detect_with_installed(&self) -> (bool, HashSet<String>, usize) {
        let mut set = HashSet::new();
        let Ok(resp) = ureq::get(&self.models_url())
            .config()
            .timeout_global(Some(std::time::Duration::from_millis(800)))
            .build()
            .call()
        else {
            return (false, set, 0);
        };

        let Ok(list) = resp.into_body().read_json::<LmStudioModelList>() else {
            return (true, set, 0);
        };
        let models = list.data;
        let count = models.len();
        for m in models {
            let lower = m.id.to_lowercase();
            set.insert(lower.clone());
            // Also insert the model part after the publisher (e.g. "lmstudio-community/Qwen3-1.7B-MLX-4bit" → "qwen3-1.7b-mlx-4bit")
            if let Some(name) = lower.split('/').next_back()
                && name != lower
            {
                set.insert(name.to_string());
            }
        }
        (true, set, count)
    }

    pub fn installed_models_counted(&self) -> (HashSet<String>, usize) {
        let (_, set, count) = self.detect_with_installed();
        (set, count)
    }
}

#[derive(serde::Deserialize)]
struct LmStudioModelList {
    data: Vec<LmStudioModel>,
}

#[derive(serde::Deserialize)]
struct LmStudioModel {
    /// Model id, e.g. "lmstudio-community/Qwen3-1.7B-MLX-4bit"
    id: String,
}

#[derive(serde::Deserialize)]
struct LmStudioDownloadResponse {
    #[serde(default)]
    #[allow(dead_code)]
    job_id: Option<String>,
    #[serde(default)]
    status: String,
    #[serde(default)]
    #[allow(dead_code)]
    total_size_bytes: Option<u64>,
}

#[derive(serde::Deserialize)]
struct LmStudioDownloadStatus {
    #[serde(default)]
    status: String,
    #[serde(default)]
    progress: Option<f64>,
    #[serde(default)]
    downloaded_bytes: Option<u64>,
    #[serde(default)]
    total_size_bytes: Option<u64>,
}

impl ModelProvider for LmStudioProvider {
    fn name(&self) -> &str {
        "LM Studio"
    }

    fn is_available(&self) -> bool {
        ureq::get(&self.models_url())
            .config()
            .timeout_global(Some(std::time::Duration::from_secs(2)))
            .build()
            .call()
            .is_ok()
    }

    fn installed_models(&self) -> HashSet<String> {
        let (set, _) = self.installed_models_counted();
        set
    }

    fn start_pull(&self, model_tag: &str) -> Result<PullHandle, String> {
        let download_url = self.download_url();
        let models_url = self.models_url();
        let tag = lmstudio_pull_tag(model_tag).unwrap_or_else(|| model_tag.to_string());
        let model_tag_owned = model_tag.to_string();
        let (tx, rx) = std::sync::mpsc::channel();

        let body = serde_json::json!({
            "model": tag,
        });

        std::thread::spawn(move || {
            // LM Studio may stream download progress as newline-delimited JSON
            // from the POST response, or it may acknowledge the request and
            // close the stream while the download proceeds in the background.
            // In the latter case we poll the installed models list to detect
            // eventual completion.
            let resp = ureq::post(&download_url)
                .config()
                .timeout_global(Some(std::time::Duration::from_secs(3600)))
                .build()
                .send_json(&body);

            match resp {
                Ok(resp) => {
                    let reader = std::io::BufReader::new(resp.into_body().into_reader());
                    use std::io::BufRead;

                    let mut saw_completion = false;
                    for line in reader.lines() {
                        let Ok(line) = line else { break };
                        if line.is_empty() {
                            continue;
                        }

                        // Handle SSE "data: {json}" or plain JSON lines
                        let json_str = line.strip_prefix("data: ").unwrap_or(&line);

                        // Try single status object, then first element of an array
                        let status_opt: Option<LmStudioDownloadStatus> =
                            serde_json::from_str(json_str).ok().or_else(|| {
                                serde_json::from_str::<Vec<LmStudioDownloadStatus>>(json_str)
                                    .ok()
                                    .and_then(|v| v.into_iter().next())
                            });

                        // Also try the initial response format (has job_id)
                        if status_opt.is_none() {
                            if let Ok(dl_resp) =
                                serde_json::from_str::<LmStudioDownloadResponse>(json_str)
                            {
                                if dl_resp.status == "already_downloaded" {
                                    let _ = tx.send(PullEvent::Progress {
                                        status: "Already downloaded".to_string(),
                                        percent: Some(100.0),
                                    });
                                    let _ = tx.send(PullEvent::Done);
                                    return;
                                }
                                if dl_resp.status == "failed" {
                                    let _ = tx.send(PullEvent::Error(
                                        "LM Studio download failed".to_string(),
                                    ));
                                    return;
                                }

                                let _ = tx.send(PullEvent::Progress {
                                    status: format!(
                                        "Downloading via LM Studio ({})",
                                        dl_resp.status
                                    ),
                                    percent: Some(0.0),
                                });
                                continue;
                            }
                            continue;
                        }

                        let st = status_opt.unwrap();

                        let percent = st.progress.map(|p| p * 100.0).or_else(|| {
                            match (st.downloaded_bytes, st.total_size_bytes) {
                                (Some(dl), Some(total)) if total > 0 => {
                                    Some(dl as f64 / total as f64 * 100.0)
                                }
                                _ => None,
                            }
                        });

                        if st.status == "completed" || st.status == "already_downloaded" {
                            let _ = tx.send(PullEvent::Progress {
                                status: "Download complete".to_string(),
                                percent: Some(100.0),
                            });
                            let _ = tx.send(PullEvent::Done);
                            saw_completion = true;
                            break;
                        }

                        if st.status == "failed" {
                            let _ =
                                tx.send(PullEvent::Error("LM Studio download failed".to_string()));
                            return;
                        }

                        let _ = tx.send(PullEvent::Progress {
                            status: "Downloading via LM Studio...".to_string(),
                            percent,
                        });
                    }

                    if !saw_completion {
                        // Stream ended without a completion event. The POST
                        // succeeded so LM Studio accepted the request — it
                        // is likely downloading in the background. Poll the
                        // installed models list to detect completion.
                        let candidates = hf_name_to_lmstudio_candidates(&model_tag_owned);
                        let poll_interval = std::time::Duration::from_secs(3);
                        let max_polls = 600; // 30 minutes max

                        let _ = tx.send(PullEvent::Progress {
                            status: "Downloading via LM Studio (tracking)...".to_string(),
                            percent: None,
                        });

                        for poll_num in 0..max_polls {
                            std::thread::sleep(poll_interval);

                            let Ok(resp) = ureq::get(&models_url)
                                .config()
                                .timeout_global(Some(std::time::Duration::from_secs(5)))
                                .build()
                                .call()
                            else {
                                continue;
                            };

                            let Ok(list) = resp.into_body().read_json::<LmStudioModelList>() else {
                                continue;
                            };

                            let installed: HashSet<String> =
                                list.data.into_iter().map(|m| m.id.to_lowercase()).collect();

                            for candidate in &candidates {
                                if installed.contains(candidate.as_str()) {
                                    let _ = tx.send(PullEvent::Progress {
                                        status: "Download complete".to_string(),
                                        percent: Some(100.0),
                                    });
                                    let _ = tx.send(PullEvent::Done);
                                    return;
                                }
                            }

                            // Send periodic progress so the UI knows we're
                            // still tracking the background download.
                            if poll_num % 10 == 9 {
                                let elapsed_secs = (poll_num + 1) as u64 * poll_interval.as_secs();
                                let _ = tx.send(PullEvent::Progress {
                                    status: format!(
                                        "Downloading via LM Studio ({}s elapsed)...",
                                        elapsed_secs
                                    ),
                                    percent: None,
                                });
                            }
                        }

                        let _ =
                            tx.send(PullEvent::Error("LM Studio download timed out".to_string()));
                    }
                }
                Err(e) => {
                    let _ = tx.send(PullEvent::Error(format!("LM Studio download error: {e}")));
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
// LM Studio name-matching helpers
// ---------------------------------------------------------------------------

/// LM Studio uses HuggingFace model names directly. We match against the
/// model's GGUF sources and common naming patterns.
pub fn hf_name_to_lmstudio_candidates(hf_name: &str) -> Vec<String> {
    let repo = hf_name
        .split('/')
        .next_back()
        .unwrap_or(hf_name)
        .to_lowercase();
    let mut candidates = vec![hf_name.to_lowercase()];
    if repo != hf_name.to_lowercase() {
        candidates.push(repo.clone());
    }
    // Strip common suffixes for matching
    let stripped = repo
        .replace("-instruct", "")
        .replace("-chat", "")
        .replace("-hf", "")
        .replace("-it", "");
    if stripped != repo {
        candidates.push(stripped);
    }
    candidates
}

/// Check if any LM Studio candidates for an HF model appear in the installed set.
pub fn is_model_installed_lmstudio(hf_name: &str, installed: &HashSet<String>) -> bool {
    let candidates = hf_name_to_lmstudio_candidates(hf_name);
    candidates.iter().any(|candidate| {
        installed
            .iter()
            .any(|installed_name| installed_name.contains(candidate))
    })
}

/// LM Studio can download any HuggingFace model, so we always return true
/// if the model has GGUF sources (which have HF repo IDs).
pub fn has_lmstudio_mapping(hf_name: &str) -> bool {
    // LM Studio can download from HF directly, so any model with a known
    // GGUF source or a HF name is potentially downloadable.
    !hf_name.is_empty()
}

/// Build a HuggingFace resolve URL for a specific GGUF file.
fn lmstudio_gguf_resolve_url(repo_id: &str, filename: &str) -> String {
    format!(
        "https://huggingface.co/{}/resolve/main/{}",
        repo_id, filename
    )
}

/// Try to find a direct GGUF file URL for an HF model name.
///
/// LM Studio's download endpoint rejects base model repos (which contain
/// safetensors/pytorch weights) and requires a direct link to a `.gguf` file.
/// This function looks up known GGUF repos, lists their files, selects the
/// best quantization that fits in system RAM, and returns a resolve URL.
///
/// Returns `None` if no GGUF files are found or the network is unavailable.
fn lmstudio_find_gguf_url(hf_name: &str) -> Option<String> {
    let mut sys = sysinfo::System::new_all();
    sys.refresh_memory();
    let system_ram_gb = sys.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
    // Leave headroom for OS and overhead
    let budget_gb = system_ram_gb * 0.85;

    // Try known mappings first
    if let Some(repo) = lookup_gguf_repo(hf_name) {
        if let Some(url) = try_gguf_repo(repo, budget_gb) {
            return Some(url);
        }
    }

    // Try heuristic candidates (bartowski/, ggml-org/, TheBloke/)
    for candidate in hf_name_to_gguf_candidates(hf_name) {
        if let Some(url) = try_gguf_repo(&candidate, budget_gb) {
            return Some(url);
        }
    }

    // Try the base repo itself (some repos host GGUF directly)
    if hf_name.contains('/') {
        if let Some(url) = try_gguf_repo(hf_name, budget_gb) {
            return Some(url);
        }
    }

    None
}

/// Try to find a GGUF file in a specific repo.
fn try_gguf_repo(repo_id: &str, budget_gb: f64) -> Option<String> {
    let files = LlamaCppProvider::list_repo_gguf_files(repo_id);
    if files.is_empty() {
        return None;
    }
    let (filename, _) = LlamaCppProvider::select_best_gguf(&files, budget_gb)?;
    Some(lmstudio_gguf_resolve_url(repo_id, &filename))
}

/// Given an HF model name, return the model identifier to use for LM Studio download.
///
/// LM Studio's `/api/v1/models/download` requires a direct link to a `.gguf`
/// file. For HF repo IDs, we first attempt to resolve a GGUF file URL by
/// looking up known GGUF repos and selecting the best quantization. If that
/// fails (network unavailable or no GGUF found), we fall back to wrapping
/// the repo in a base HF URL for backward compatibility.
///
/// Full HTTP(S) URLs are passed through unchanged. Bare short names (no slash)
/// are assumed to be LM Studio first-party catalog entries.
pub fn lmstudio_pull_tag(hf_name: &str) -> Option<String> {
    if hf_name.is_empty() {
        return None;
    }

    // Pass through existing URLs and catalog short names
    if hf_name.starts_with("https://") || hf_name.starts_with("http://") || !hf_name.contains('/') {
        return Some(hf_name.to_string());
    }

    // Try to find a direct GGUF file URL
    if let Some(url) = lmstudio_find_gguf_url(hf_name) {
        return Some(url);
    }

    // Fallback: wrap in base HF URL (preserves pre-fix behavior)
    Some(format!("https://huggingface.co/{}", hf_name))
}

// ---------------------------------------------------------------------------
// vLLM provider
// ---------------------------------------------------------------------------

/// vLLM — high-throughput inference server with an OpenAI-compatible API.
///
/// Exposes `GET /v1/models` to list loaded models at
/// `http://localhost:8000` by default. Override with `VLLM_HOST`.
///
/// vLLM does not have a pull/download endpoint — models are loaded at
/// server start via HuggingFace. The `start_pull` implementation
/// returns an informational error directing users to restart vLLM with
/// the desired model.
pub struct VllmProvider {
    base_url: String,
}

fn normalize_vllm_host(raw: &str) -> Option<String> {
    let host = raw.trim();
    if host.is_empty() {
        return None;
    }

    if host.starts_with("http://") || host.starts_with("https://") {
        return Some(host.to_string());
    }

    if host.contains("://") {
        return None;
    }

    Some(format!("http://{host}"))
}

impl Default for VllmProvider {
    fn default() -> Self {
        let base_url = std::env::var("VLLM_HOST")
            .ok()
            .and_then(|raw| {
                let normalized = normalize_vllm_host(&raw);
                if normalized.is_none() {
                    eprintln!(
                        "Warning: could not parse VLLM_HOST='{}'. \
                         Expected host:port or http(s)://host:port",
                        raw
                    );
                }
                normalized
            })
            .unwrap_or_else(|| "http://localhost:8000".to_string());
        Self { base_url }
    }
}

impl VllmProvider {
    pub fn new() -> Self {
        Self::default()
    }

    fn models_url(&self) -> String {
        format!("{}/v1/models", self.base_url.trim_end_matches('/'))
    }

    /// Single-pass startup probe.
    /// Returns `(available, installed_models, count)`.
    pub fn detect_with_installed(&self) -> (bool, HashSet<String>, usize) {
        let mut set = HashSet::new();
        let Ok(resp) = ureq::get(&self.models_url())
            .config()
            .timeout_global(Some(std::time::Duration::from_millis(800)))
            .build()
            .call()
        else {
            return (false, set, 0);
        };

        let Ok(list) = resp.into_body().read_json::<VllmModelList>() else {
            return (true, set, 0);
        };
        let models = list.data;
        let count = models.len();
        for m in models {
            let lower = m.id.to_lowercase();
            set.insert(lower.clone());
            // Also insert the model part after the publisher
            // e.g. "meta-llama/Llama-3.1-8B-Instruct" → "llama-3.1-8b-instruct"
            if let Some(name) = lower.split('/').next_back()
                && name != lower
            {
                set.insert(name.to_string());
            }
        }
        (true, set, count)
    }

    pub fn installed_models_counted(&self) -> (HashSet<String>, usize) {
        let (_, set, count) = self.detect_with_installed();
        (set, count)
    }
}

#[derive(serde::Deserialize)]
struct VllmModelList {
    data: Vec<VllmModel>,
}

#[derive(serde::Deserialize)]
struct VllmModel {
    /// Model id, e.g. "meta-llama/Llama-3.1-8B-Instruct"
    id: String,
}

impl ModelProvider for VllmProvider {
    fn name(&self) -> &str {
        "vLLM"
    }

    fn is_available(&self) -> bool {
        ureq::get(&self.models_url())
            .config()
            .timeout_global(Some(std::time::Duration::from_secs(2)))
            .build()
            .call()
            .is_ok()
    }

    fn installed_models(&self) -> HashSet<String> {
        let (set, _) = self.installed_models_counted();
        set
    }

    fn start_pull(&self, _model_tag: &str) -> Result<PullHandle, String> {
        Err("vLLM does not support downloading models at runtime. \
             Restart the vLLM server with the desired model \
             (e.g. `vllm serve <model>`)."
            .to_string())
    }
}

// ---------------------------------------------------------------------------
// vLLM name-matching helpers
// ---------------------------------------------------------------------------

/// vLLM uses HuggingFace model names directly. We match against the
/// model's full HF name and common naming patterns.
pub fn hf_name_to_vllm_candidates(hf_name: &str) -> Vec<String> {
    let repo = hf_name
        .split('/')
        .next_back()
        .unwrap_or(hf_name)
        .to_lowercase();
    let mut candidates = vec![hf_name.to_lowercase()];
    if repo != hf_name.to_lowercase() {
        candidates.push(repo.clone());
    }
    // Strip common suffixes for matching
    let stripped = repo
        .replace("-instruct", "")
        .replace("-chat", "")
        .replace("-hf", "")
        .replace("-it", "");
    if stripped != repo {
        candidates.push(stripped);
    }
    candidates
}

/// Check if any vLLM candidates for an HF model appear in the installed set.
pub fn is_model_installed_vllm(hf_name: &str, installed: &HashSet<String>) -> bool {
    let candidates = hf_name_to_vllm_candidates(hf_name);
    candidates.iter().any(|candidate| {
        installed
            .iter()
            .any(|installed_name| installed_name.contains(candidate))
    })
}

/// vLLM can serve any HuggingFace model, so we always return true.
pub fn has_vllm_mapping(hf_name: &str) -> bool {
    !hf_name.is_empty()
}

/// Given an HF model name, return the model identifier to use for vLLM.
/// vLLM accepts HF model names directly.
pub fn vllm_pull_tag(hf_name: &str) -> Option<String> {
    if hf_name.is_empty() {
        return None;
    }
    Some(hf_name.to_string())
}

// ---------------------------------------------------------------------------
// Docker Model Runner name-matching helpers
// ---------------------------------------------------------------------------

/// Embedded catalog of HF models confirmed to exist in Docker Hub's ai/ namespace.
/// Generated by `scripts/scrape_docker_models.py` and refreshed alongside the model DB.
const DOCKER_MODELS_JSON: &str = include_str!("../data/docker_models.json");

#[derive(serde::Deserialize)]
struct DockerModelCatalog {
    models: Vec<DockerModelEntry>,
}

#[derive(serde::Deserialize)]
struct DockerModelEntry {
    hf_name: String,
    docker_tag: String,
}

/// Lazily parsed Docker Model Runner catalog.
fn docker_mr_catalog() -> &'static [(String, String)] {
    use std::sync::OnceLock;
    static CATALOG: OnceLock<Vec<(String, String)>> = OnceLock::new();
    CATALOG.get_or_init(|| {
        let Ok(catalog) = serde_json::from_str::<DockerModelCatalog>(DOCKER_MODELS_JSON) else {
            return Vec::new();
        };
        catalog
            .models
            .into_iter()
            .map(|e| (e.hf_name.to_lowercase(), e.docker_tag))
            .collect()
    })
}

/// Returns `true` if this HF model has a confirmed Docker Model Runner image.
pub fn has_docker_mr_mapping(hf_name: &str) -> bool {
    docker_mr_pull_tag(hf_name).is_some()
}

/// Given an HF model name, return the Docker Model Runner tag to use for pulling.
/// Returns `None` if the model has no confirmed Docker image.
pub fn docker_mr_pull_tag(hf_name: &str) -> Option<String> {
    let lower = hf_name.to_lowercase();
    docker_mr_catalog()
        .iter()
        .find(|(name, _)| *name == lower)
        .map(|(_, tag)| tag.clone())
}

/// Docker Model Runner uses the Ollama naming convention (e.g. "ai/llama3.1:8b").
/// We generate candidates from the confirmed catalog, plus base-name variants for
/// matching against locally installed models.
pub fn hf_name_to_docker_mr_candidates(hf_name: &str) -> Vec<String> {
    let Some(tag) = docker_mr_pull_tag(hf_name) else {
        return Vec::new();
    };
    let mut candidates = vec![tag.clone()];
    // Also add without "ai/" prefix for matching installed models
    if let Some(stripped) = tag.strip_prefix("ai/") {
        candidates.push(stripped.to_string());
    }
    // Add base repo name (without size tag) e.g. "ai/llama3.1"
    if let Some(base) = tag.split(':').next() {
        candidates.push(base.to_string());
    }
    candidates
}

/// Check if any of the Docker Model Runner candidates for an HF model
/// appear in the installed set.
pub fn is_model_installed_docker_mr(hf_name: &str, installed: &HashSet<String>) -> bool {
    let candidates = hf_name_to_docker_mr_candidates(hf_name);
    candidates.iter().any(|candidate| {
        installed
            .iter()
            .any(|installed_name| docker_mr_installed_matches(installed_name, candidate))
    })
}

fn docker_mr_installed_matches(installed_name: &str, candidate: &str) -> bool {
    if installed_name == candidate {
        return true;
    }
    // Allow variant tags, e.g. candidate "ai/llama3.1:8b" matching
    // installed "ai/llama3.1:8b-q4_k_m"
    if candidate.contains(':') {
        return installed_name.starts_with(&format!("{candidate}-"));
    }
    false
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

fn push_unique_candidate(candidates: &mut Vec<String>, candidate: String) {
    if !candidate.is_empty() && !candidates.iter().any(|c| c == &candidate) {
        candidates.push(candidate);
    }
}

fn strip_trailing_quant_suffix(name: &str) -> String {
    for suffix in ["-4bit", "-6bit", "-8bit"] {
        if let Some(stripped) = name.strip_suffix(suffix) {
            return stripped.to_string();
        }
    }
    name.to_string()
}

fn normalize_mlx_repo_base(repo_lower: &str) -> String {
    let without_quant = strip_trailing_quant_suffix(repo_lower);

    without_quant
        .strip_suffix("-mlx")
        .unwrap_or(&without_quant)
        .trim_matches('-')
        .to_string()
}

fn strip_trailing_common_model_suffixes(name: &str) -> String {
    let mut out = name.to_string();
    loop {
        let mut changed = false;
        for suffix in ["-instruct", "-chat", "-hf", "-it", "-base"] {
            if let Some(stripped) = out.strip_suffix(suffix) {
                out = stripped.trim_end_matches('-').to_string();
                changed = true;
                break;
            }
        }
        if !changed {
            break;
        }
    }
    out
}

fn explicit_mlx_repo_id(hf_name: &str) -> Option<String> {
    if hf_name.matches('/').count() != 1 {
        return None;
    }
    let mut parts = hf_name.splitn(2, '/');
    let owner = parts.next()?.trim();
    let repo = parts.next()?.trim();
    if owner.is_empty() || repo.is_empty() || !is_likely_mlx_repo(owner, repo) {
        return None;
    }
    Some(format!("{}/{}", owner.to_lowercase(), repo.to_lowercase()))
}

/// Map a HuggingFace model name to mlx-community repo name candidates.
/// Pattern: mlx-community/{RepoName}-{quant}bit
pub fn hf_name_to_mlx_candidates(hf_name: &str) -> Vec<String> {
    let mut candidates = Vec::new();

    if let Some(repo_id) = explicit_mlx_repo_id(hf_name) {
        push_unique_candidate(&mut candidates, repo_id.clone());
        if let Some(repo_name) = repo_id.split('/').next_back() {
            push_unique_candidate(&mut candidates, repo_name.to_string());
        }
    }

    let repo = hf_name.split('/').next_back().unwrap_or(hf_name);
    let repo_lower = repo.to_lowercase();
    push_unique_candidate(&mut candidates, repo_lower.clone());

    let normalized_repo = normalize_mlx_repo_base(&repo_lower);

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
        ("Qwen3-1.7B", "Qwen3-1.7B"),
        ("Qwen3-0.6B", "Qwen3-0.6B"),
        ("Qwen3-30B-A3B", "Qwen3-30B-A3B"),
        ("Qwen3-235B-A22B", "Qwen3-235B-A22B"),
        // Qwen3.5
        ("Qwen3.5-0.6B", "Qwen3.5-0.6B"),
        ("Qwen3.5-1.7B", "Qwen3.5-1.7B"),
        ("Qwen3.5-4B", "Qwen3.5-4B"),
        ("Qwen3.5-8B", "Qwen3.5-8B"),
        ("Qwen3.5-9B", "Qwen3.5-9B"),
        ("Qwen3.5-14B", "Qwen3.5-14B"),
        ("Qwen3.5-27B", "Qwen3.5-27B"),
        ("Qwen3.5-32B", "Qwen3.5-32B"),
        ("Qwen3.5-35B-A3B", "Qwen3.5-35B-A3B"),
        ("Qwen3.5-72B", "Qwen3.5-72B"),
        ("Qwen3.5-122B-A10B", "Qwen3.5-122B-A10B"),
        ("Qwen3.5-397B-A17B", "Qwen3.5-397B-A17B"),
        // Mistral
        ("Mistral-7B-Instruct-v0.3", "Mistral-7B-Instruct-v0.3"),
        (
            "Mistral-Small-24B-Instruct-2501",
            "Mistral-Small-24B-Instruct-2501",
        ),
        ("Mixtral-8x7B-Instruct-v0.1", "Mixtral-8x7B-Instruct-v0.1"),
        (
            "Mistral-Small-3.1-24B-Instruct-2503",
            "Mistral-Small-3.1-24B-Instruct-2503",
        ),
        ("Ministral-8B-Instruct-2410", "Ministral-8B-Instruct-2410"),
        ("Mistral-Nemo-Instruct-2407", "Mistral-Nemo-Instruct-2407"),
        // DeepSeek
        (
            "DeepSeek-R1-Distill-Qwen-32B",
            "DeepSeek-R1-Distill-Qwen-32B",
        ),
        ("DeepSeek-R1-Distill-Qwen-7B", "DeepSeek-R1-Distill-Qwen-7B"),
        (
            "DeepSeek-R1-Distill-Qwen-14B",
            "DeepSeek-R1-Distill-Qwen-14B",
        ),
        (
            "DeepSeek-R1-Distill-Llama-8B",
            "DeepSeek-R1-Distill-Llama-8B",
        ),
        (
            "DeepSeek-R1-Distill-Llama-70B",
            "DeepSeek-R1-Distill-Llama-70B",
        ),
        // Gemma
        ("gemma-3-12b-it", "gemma-3-12b-it"),
        ("gemma-2-27b-it", "gemma-2-27b-it"),
        ("gemma-2-9b-it", "gemma-2-9b-it"),
        ("gemma-2-2b-it", "gemma-2-2b-it"),
        ("gemma-3-1b-it", "gemma-3-1b-it"),
        ("gemma-3-4b-it", "gemma-3-4b-it"),
        ("gemma-3-27b-it", "gemma-3-27b-it"),
        ("gemma-3n-E4B-it", "gemma-3n-E4B-it"),
        ("gemma-3n-E2B-it", "gemma-3n-E2B-it"),
        // Phi
        ("Phi-4", "Phi-4"),
        ("Phi-3.5-mini-instruct", "Phi-3.5-mini-instruct"),
        ("Phi-3-mini-4k-instruct", "Phi-3-mini-4k-instruct"),
        ("Phi-4-mini-instruct", "Phi-4-mini-instruct"),
        ("Phi-4-reasoning", "Phi-4-reasoning"),
        ("Phi-4-mini-reasoning", "Phi-4-mini-reasoning"),
        // Llama 4
        (
            "Llama-4-Scout-17B-16E-Instruct",
            "Llama-4-Scout-17B-16E-Instruct",
        ),
        (
            "Llama-4-Maverick-17B-128E-Instruct",
            "Llama-4-Maverick-17B-128E-Instruct",
        ),
    ];

    for &(hf_suffix, mlx_base) in mappings {
        let mapped_suffix = hf_suffix.to_lowercase();
        if repo_lower == mapped_suffix || normalized_repo == mapped_suffix {
            let base_lower = mlx_base.to_lowercase();
            push_unique_candidate(&mut candidates, format!("{}-4bit", base_lower));
            push_unique_candidate(&mut candidates, format!("{}-8bit", base_lower));
            push_unique_candidate(&mut candidates, base_lower);
            return candidates;
        }
    }

    // Fallback heuristic: normalize explicit MLX names and try common variants.
    if !normalized_repo.is_empty() {
        push_unique_candidate(&mut candidates, format!("{}-4bit", normalized_repo));
        push_unique_candidate(&mut candidates, format!("{}-8bit", normalized_repo));
        // Some mlx-community repos use a -MLX- infix (e.g. Model-MLX-4bit)
        push_unique_candidate(&mut candidates, format!("{}-mlx-4bit", normalized_repo));
        push_unique_candidate(&mut candidates, format!("{}-mlx-8bit", normalized_repo));
        push_unique_candidate(&mut candidates, normalized_repo.clone());
    }

    let stripped = strip_trailing_common_model_suffixes(&normalized_repo);
    if !stripped.is_empty() && stripped != normalized_repo {
        push_unique_candidate(&mut candidates, format!("{}-4bit", stripped));
        push_unique_candidate(&mut candidates, format!("{}-8bit", stripped));
        push_unique_candidate(&mut candidates, format!("{}-mlx-4bit", stripped));
        push_unique_candidate(&mut candidates, format!("{}-mlx-8bit", stripped));
        push_unique_candidate(&mut candidates, stripped);
    }

    candidates
}

/// Check if any MLX candidates for an HF model appear in the installed set.
pub fn is_model_installed_mlx(hf_name: &str, installed: &HashSet<String>) -> bool {
    // Quick check: installed set may contain the full HF name (lowercased)
    if installed.contains(&hf_name.to_lowercase()) {
        return true;
    }

    let candidates = hf_name_to_mlx_candidates(hf_name);
    candidates.iter().any(|c| installed.contains(c))
}

/// Given an HF model name, return the best MLX tag to use for pulling.
pub fn mlx_pull_tag(hf_name: &str) -> String {
    if let Some(repo_id) = explicit_mlx_repo_id(hf_name) {
        return repo_id;
    }
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
    // Qwen 3.5
    ("qwen3.5-27b", "qwen3.5"),
    ("qwen3.5-35b-a3b", "qwen3.5:35b"),
    ("qwen3.5-122b-a10b", "qwen3.5:122b"),
    // Qwen3-Coder-Next
    ("qwen3-coder-next", "qwen3-coder-next"),
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
    // Google Gemma 3n
    ("gemma-3n-e4b-it", "gemma3n:e4b"),
    ("gemma-3n-e2b-it", "gemma3n:e2b"),
    // Microsoft Phi-4 reasoning
    ("phi-4-reasoning", "phi4-reasoning"),
    ("phi-4-mini-reasoning", "phi4-mini-reasoning"),
    // DeepSeek V3.2 Speciale (no local Ollama tag yet, maps to v3)
    ("deepseek-v3.2-speciale", "deepseek-v3"),
    // Liquid AI LFM2
    ("lfm2-350m", "lfm2:350m"),
    ("lfm2-700m", "lfm2:700m"),
    ("lfm2-1.2b", "lfm2:1.2b"),
    ("lfm2-2.6b", "lfm2:2.6b"),
    ("lfm2-2.6b-exp", "lfm2:2.6b"),
    ("lfm2-8b-a1b", "lfm2:8b-a1b"),
    ("lfm2-24b-a2b", "lfm2:24b"),
    // Liquid AI LFM2.5
    ("lfm2.5-1.2b-instruct", "lfm2.5:1.2b"),
    ("lfm2.5-1.2b-thinking", "lfm2.5-thinking:1.2b"),
];

/// Split a lowercased model name into (family_name, size_tag) by finding
/// the rightmost segment that looks like a parameter size (e.g. "7b", "70b",
/// "30b-a3b" for MoE).  Returns `None` if no size-like segment is found.
///
/// Examples:
///   "qwen2.5-coder-14b"       → Some(("qwen2.5-coder", "14b"))
///   "deepseek-r1-distill-qwen-32b" → Some(("deepseek-r1-distill-qwen", "32b"))
///   "qwen3-coder-30b-a3b"     → Some(("qwen3-coder", "30b-a3b"))
///   "phi-4"                    → None (no "b" suffix — "4" isn't a size tag)
fn split_name_and_size(name: &str) -> Option<(&str, &str)> {
    // Walk segments from the right looking for one that matches a size
    // pattern like "7b", "70b", "1.7b", "30b-a3b" (MoE active params).
    let segments: Vec<&str> = name.split('-').collect();
    for i in (0..segments.len()).rev() {
        let seg = segments[i];
        // Check for a segment ending in 'b' with digits (e.g. "7b", "70b", "1.7b")
        if seg.ends_with('b') && seg.len() > 1 {
            let before_b = &seg[..seg.len() - 1];
            if before_b.chars().all(|c| c.is_ascii_digit() || c == '.') {
                // Include any trailing MoE segment like "-a3b"
                let size_start = segments[..i]
                    .iter()
                    .map(|s| s.len() + 1) // +1 for the '-'
                    .sum::<usize>();
                if size_start == 0 || size_start > name.len() {
                    return None;
                }
                let family = &name[..size_start - 1]; // trim trailing '-'
                let size = &name[size_start..];
                if !family.is_empty() && !size.is_empty() {
                    return Some((family, size));
                }
            }
        }
    }
    None
}

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
/// Tries the authoritative mapping table first, then falls back to heuristic
/// candidate generation so models without explicit mappings can still be
/// detected as installed.
pub fn hf_name_to_ollama_candidates(hf_name: &str) -> Vec<String> {
    if let Some(tag) = lookup_ollama_tag(hf_name) {
        return vec![tag.to_string()];
    }

    // Fallback: generate candidates from the HF repo name convention.
    // e.g. "Qwen/Qwen3-Coder-30B-A3B-Instruct" → ["qwen3-coder-30b-a3b", "qwen3-coder:30b-a3b", ...]
    let repo = hf_name
        .split('/')
        .next_back()
        .unwrap_or(hf_name)
        .to_lowercase();

    let base = strip_trailing_common_model_suffixes(&repo);

    let mut candidates = Vec::new();

    // Try to split off the size tag (e.g. "qwen3-coder-30b-a3b" → ("qwen3-coder", "30b-a3b"))
    // Ollama uses "name:size" format, so we look for a size-like segment.
    if let Some((name, size)) = split_name_and_size(&base) {
        // "name:size" is the primary Ollama format
        candidates.push(format!("{}:{}", name, size));
        // Also try bare family name (Ollama inserts both into the installed set)
        candidates.push(name.to_string());
    }

    // Also try the full lowered name and stripped name as-is
    candidates.push(base.clone());
    if base != repo {
        candidates.push(repo);
    }

    candidates.dedup();
    candidates
}

/// Returns `true` if this HF model has a known Ollama registry entry
/// and can be pulled.
pub fn has_ollama_mapping(hf_name: &str) -> bool {
    lookup_ollama_tag(hf_name).is_some()
}

fn ollama_installed_matches_candidate(installed_name: &str, candidate: &str) -> bool {
    if installed_name == candidate {
        return true;
    }

    // Allow variant tags reported by `ollama list`, e.g.
    // candidate: "qwen2.5-coder:7b"
    // installed: "qwen2.5-coder:7b-instruct-q4_K_M"
    if candidate.contains(':') {
        return installed_name.starts_with(&format!("{candidate}-"));
    }

    false
}

/// Check if any of the Ollama candidates for an HF model appear in the
/// installed set.
pub fn is_model_installed(hf_name: &str, installed: &HashSet<String>) -> bool {
    // Quick check: the installed set may contain the full HF name (lowercased)
    // from providers that report it verbatim (e.g. MLX server, /api/v1/installed).
    if installed.contains(&hf_name.to_lowercase()) {
        return true;
    }

    let candidates = hf_name_to_ollama_candidates(hf_name);
    candidates.iter().any(|candidate| {
        installed
            .iter()
            .any(|installed_name| ollama_installed_matches_candidate(installed_name, candidate))
    })
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
    fn test_hf_name_to_mlx_candidates_qwen35() {
        let candidates = hf_name_to_mlx_candidates("Qwen/Qwen3.5-9B");
        assert!(candidates.iter().any(|c| c == "qwen3.5-9b-4bit"));
        assert!(candidates.iter().any(|c| c == "qwen3.5-9b-8bit"));
    }

    #[test]
    fn test_hf_name_to_mlx_candidates_llama4() {
        let candidates = hf_name_to_mlx_candidates("meta-llama/Llama-4-Scout-17B-16E-Instruct");
        assert!(candidates.iter().any(|c| c.contains("llama-4-scout")));
        assert!(candidates.iter().any(|c| c.ends_with("-4bit")));
    }

    #[test]
    fn test_hf_name_to_mlx_candidates_gemma3() {
        let candidates = hf_name_to_mlx_candidates("google/gemma-3-27b-it");
        assert!(candidates.iter().any(|c| c == "gemma-3-27b-it-4bit"));
        assert!(candidates.iter().any(|c| c == "gemma-3-27b-it-8bit"));
    }

    #[test]
    fn test_hf_name_to_mlx_fallback_generates_mlx_infix_candidates() {
        // For models not in the explicit mapping, the fallback should also
        // generate candidates with the -mlx- infix pattern
        let candidates = hf_name_to_mlx_candidates("SomeOrg/SomeNewModel-7B");
        assert!(candidates.iter().any(|c| c == "somenewmodel-7b-mlx-4bit"));
        assert!(candidates.iter().any(|c| c == "somenewmodel-7b-mlx-8bit"));
    }

    #[test]
    fn test_hf_name_to_mlx_candidates_normalizes_explicit_mlx_repo() {
        let candidates =
            hf_name_to_mlx_candidates("lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-8bit");

        assert!(
            candidates
                .contains(&"lmstudio-community/qwen3-coder-30b-a3b-instruct-mlx-8bit".to_string())
        );
        assert!(candidates.contains(&"qwen3-coder-30b-a3b-instruct-4bit".to_string()));
        assert!(candidates.contains(&"qwen3-coder-30b-a3b-instruct-8bit".to_string()));
        assert!(!candidates.iter().any(|c| c.contains("-8bit-4bit")));
        assert!(!candidates.iter().any(|c| c.contains("-8bit-8bit")));
    }

    #[test]
    fn test_mlx_pull_tag_prefers_explicit_repo_id() {
        let tag = mlx_pull_tag("lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-8bit");
        assert_eq!(
            tag,
            "lmstudio-community/qwen3-coder-30b-a3b-instruct-mlx-8bit"
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
    fn test_lmstudio_pull_tag_resolves_gguf_url() {
        // HF repo IDs should resolve to a direct GGUF file URL via known
        // mappings or heuristic repo lookups. The exact URL depends on
        // available files and system RAM, so we only assert the shape.
        let tag = lmstudio_pull_tag("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct").unwrap();
        // Should either be a GGUF resolve URL or fall back to base repo URL
        assert!(
            tag.starts_with("https://huggingface.co/")
                && (tag.contains("/resolve/main/") && tag.ends_with(".gguf"))
                || !tag.contains("/resolve/main/")
        );
    }

    #[test]
    fn test_lmstudio_pull_tag_passes_through_full_url() {
        let url = "https://huggingface.co/lmstudio-community/deepseek-coder-v2-lite-instruct-gguf";
        assert_eq!(lmstudio_pull_tag(url).unwrap(), url);

        let http = "http://example.com/some/model";
        assert_eq!(lmstudio_pull_tag(http).unwrap(), http);
    }

    #[test]
    fn test_lmstudio_pull_tag_leaves_catalog_short_name_unchanged() {
        // No slash → assumed to be an LM Studio first-party catalog entry.
        assert_eq!(lmstudio_pull_tag("llama-3.1-8b").unwrap(), "llama-3.1-8b");
    }

    #[test]
    fn test_lmstudio_pull_tag_empty_returns_none() {
        assert!(lmstudio_pull_tag("").is_none());
    }

    #[test]
    fn test_lmstudio_pull_tag_is_idempotent() {
        // A resolved URL must be safe to apply twice — start_pull and the TUI
        // both route through the same resolver.
        let once = lmstudio_pull_tag("Qwen/Qwen2.5-7B-Instruct").unwrap();
        let twice = lmstudio_pull_tag(&once).unwrap();
        assert_eq!(once, twice);
    }

    #[test]
    fn test_lmstudio_gguf_resolve_url_format() {
        let url = lmstudio_gguf_resolve_url(
            "lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
            "DeepSeek-Coder-V2-Lite-Instruct-Q6_K.gguf",
        );
        assert_eq!(
            url,
            "https://huggingface.co/lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF/resolve/main/DeepSeek-Coder-V2-Lite-Instruct-Q6_K.gguf"
        );
    }

    #[test]
    fn test_hf_name_to_lmstudio_candidates_full_repo() {
        let candidates = hf_name_to_lmstudio_candidates("lmstudio-community/Qwen3-1.7B-GGUF");
        assert!(candidates.contains(&"lmstudio-community/qwen3-1.7b-gguf".to_string()));
        assert!(candidates.contains(&"qwen3-1.7b-gguf".to_string()));
    }

    #[test]
    fn test_hf_name_to_lmstudio_candidates_strips_suffixes() {
        let candidates = hf_name_to_lmstudio_candidates("meta-llama/Llama-3-8B-Instruct");
        assert!(candidates.contains(&"meta-llama/llama-3-8b-instruct".to_string()));
        assert!(candidates.contains(&"llama-3-8b-instruct".to_string()));
        // Stripped variant (without -instruct)
        assert!(candidates.contains(&"llama-3-8b".to_string()));
    }

    #[test]
    fn test_hf_name_to_lmstudio_candidates_bare_name() {
        let candidates = hf_name_to_lmstudio_candidates("qwen3");
        assert!(candidates.contains(&"qwen3".to_string()));
        // No slash, so repo == full name — no duplicate
        assert_eq!(candidates.len(), 1);
    }

    #[test]
    fn test_is_model_installed_mlx_with_owner_prefixed_repo_id() {
        let mut installed = HashSet::new();
        installed.insert("lmstudio-community/qwen3-coder-30b-a3b-instruct-mlx-8bit".to_string());

        assert!(is_model_installed_mlx(
            "lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-8bit",
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
    fn test_installed_variant_suffix_matches_ollama_candidate() {
        // Real-world `ollama list` may include variant suffixes that still map
        // to the canonical pull tag in OLLAMA_MAPPINGS.
        let mut installed = HashSet::new();
        installed.insert("qwen2.5-coder:7b-instruct".to_string());

        assert!(is_model_installed(
            "Qwen/Qwen2.5-Coder-7B-Instruct",
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

    #[test]
    fn test_normalize_ollama_host_with_scheme() {
        assert_eq!(
            normalize_ollama_host("https://ollama.example.com:11434"),
            Some("https://ollama.example.com:11434".to_string())
        );
    }

    #[test]
    fn test_normalize_ollama_host_without_scheme() {
        assert_eq!(
            normalize_ollama_host("ollama.example.com:11434"),
            Some("http://ollama.example.com:11434".to_string())
        );
    }

    #[test]
    fn test_normalize_ollama_host_rejects_unsupported_scheme() {
        assert_eq!(
            normalize_ollama_host("ftp://ollama.example.com:11434"),
            None
        );
    }

    #[test]
    fn test_is_wildcard_bind_address_ipv4() {
        assert!(is_wildcard_bind_address("0.0.0.0"));
        assert!(is_wildcard_bind_address("0.0.0.0:11434"));
        assert!(is_wildcard_bind_address("http://0.0.0.0"));
        assert!(is_wildcard_bind_address("http://0.0.0.0:11434"));
        assert!(is_wildcard_bind_address("https://0.0.0.0:11434"));
        assert!(is_wildcard_bind_address("http://0.0.0.0:11434/api/tags"));
    }

    #[test]
    fn test_is_wildcard_bind_address_ipv6() {
        assert!(is_wildcard_bind_address("[::]"));
        assert!(is_wildcard_bind_address("[::]:11434"));
        assert!(is_wildcard_bind_address("http://[::]:11434"));
        assert!(is_wildcard_bind_address("http://[0:0:0:0:0:0:0:0]:11434"));
    }

    #[test]
    fn test_is_wildcard_bind_address_rejects_routable_hosts() {
        assert!(!is_wildcard_bind_address("localhost"));
        assert!(!is_wildcard_bind_address("http://localhost:11434"));
        assert!(!is_wildcard_bind_address("127.0.0.1"));
        assert!(!is_wildcard_bind_address("http://127.0.0.1:11434"));
        assert!(!is_wildcard_bind_address("http://[::1]:11434"));
        assert!(!is_wildcard_bind_address("http://ollama.example.com:11434"));
        // Hostnames or IPs that merely contain "0.0.0.0" as a substring must not match.
        assert!(!is_wildcard_bind_address("http://10.0.0.0.example.com"));
        assert!(!is_wildcard_bind_address("http://10.0.0.1:11434"));
    }

    #[test]
    fn test_validate_gguf_filename_valid() {
        assert!(validate_gguf_filename("Llama-3.1-8B-Q4_K_M.gguf").is_ok());
        assert!(validate_gguf_filename("model.gguf").is_ok());
    }

    #[test]
    fn test_validate_gguf_filename_traversal() {
        assert!(validate_gguf_filename("../../outside.gguf").is_err());
        assert!(validate_gguf_filename("../evil.gguf").is_err());
        assert!(validate_gguf_filename("foo/../bar.gguf").is_err());
    }

    #[test]
    fn test_validate_gguf_filename_absolute() {
        assert!(validate_gguf_filename("/etc/passwd").is_err());
        assert!(validate_gguf_filename("/tmp/model.gguf").is_err());
    }

    #[test]
    fn test_validate_gguf_filename_bad_extension() {
        assert!(validate_gguf_filename("malware.exe").is_err());
        assert!(validate_gguf_filename("script.sh").is_err());
        assert!(validate_gguf_filename("./model.guuf").is_err());
    }

    #[test]
    fn test_validate_gguf_filename_empty() {
        assert!(validate_gguf_filename("").is_err());
    }

    #[test]
    fn test_validate_gguf_filename_subdirectory() {
        assert!(validate_gguf_filename("subdir/model.gguf").is_err());
    }

    #[test]
    fn test_validate_gguf_filename_rejects_non_basename_forms() {
        assert!(validate_gguf_filename("./model.gguf").is_err());
        assert!(validate_gguf_filename("model.gguf/").is_err());
        assert!(validate_gguf_filename(".\\model.gguf").is_err());
        assert!(validate_gguf_filename("C:/models/model.gguf").is_err());
        assert!(validate_gguf_filename("C:\\models\\model.gguf").is_err());
    }

    // ── validate_gguf_repo_path ────────────────────────────────────

    #[test]
    fn test_validate_gguf_repo_path_valid() {
        assert!(validate_gguf_repo_path("model.gguf").is_ok());
        assert!(validate_gguf_repo_path("Q4_K_M/model.gguf").is_ok());
        assert!(validate_gguf_repo_path("deep/nested/model.gguf").is_ok());
    }

    #[test]
    fn test_validate_gguf_repo_path_rejects_traversal() {
        assert!(validate_gguf_repo_path("../escape.gguf").is_err());
        assert!(validate_gguf_repo_path("foo/../bar.gguf").is_err());
        assert!(validate_gguf_repo_path("./model.gguf").is_err());
    }

    #[test]
    fn test_validate_gguf_repo_path_rejects_absolute() {
        assert!(validate_gguf_repo_path("/etc/passwd").is_err());
        assert!(validate_gguf_repo_path("/tmp/model.gguf").is_err());
    }

    #[test]
    fn test_validate_gguf_repo_path_rejects_backslash() {
        assert!(validate_gguf_repo_path("dir\\model.gguf").is_err());
        assert!(validate_gguf_repo_path("C:\\models\\model.gguf").is_err());
    }

    #[test]
    fn test_validate_gguf_repo_path_rejects_non_gguf() {
        assert!(validate_gguf_repo_path("malware.exe").is_err());
        assert!(validate_gguf_repo_path("subdir/readme.md").is_err());
    }

    #[test]
    fn test_validate_gguf_repo_path_rejects_empty() {
        assert!(validate_gguf_repo_path("").is_err());
    }

    #[test]
    fn test_parse_repo_gguf_entries_filters_unsafe_paths() {
        let entries = vec![
            serde_json::json!({"path": "good.gguf", "size": 123u64}),
            serde_json::json!({"path": "../escape.gguf", "size": 456u64}),
            serde_json::json!({"path": "nested/model.gguf", "size": 789u64}),
            serde_json::json!({"path": "./model.gguf", "size": 99u64}),
            serde_json::json!({"path": "readme.md", "size": 12u64}),
        ];

        let files = parse_repo_gguf_entries(entries);
        assert_eq!(
            files,
            vec![
                ("good.gguf".to_string(), 123u64),
                ("nested/model.gguf".to_string(), 789u64),
            ]
        );
    }

    // ────────────────────────────────────────────────────────────────────
    // GGUF candidate generation tests
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn test_hf_name_to_gguf_candidates_generates_common_patterns() {
        // Use a model without a hardcoded mapping to test heuristic generation
        let candidates = hf_name_to_gguf_candidates("SomeOrg/Cool-Model-7B");
        assert!(
            candidates
                .iter()
                .any(|c| c == "bartowski/Cool-Model-7B-GGUF"),
            "Should generate bartowski candidate, got: {:?}",
            candidates
        );
        assert!(
            candidates
                .iter()
                .any(|c| c == "ggml-org/Cool-Model-7B-GGUF"),
            "Should generate ggml-org candidate, got: {:?}",
            candidates
        );
        assert!(
            candidates
                .iter()
                .any(|c| c == "TheBloke/Cool-Model-7B-GGUF"),
            "Should generate TheBloke candidate, got: {:?}",
            candidates
        );
    }

    #[test]
    fn test_hf_name_to_gguf_candidates_strips_owner() {
        // Should use the model name part, not the full "owner/name"
        let candidates = hf_name_to_gguf_candidates("Qwen/Qwen2.5-7B-Instruct");
        for c in &candidates {
            assert!(
                !c.contains("Qwen/Qwen"),
                "Candidate should not contain original owner prefix: {}",
                c
            );
        }
    }

    #[test]
    fn test_lookup_gguf_repo_known_mappings() {
        // Models with hardcoded mappings should be found
        assert!(lookup_gguf_repo("meta-llama/Llama-3.1-8B-Instruct").is_some());
        assert!(lookup_gguf_repo("deepseek-r1").is_some());
    }

    #[test]
    fn test_lookup_gguf_repo_unknown_returns_none() {
        assert!(lookup_gguf_repo("totally-unknown/model-xyz").is_none());
    }

    #[test]
    fn test_has_gguf_mapping_matches_known_models() {
        assert!(has_gguf_mapping("meta-llama/Llama-3.1-8B-Instruct"));
        assert!(!has_gguf_mapping("some-random/UnknownModel"));
    }

    #[test]
    fn test_gguf_candidates_fallback_covers_major_providers() {
        // For a model without a hardcoded mapping, candidates should cover
        // the major GGUF providers
        let candidates = hf_name_to_gguf_candidates("SomeOrg/NewModel-7B");
        assert!(candidates.iter().any(|c| c.starts_with("bartowski/")));
        assert!(candidates.iter().any(|c| c.starts_with("ggml-org/")));
        assert!(candidates.iter().any(|c| c.starts_with("TheBloke/")));
        assert!(candidates.iter().all(|c| c.ends_with("-GGUF")));
    }

    #[test]
    fn test_gguf_candidates_known_mapping_returns_single() {
        // Models with a hardcoded mapping should return just that repo
        let candidates = hf_name_to_gguf_candidates("meta-llama/Llama-3.1-8B-Instruct");
        assert_eq!(candidates.len(), 1);
        assert!(candidates[0].contains("GGUF"));
    }

    // ── select_best_gguf ─────────────────────────────────────────────

    #[test]
    fn test_select_best_gguf_prefers_higher_quality() {
        let files = vec![
            ("model-Q2_K.gguf".to_string(), 2_000_000_000u64),
            ("model-Q4_K_M.gguf".to_string(), 4_000_000_000u64),
            ("model-Q8_0.gguf".to_string(), 8_000_000_000u64),
        ];
        let result = LlamaCppProvider::select_best_gguf(&files, 10.0);
        assert!(result.is_some());
        let (name, _) = result.unwrap();
        assert!(name.contains("Q8_0"), "should prefer Q8, got: {}", name);
    }

    #[test]
    fn test_select_best_gguf_respects_budget() {
        let files = vec![
            ("model-Q2_K.gguf".to_string(), 2_000_000_000u64),
            ("model-Q4_K_M.gguf".to_string(), 4_000_000_000u64),
            ("model-Q8_0.gguf".to_string(), 8_000_000_000u64),
        ];
        // Budget ~3.7GB → Q2_K fits
        let result = LlamaCppProvider::select_best_gguf(&files, 3.7);
        assert!(result.is_some());
        let (name, _) = result.unwrap();
        assert!(
            name.contains("Q2_K"),
            "should select Q2_K for 3.7GB budget, got: {}",
            name
        );
    }

    #[test]
    fn test_select_best_gguf_nothing_fits() {
        let files = vec![("model-Q2_K.gguf".to_string(), 8_000_000_000u64)];
        let result = LlamaCppProvider::select_best_gguf(&files, 1.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_select_best_gguf_prefers_shard_group_over_lower_quant() {
        // A complete Q4_K_M shard set should beat a non-shard Q2_K when both
        // fit in the budget (Q4 > Q2 in the preference order).
        let files = vec![
            (
                "model-Q4_K_M-00001-of-00003.gguf".to_string(),
                4_000_000_000u64,
            ),
            (
                "model-Q4_K_M-00002-of-00003.gguf".to_string(),
                4_000_000_000u64,
            ),
            (
                "model-Q4_K_M-00003-of-00003.gguf".to_string(),
                4_000_000_000u64,
            ),
            ("model-Q2_K.gguf".to_string(), 2_000_000_000u64),
        ];
        let (name, size) = LlamaCppProvider::select_best_gguf(&files, 16.0).unwrap();
        assert!(name.contains("Q4_K_M-00001-of-00003"), "got: {}", name);
        assert_eq!(size, 12_000_000_000u64);
    }

    #[test]
    fn test_select_best_gguf_empty_list() {
        let result = LlamaCppProvider::select_best_gguf(&[], 10.0);
        assert!(result.is_none());
    }

    // ── parse_shard_info smoke checks ────────────────────────────────

    #[test]
    fn test_parse_shard_info_smoke() {
        assert!(parse_shard_info("model-00001-of-00003.gguf").is_some());
        assert!(parse_shard_info("model-Q4_K_M.gguf").is_none());
        assert!(parse_shard_info("model.gguf").is_none());
    }

    // ── parse_shard_info ─────────────────────────────────────────────

    #[test]
    fn test_parse_shard_info_basic() {
        assert_eq!(
            parse_shard_info("Qwen3-Coder-Next-Q5_K_M-00001-of-00003.gguf"),
            Some((1, 3))
        );
        assert_eq!(
            parse_shard_info("Q5_K_M/Qwen3-Coder-Next-Q5_K_M-00003-of-00003.gguf"),
            Some((3, 3))
        );
    }

    #[test]
    fn test_parse_shard_info_rejects_non_shards() {
        assert_eq!(parse_shard_info("model.gguf"), None);
        assert_eq!(parse_shard_info("model-Q4_K_M.gguf"), None);
        // "of" without trailing digits
        assert_eq!(parse_shard_info("model-of-tea.gguf"), None);
        // wrong extension
        assert_eq!(parse_shard_info("model-00001-of-00003.bin"), None);
        // index out of range
        assert_eq!(parse_shard_info("model-00004-of-00003.gguf"), None);
        // index zero
        assert_eq!(parse_shard_info("model-00000-of-00003.gguf"), None);
    }

    // ── collect_shard_set ────────────────────────────────────────────

    #[test]
    fn test_collect_shard_set_returns_all_shards_sorted() {
        let files = vec![
            (
                "Q5_K_M/Qwen3-Coder-Next-Q5_K_M-00002-of-00003.gguf".to_string(),
                3_000_000_000u64,
            ),
            (
                "Q5_K_M/Qwen3-Coder-Next-Q5_K_M-00001-of-00003.gguf".to_string(),
                3_000_000_000u64,
            ),
            (
                "Q5_K_M/Qwen3-Coder-Next-Q5_K_M-00003-of-00003.gguf".to_string(),
                2_500_000_000u64,
            ),
            // Unrelated file in the same listing
            (
                "Q4_K_M/Qwen3-Coder-Next-Q4_K_M.gguf".to_string(),
                4_000_000_000u64,
            ),
        ];
        let shards =
            collect_shard_set(&files, "Q5_K_M/Qwen3-Coder-Next-Q5_K_M-00001-of-00003.gguf")
                .expect("should detect shard set");
        assert_eq!(shards.len(), 3);
        assert!(shards[0].0.contains("00001-of-00003"));
        assert!(shards[1].0.contains("00002-of-00003"));
        assert!(shards[2].0.contains("00003-of-00003"));
    }

    #[test]
    fn test_collect_shard_set_returns_none_for_non_shard() {
        let files = vec![("model-Q4_K_M.gguf".to_string(), 4_000_000_000u64)];
        assert!(collect_shard_set(&files, "model-Q4_K_M.gguf").is_none());
    }

    #[test]
    fn test_collect_shard_set_does_not_mix_groups() {
        // Two distinct shard groups in the same repo (different quants).
        let files = vec![
            ("Q4_K_M/m-Q4_K_M-00001-of-00002.gguf".to_string(), 1_000),
            ("Q4_K_M/m-Q4_K_M-00002-of-00002.gguf".to_string(), 1_000),
            ("Q5_K_M/m-Q5_K_M-00001-of-00003.gguf".to_string(), 2_000),
            ("Q5_K_M/m-Q5_K_M-00002-of-00003.gguf".to_string(), 2_000),
            ("Q5_K_M/m-Q5_K_M-00003-of-00003.gguf".to_string(), 2_000),
        ];
        let q4 = collect_shard_set(&files, "Q4_K_M/m-Q4_K_M-00001-of-00002.gguf").unwrap();
        assert_eq!(q4.len(), 2);
        let q5 = collect_shard_set(&files, "Q5_K_M/m-Q5_K_M-00002-of-00003.gguf").unwrap();
        assert_eq!(q5.len(), 3);
    }

    // ── select_best_gguf shard awareness ─────────────────────────────

    #[test]
    fn test_select_best_gguf_picks_shard_group() {
        // Repo only has a Q5_K_M shard set; it should be selected (and the
        // returned size should be the sum of all shards).
        let files = vec![
            (
                "Q5_K_M/m-Q5_K_M-00001-of-00003.gguf".to_string(),
                3_000_000_000u64,
            ),
            (
                "Q5_K_M/m-Q5_K_M-00002-of-00003.gguf".to_string(),
                3_000_000_000u64,
            ),
            (
                "Q5_K_M/m-Q5_K_M-00003-of-00003.gguf".to_string(),
                2_000_000_000u64,
            ),
        ];
        let (path, size) = LlamaCppProvider::select_best_gguf(&files, 16.0)
            .expect("shard group should be selectable");
        assert!(path.contains("00001-of-00003"), "got: {}", path);
        assert_eq!(size, 8_000_000_000u64);
    }

    #[test]
    fn test_select_best_gguf_shard_group_respects_budget() {
        let files = vec![
            (
                "Q5_K_M/m-Q5_K_M-00001-of-00003.gguf".to_string(),
                3_000_000_000u64,
            ),
            (
                "Q5_K_M/m-Q5_K_M-00002-of-00003.gguf".to_string(),
                3_000_000_000u64,
            ),
            (
                "Q5_K_M/m-Q5_K_M-00003-of-00003.gguf".to_string(),
                2_000_000_000u64,
            ),
            ("Q2_K/m-Q2_K.gguf".to_string(), 1_500_000_000u64),
        ];
        // 4GB budget: shard group (8GB) doesn't fit, Q2_K does.
        let (path, _) = LlamaCppProvider::select_best_gguf(&files, 4.0).unwrap();
        assert!(path.contains("Q2_K") && !path.contains("-of-"));
    }

    // ── urlencoding ──────────────────────────────────────────────────

    #[test]
    fn test_urlencoding_ascii() {
        assert_eq!(urlencoding::encode("hello"), "hello");
        assert_eq!(urlencoding::encode("test-model_v1.0"), "test-model_v1.0");
    }

    #[test]
    fn test_urlencoding_special_chars() {
        assert_eq!(urlencoding::encode("hello world"), "hello%20world");
        assert_eq!(urlencoding::encode("a+b"), "a%2Bb");
        assert_eq!(urlencoding::encode("foo/bar"), "foo%2Fbar");
    }

    #[test]
    fn test_urlencoding_empty() {
        assert_eq!(urlencoding::encode(""), "");
    }

    // ── is_model_installed_llamacpp ──────────────────────────────────

    #[test]
    fn test_is_model_installed_llamacpp_exact() {
        let mut installed = HashSet::new();
        installed.insert("llama-3.1-8b-instruct".to_string());
        assert!(is_model_installed_llamacpp(
            "meta-llama/Llama-3.1-8B-Instruct",
            &installed
        ));
    }

    #[test]
    fn test_is_model_installed_llamacpp_stripped_suffixes() {
        let mut installed = HashSet::new();
        installed.insert("llama-3.1-8b".to_string());
        assert!(is_model_installed_llamacpp(
            "meta-llama/Llama-3.1-8B-Instruct",
            &installed
        ));
    }

    #[test]
    fn test_is_model_installed_llamacpp_not_installed() {
        let installed = HashSet::new();
        assert!(!is_model_installed_llamacpp(
            "meta-llama/Llama-3.1-8B-Instruct",
            &installed
        ));
    }

    // ── gguf_pull_tag ────────────────────────────────────────────────

    #[test]
    fn test_gguf_pull_tag_known() {
        let tag = gguf_pull_tag("meta-llama/Llama-3.1-8B-Instruct");
        assert!(tag.is_some());
        assert!(tag.unwrap().contains("GGUF"));
    }

    #[test]
    fn test_gguf_pull_tag_unknown() {
        assert!(gguf_pull_tag("totally-unknown/model-xyz").is_none());
    }

    // ── has_ollama_mapping ───────────────────────────────────────────

    #[test]
    fn test_has_ollama_mapping_known() {
        assert!(has_ollama_mapping("meta-llama/Llama-3.1-8B-Instruct"));
        assert!(has_ollama_mapping("Qwen/Qwen2.5-7B-Instruct"));
    }

    #[test]
    fn test_has_ollama_mapping_unknown() {
        assert!(!has_ollama_mapping("totally-unknown/model-xyz"));
    }

    // ── ollama_pull_tag ──────────────────────────────────────────────

    #[test]
    fn test_ollama_pull_tag_known() {
        let tag = ollama_pull_tag("meta-llama/Llama-3.1-8B-Instruct");
        assert_eq!(tag, Some("llama3.1:8b".to_string()));
    }

    #[test]
    fn test_ollama_pull_tag_unknown() {
        assert!(ollama_pull_tag("totally-unknown/model-xyz").is_none());
    }

    // ── mlx_pull_tag ─────────────────────────────────────────────────

    #[test]
    fn test_mlx_pull_tag_prefers_4bit() {
        let tag = mlx_pull_tag("meta-llama/Llama-3.1-8B-Instruct");
        assert!(tag.ends_with("-4bit"), "should prefer 4bit, got: {}", tag);
    }

    #[test]
    fn test_mlx_pull_tag_fallback() {
        let tag = mlx_pull_tag("SomeUnknown/Model-7B");
        assert!(!tag.is_empty());
    }

    // ── ollama_installed_matches_candidate ────────────────────────────

    #[test]
    fn test_ollama_installed_matches_exact() {
        assert!(ollama_installed_matches_candidate(
            "llama3.1:8b",
            "llama3.1:8b"
        ));
    }

    #[test]
    fn test_ollama_installed_matches_variant_suffix() {
        assert!(ollama_installed_matches_candidate(
            "llama3.1:8b-instruct-q4_K_M",
            "llama3.1:8b"
        ));
    }

    #[test]
    fn test_ollama_installed_no_match() {
        assert!(!ollama_installed_matches_candidate(
            "qwen2.5:7b",
            "llama3.1:8b"
        ));
    }

    // ── parse_repo_gguf_entries ──────────────────────────────────────

    #[test]
    fn test_parse_repo_gguf_entries_valid() {
        let entries = vec![
            serde_json::json!({"path": "model-Q4_K_M.gguf", "size": 4_000_000_000u64}),
            serde_json::json!({"path": "model-Q8_0.gguf", "size": 8_000_000_000u64}),
        ];
        let files = parse_repo_gguf_entries(entries);
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].0, "model-Q4_K_M.gguf");
        assert_eq!(files[1].0, "model-Q8_0.gguf");
    }

    #[test]
    fn test_parse_repo_gguf_entries_missing_size_defaults_to_zero() {
        let entries = vec![serde_json::json!({"path": "model.gguf"})];
        let files = parse_repo_gguf_entries(entries);
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].1, 0);
    }

    #[test]
    fn test_parse_repo_gguf_entries_skips_non_gguf() {
        let entries = vec![
            serde_json::json!({"path": "README.md", "size": 1000u64}),
            serde_json::json!({"path": "config.json", "size": 500u64}),
            serde_json::json!({"path": "model.gguf", "size": 4_000_000_000u64}),
        ];
        let files = parse_repo_gguf_entries(entries);
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].0, "model.gguf");
    }

    // ── hf_name_to_mlx_candidates edge cases ─────────────────────────

    #[test]
    fn test_hf_name_to_mlx_candidates_bare_model_name() {
        let candidates = hf_name_to_mlx_candidates("Phi-4");
        assert!(candidates.iter().any(|c| c.contains("phi-4")));
        assert!(candidates.iter().any(|c| c.ends_with("-4bit")));
    }

    #[test]
    fn test_hf_name_to_mlx_candidates_no_duplicates() {
        let candidates = hf_name_to_mlx_candidates("meta-llama/Llama-3.1-8B-Instruct");
        let unique: HashSet<_> = candidates.iter().collect();
        assert_eq!(
            unique.len(),
            candidates.len(),
            "candidates should have no duplicates: {:?}",
            candidates
        );
    }

    // ── hf_name_to_ollama_candidates edge cases ──────────────────────

    #[test]
    fn test_hf_name_to_ollama_candidates_unknown_generates_fallback() {
        // Models without an explicit mapping should still generate
        // heuristic candidates so installed detection has something to match.
        let candidates = hf_name_to_ollama_candidates("totally-unknown/model-xyz");
        assert!(
            !candidates.is_empty(),
            "fallback candidate generation should produce at least one entry"
        );
        // All candidates should be lowercased
        for c in &candidates {
            assert_eq!(c, &c.to_lowercase(), "candidate should be lowercase: {c}");
        }
    }

    #[test]
    fn test_hf_name_to_ollama_candidates_multiple_models() {
        // Test a variety of known models
        assert!(!hf_name_to_ollama_candidates("meta-llama/Llama-3.1-8B-Instruct").is_empty());
        assert!(!hf_name_to_ollama_candidates("Qwen/Qwen2.5-Coder-7B-Instruct").is_empty());
        assert!(!hf_name_to_ollama_candidates("google/gemma-2-9b-it").is_empty());
    }

    // ── split_name_and_size ───────────────────────────────────────

    #[test]
    fn test_split_name_and_size_basic() {
        assert_eq!(
            split_name_and_size("qwen2.5-coder-14b"),
            Some(("qwen2.5-coder", "14b"))
        );
    }

    #[test]
    fn test_split_name_and_size_moe() {
        assert_eq!(
            split_name_and_size("qwen3-coder-30b-a3b"),
            Some(("qwen3-coder", "30b-a3b"))
        );
    }

    #[test]
    fn test_split_name_and_size_no_size() {
        // "phi-4" has no "b" suffix — "4" is not a size tag
        assert_eq!(split_name_and_size("phi-4"), None);
    }

    #[test]
    fn test_split_name_and_size_deepseek() {
        assert_eq!(
            split_name_and_size("deepseek-r1-distill-qwen-32b"),
            Some(("deepseek-r1-distill-qwen", "32b"))
        );
    }

    #[test]
    fn test_split_name_and_size_fractional() {
        assert_eq!(split_name_and_size("qwen3-1.7b"), Some(("qwen3", "1.7b")));
    }

    // ── fallback ollama candidate matching ──────────────────────────

    #[test]
    fn test_fallback_ollama_candidates_match_installed() {
        // Simulate a model NOT in OLLAMA_MAPPINGS but running in Ollama
        let candidates = hf_name_to_ollama_candidates("SomeOrg/CoolModel-13B-Instruct");
        // Should generate "coolmodel:13b" as a candidate
        assert!(
            candidates.contains(&"coolmodel:13b".to_string()),
            "expected 'coolmodel:13b' in candidates: {:?}",
            candidates
        );

        // Verify it matches against an installed set
        let mut installed = HashSet::new();
        installed.insert("coolmodel:13b".to_string());
        installed.insert("coolmodel".to_string());
        assert!(is_model_installed(
            "SomeOrg/CoolModel-13B-Instruct",
            &installed
        ));
    }

    #[test]
    fn test_fallback_ollama_moe_candidate() {
        // Use a fictitious MoE model that is NOT in OLLAMA_MAPPINGS
        let candidates = hf_name_to_ollama_candidates("FakeOrg/FakeModel-30B-A3B-Instruct");
        assert!(
            candidates.contains(&"fakemodel:30b-a3b".to_string()),
            "expected 'fakemodel:30b-a3b' in candidates: {:?}",
            candidates
        );
    }

    #[test]
    fn test_installed_hf_name_direct_match() {
        // /api/v1/installed returns the full HF name lowercased
        let mut installed = HashSet::new();
        installed.insert("deepseek-ai/deepseek-r1-distill-qwen-32b".to_string());
        assert!(is_model_installed(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            &installed
        ));
    }

    // ── Docker Model Runner ─────────────────────────────────────────

    #[test]
    fn test_docker_mr_catalog_parses() {
        // The embedded catalog should parse without errors
        let catalog = docker_mr_catalog();
        assert!(!catalog.is_empty(), "Docker MR catalog should not be empty");
    }

    #[test]
    fn test_has_docker_mr_mapping_known() {
        // Llama 3.1 70B is in both our HF database and Docker Hub ai/ namespace
        assert!(has_docker_mr_mapping("meta-llama/Llama-3.1-70B-Instruct"));
    }

    #[test]
    fn test_has_docker_mr_mapping_unknown() {
        assert!(!has_docker_mr_mapping("totally-unknown/model-xyz"));
    }

    #[test]
    fn test_docker_mr_pull_tag_returns_ai_prefixed() {
        let tag = docker_mr_pull_tag("meta-llama/Llama-3.1-70B-Instruct");
        assert!(tag.is_some());
        assert!(tag.unwrap().starts_with("ai/"));
    }

    #[test]
    fn test_docker_mr_candidates_includes_ai_prefix() {
        let candidates = hf_name_to_docker_mr_candidates("meta-llama/Llama-3.1-70B-Instruct");
        assert!(candidates.iter().any(|c| c.starts_with("ai/")));
    }

    #[test]
    fn test_docker_mr_candidates_unknown_returns_empty() {
        let candidates = hf_name_to_docker_mr_candidates("totally-unknown/model-xyz");
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_is_model_installed_docker_mr_exact() {
        let mut installed = HashSet::new();
        installed.insert("ai/llama3.1:70b".to_string());
        installed.insert("llama3.1:70b".to_string());
        installed.insert("llama3.1".to_string());
        assert!(is_model_installed_docker_mr(
            "meta-llama/Llama-3.1-70B-Instruct",
            &installed
        ));
    }

    #[test]
    fn test_is_model_installed_docker_mr_variant_suffix() {
        let mut installed = HashSet::new();
        installed.insert("ai/llama3.1:70b-q4_k_m".to_string());
        assert!(is_model_installed_docker_mr(
            "meta-llama/Llama-3.1-70B-Instruct",
            &installed
        ));
    }

    #[test]
    fn test_is_model_installed_docker_mr_not_installed() {
        let installed = HashSet::new();
        assert!(!is_model_installed_docker_mr(
            "meta-llama/Llama-3.1-70B-Instruct",
            &installed
        ));
    }

    #[test]
    fn test_normalize_docker_mr_host_with_scheme() {
        assert_eq!(
            normalize_docker_mr_host("https://docker.example.com:12434"),
            Some("https://docker.example.com:12434".to_string())
        );
    }

    #[test]
    fn test_normalize_docker_mr_host_without_scheme() {
        assert_eq!(
            normalize_docker_mr_host("docker.example.com:12434"),
            Some("http://docker.example.com:12434".to_string())
        );
    }

    #[test]
    fn test_normalize_docker_mr_host_rejects_unsupported_scheme() {
        assert_eq!(
            normalize_docker_mr_host("ftp://docker.example.com:12434"),
            None
        );
    }

    // ── vLLM ──────────────────────────────────────────────────────────

    #[test]
    fn test_hf_name_to_vllm_candidates() {
        let candidates = hf_name_to_vllm_candidates("meta-llama/Llama-3.1-8B-Instruct");
        assert!(
            candidates
                .iter()
                .any(|c| c == "meta-llama/llama-3.1-8b-instruct")
        );
        assert!(candidates.iter().any(|c| c == "llama-3.1-8b-instruct"));
        // stripped variant (without -instruct)
        assert!(candidates.iter().any(|c| c == "llama-3.1-8b"));
    }

    #[test]
    fn test_is_model_installed_vllm() {
        let mut installed = HashSet::new();
        installed.insert("meta-llama/llama-3.1-8b-instruct".to_string());
        assert!(is_model_installed_vllm(
            "meta-llama/Llama-3.1-8B-Instruct",
            &installed
        ));
        assert!(!is_model_installed_vllm(
            "meta-llama/Llama-3.1-70B-Instruct",
            &installed
        ));
    }

    #[test]
    fn test_has_vllm_mapping() {
        assert!(has_vllm_mapping("meta-llama/Llama-3.1-8B-Instruct"));
        assert!(!has_vllm_mapping(""));
    }

    #[test]
    fn test_vllm_pull_tag() {
        assert_eq!(
            vllm_pull_tag("meta-llama/Llama-3.1-8B-Instruct"),
            Some("meta-llama/Llama-3.1-8B-Instruct".to_string())
        );
        assert_eq!(vllm_pull_tag(""), None);
    }

    #[test]
    fn test_normalize_vllm_host_with_scheme() {
        assert_eq!(
            normalize_vllm_host("http://myhost:8000"),
            Some("http://myhost:8000".to_string())
        );
    }

    #[test]
    fn test_normalize_vllm_host_without_scheme() {
        assert_eq!(
            normalize_vllm_host("myhost:8000"),
            Some("http://myhost:8000".to_string())
        );
    }

    #[test]
    fn test_normalize_vllm_host_rejects_unsupported_scheme() {
        assert_eq!(normalize_vllm_host("ftp://myhost:8000"), None);
    }

    #[test]
    fn test_normalize_vllm_host_empty() {
        assert_eq!(normalize_vllm_host(""), None);
        assert_eq!(normalize_vllm_host("  "), None);
    }
}
