//! Runtime model providers (Ollama, etc.).
//!
//! Each provider can list locally installed models and pull new ones.
//! The trait is designed to be extended for llama.cpp, vLLM, etc.

use std::collections::HashSet;

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
        Self {
            base_url: std::env::var("OLLAMA_HOST")
                .unwrap_or_else(|_| "http://localhost:11434".to_string()),
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
        Self {
            server_url: std::env::var("MLX_LM_HOST")
                .unwrap_or_else(|_| "http://localhost:8080".to_string()),
        }
    }
}

impl MlxProvider {
    pub fn new() -> Self {
        Self::default()
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
