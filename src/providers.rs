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
    Progress { status: String, percent: Option<f64> },
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
    status: String,
    #[serde(default)]
    total: Option<u64>,
    #[serde(default)]
    completed: Option<u64>,
}

impl ModelProvider for OllamaProvider {
    fn name(&self) -> &str {
        "Ollama"
    }

    fn is_available(&self) -> bool {
        ureq::get(&self.api_url("tags"))
            .timeout(std::time::Duration::from_secs(2))
            .call()
            .is_ok()
    }

    fn installed_models(&self) -> HashSet<String> {
        let mut set = HashSet::new();
        let Ok(resp) = ureq::get(&self.api_url("tags"))
            .timeout(std::time::Duration::from_secs(5))
            .call()
        else {
            return set;
        };
        let Ok(tags): Result<TagsResponse, _> = resp.into_json() else {
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
                .timeout(std::time::Duration::from_secs(3600))
                .send_json(&body);

            match resp {
                Ok(resp) => {
                    let reader = std::io::BufReader::new(resp.into_reader());
                    use std::io::BufRead;
                    for line in reader.lines() {
                        let Ok(line) = line else { break };
                        if line.is_empty() {
                            continue;
                        }
                        if let Ok(parsed) = serde_json::from_str::<PullStreamLine>(&line) {
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
                    let _ = tx.send(PullEvent::Done);
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
// Name-matching helpers
// ---------------------------------------------------------------------------

/// Map a HuggingFace model name (e.g. "meta-llama/Llama-3.1-8B-Instruct")
/// to the Ollama tag that would serve it (e.g. "llama3.1:8b-instruct").
///
/// This is a best-effort fuzzy match. Ollama naming is not 1-to-1 with HF,
/// but we can cover the common patterns. Returns multiple candidates so that
/// `is_installed()` can check any of them.
pub fn hf_name_to_ollama_candidates(hf_name: &str) -> Vec<String> {
    let mut candidates = Vec::new();

    // Take the part after the slash (repo name)
    let repo = hf_name
        .split('/')
        .last()
        .unwrap_or(hf_name)
        .to_lowercase();

    // Common provider-specific mappings from HF repo names → Ollama tags.
    // These are checked first since they're authoritative.
    let mappings: &[(&str, &str)] = &[
        // Meta Llama family
        ("llama-3.3-70b-instruct", "llama3.3:70b"),
        ("llama-3.2-11b-vision-instruct", "llama3.2-vision:11b"),
        ("llama-3.2-3b", "llama3.2:3b"),
        ("llama-3.2-1b", "llama3.2:1b"),
        ("llama-3.1-405b-instruct", "llama3.1:405b"),
        ("llama-3.1-70b-instruct", "llama3.1:70b"),
        ("llama-3.1-8b-instruct", "llama3.1:8b"),
        ("llama-3.1-8b", "llama3.1:8b"),
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
        ("phi-3.5-mini-instruct", "phi3.5"),
        ("phi-3-mini-4k-instruct", "phi3"),
        ("phi-3-medium-14b-instruct", "phi3:14b"),
        ("orca-2-7b", "orca2:7b"),
        ("orca-2-13b", "orca2:13b"),
        // Mistral
        ("mistral-7b-instruct-v0.3", "mistral:7b"),
        ("mistral-nemo-instruct-2407", "mistral-nemo"),
        ("mistral-small-24b-instruct-2501", "mistral-small:24b"),
        ("mixtral-8x7b-instruct-v0.1", "mixtral:8x7b"),
        ("ministral-8b-instruct-2410", "ministral:8b"),
        // Qwen
        ("qwen2.5-72b-instruct", "qwen2.5:72b"),
        ("qwen2.5-32b-instruct", "qwen2.5:32b"),
        ("qwen2.5-14b-instruct", "qwen2.5:14b"),
        ("qwen2.5-7b-instruct", "qwen2.5:7b"),
        ("qwen2.5-coder-32b-instruct", "qwen2.5-coder:32b"),
        ("qwen2.5-coder-14b-instruct", "qwen2.5-coder:14b"),
        ("qwen2.5-coder-7b-instruct", "qwen2.5-coder:7b"),
        ("qwen2.5-coder-1.5b-instruct", "qwen2.5-coder:1.5b"),
        ("qwen2.5-vl-7b-instruct", "qwen2.5vl:7b"),
        ("qwen2.5-vl-3b-instruct", "qwen2.5vl:3b"),
        ("qwen3-235b-a22b", "qwen3:235b"),
        ("qwen3-32b", "qwen3:32b"),
        ("qwen3-30b-a3b", "qwen3:30b-a3b"),
        ("qwen3-14b", "qwen3:14b"),
        ("qwen3-8b", "qwen3:8b"),
        ("qwen3-4b", "qwen3:4b"),
        ("qwen3-1.7b", "qwen3:1.7b"),
        ("qwen3-0.6b", "qwen3:0.6b"),
        ("qwen3-coder-480b-a35b-instruct", "qwen3-coder"),
        // DeepSeek
        ("deepseek-v3", "deepseek-v3"),
        ("deepseek-r1-distill-qwen-32b", "deepseek-r1:32b"),
        ("deepseek-r1-distill-qwen-7b", "deepseek-r1:7b"),
        ("deepseek-coder-v2-lite-instruct", "deepseek-coder-v2:16b"),
        // Others
        ("tinyllama-1.1b-chat-v1.0", "tinyllama"),
        ("stablelm-2-1_6b-chat", "stablelm2:1.6b"),
        ("yi-6b-chat", "yi:6b"),
        ("yi-34b-chat", "yi:34b"),
        ("starcoder2-7b", "starcoder2:7b"),
        ("starcoder2-15b", "starcoder2:15b"),
        ("falcon-7b-instruct", "falcon:7b"),
        ("falcon-40b-instruct", "falcon:40b"),
        ("falcon3-7b-instruct", "falcon3:7b"),
        ("falcon3-10b-instruct", "falcon3:10b"),
        ("wizardlm-13b-v1.2", "wizardlm2:13b"),
        ("wizardcoder-15b-v1.0", "wizardcoder:15b"),
        ("openchat-3.5-0106", "openchat:7b"),
        ("vicuna-7b-v1.5", "vicuna:7b"),
        ("vicuna-13b-v1.5", "vicuna:13b"),
        ("glm-4-9b-chat", "glm4:9b"),
        ("solar-10.7b-instruct-v1.0", "solar:10.7b"),
        ("olmo-2-0325-32b-instruct", "olmo2:32b"),
        ("granite-3.1-8b-instruct", "granite3.1-dense:8b"),
        ("granite-4.0-h-micro", "granite4.0-h:micro"),
        ("granite-4.0-h-tiny", "granite4.0-h:tiny"),
        ("granite-4.0-h-small", "granite4.0-h:small"),
        ("zephyr-7b-beta", "zephyr:7b"),
        ("c4ai-command-r-v01", "command-r"),
        ("nous-hermes-2-mixtral-8x7b-dpo", "nous-hermes2-mixtral:8x7b"),
        ("nomic-embed-text-v1.5", "nomic-embed-text"),
        ("bge-large-en-v1.5", "bge-large"),
    ];

    for &(hf_suffix, ollama_tag) in mappings {
        if repo == hf_suffix {
            candidates.push(ollama_tag.to_string());
            return candidates;
        }
    }

    // Fallback: generate plausible candidates from the repo name
    // Strip common suffixes
    let stripped = repo
        .replace("-instruct", "")
        .replace("-chat", "")
        .replace("-hf", "")
        .replace("-it", "");
    candidates.push(stripped.clone());
    candidates.push(repo.clone());

    candidates
}

/// Check if any of the Ollama candidates for an HF model appear in the
/// installed set.
pub fn is_model_installed(hf_name: &str, installed: &HashSet<String>) -> bool {
    let candidates = hf_name_to_ollama_candidates(hf_name);
    candidates.iter().any(|c| installed.contains(c))
}

/// Given an HF model name, return the best Ollama tag to use for pulling.
pub fn ollama_pull_tag(hf_name: &str) -> String {
    let candidates = hf_name_to_ollama_candidates(hf_name);
    candidates.into_iter().next().unwrap_or_else(|| {
        hf_name
            .split('/')
            .last()
            .unwrap_or(hf_name)
            .to_lowercase()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen_coder_14b_matches_coder_entry() {
        // "qwen2.5-coder:14b" from `ollama list` should match
        // the HF entry "Qwen/Qwen2.5-Coder-14B-Instruct", NOT
        // the base "Qwen/Qwen2.5-14B-Instruct".
        let mut installed = HashSet::new();
        installed.insert("qwen2.5-coder:14b".to_string());
        installed.insert("qwen2.5-coder".to_string());

        assert!(is_model_installed("Qwen/Qwen2.5-Coder-14B-Instruct", &installed));
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
        assert!(!is_model_installed("Qwen/Qwen2.5-Coder-14B-Instruct", &installed));
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
        let candidates = hf_name_to_ollama_candidates("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct");
        assert!(candidates.contains(&"deepseek-coder-v2:16b".to_string()));
    }
}
