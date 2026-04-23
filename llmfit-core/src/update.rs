//! Online model database updates via the HuggingFace Hub API.
//!
//! Fetches trending / top-downloaded text-generation models and caches them in
//! `~/.llmfit/hf_models_cache.json` (Linux/macOS) or
//! `%APPDATA%\llmfit\hf_models_cache.json` (Windows).
//!
//! The cache is automatically merged with the embedded model list each time
//! `ModelDatabase::new()` is called, so users immediately benefit from any
//! previously fetched models without needing to rebuild the binary.

use serde::Deserialize;
use std::collections::HashSet;
use std::path::PathBuf;

use crate::models::{LlmModel, ModelFormat};

const HF_API: &str = "https://huggingface.co/api/models";

/// Bump this when the `LlmModel` schema changes in a breaking way.
/// A cache written by an older version will be discarded and re-fetched.
const CACHE_VERSION: u32 = 3;

// ── Cache helpers ─────────────────────────────────────────────────────────────

#[derive(serde::Serialize, serde::Deserialize)]
struct CacheEnvelope {
    version: u32,
    models: Vec<LlmModel>,
}

/// Returns the llmfit data directory.
/// `~/.llmfit` on Linux/macOS, `%APPDATA%\llmfit` on Windows.
pub fn cache_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        std::env::var("APPDATA")
            .ok()
            .map(|p| PathBuf::from(p).join("llmfit"))
    }
    #[cfg(not(target_os = "windows"))]
    {
        std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .ok()
            .map(|p| PathBuf::from(p).join(".llmfit"))
    }
}

/// Full path to the cached model list JSON file.
pub fn cache_file() -> Option<PathBuf> {
    Some(cache_dir()?.join("hf_models_cache.json"))
}

/// Load any previously cached models.
///
/// Returns an empty vec if the cache is missing, corrupt, or was written by
/// a different schema version (triggering a silent re-fetch on next update).
pub fn load_cache() -> Vec<LlmModel> {
    let path = match cache_file() {
        Some(p) if p.exists() => p,
        _ => return vec![],
    };
    let Ok(content) = std::fs::read_to_string(&path) else {
        return vec![];
    };
    match serde_json::from_str::<CacheEnvelope>(&content) {
        Ok(env) if env.version == CACHE_VERSION => env.models,
        // Version mismatch or old unversioned format — discard stale cache.
        _ => vec![],
    }
}

/// Persist a model list to the cache file, creating the directory if needed.
pub fn save_cache(models: &[LlmModel]) -> Result<(), String> {
    let path = cache_file().ok_or_else(|| "Cannot determine cache directory".to_string())?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("Failed to create cache dir: {e}"))?;
    }
    let envelope = CacheEnvelope {
        version: CACHE_VERSION,
        models: models.to_vec(),
    };
    let json =
        serde_json::to_string_pretty(&envelope).map_err(|e| format!("Serialize error: {e}"))?;
    std::fs::write(&path, json).map_err(|e| format!("Failed to write cache: {e}"))?;
    Ok(())
}

/// Delete the cache file.  Returns the number of models that were removed.
pub fn clear_cache() -> Result<usize, String> {
    let path = match cache_file() {
        Some(p) if p.exists() => p,
        _ => return Ok(0),
    };
    let count = load_cache().len();
    std::fs::remove_file(&path).map_err(|e| format!("Failed to delete cache: {e}"))?;
    Ok(count)
}

// ── HuggingFace API types ─────────────────────────────────────────────────────

#[derive(Deserialize, Debug)]
struct HfApiModel {
    id: String,
    #[serde(default)]
    author: Option<String>,
    #[serde(default)]
    pipeline_tag: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(rename = "createdAt", default)]
    created_at: Option<String>,
    /// Exact parameter count when safetensors metadata is available.
    #[serde(default)]
    safetensors: Option<SafetensorsInfo>,
    #[serde(default)]
    license: Option<String>,
}

#[derive(Deserialize, Debug, Default)]
struct SafetensorsInfo {
    #[serde(default)]
    total: Option<u64>,
}

// ── Parameter extraction ──────────────────────────────────────────────────────

/// Parse "7B" → 7_000_000_000u64, "1.5B" → 1_500_000_000u64, "500M" → 500_000_000u64.
fn parse_param_str(s: &str) -> Option<u64> {
    if let Some(b) = s.strip_suffix('B') {
        let v: f64 = b.parse().ok()?;
        if (0.05..=2_000_000.0).contains(&v) {
            return Some((v * 1_000_000_000.0) as u64);
        }
    }
    if let Some(m) = s.strip_suffix('M') {
        let v: f64 = m.parse().ok()?;
        if (1.0..=999_999.0).contains(&v) {
            return Some((v * 1_000_000.0) as u64);
        }
    }
    None
}

/// Parse "16E", "128E" → Some(16), Some(128).
fn parse_expert_suffix(s: &str) -> Option<u32> {
    s.strip_suffix('E')?.parse().ok()
}

/// Derive (param_str, params_raw, is_moe, num_experts, active_experts, active_params)
/// from a model identifier string.
fn extract_model_params(
    model_id: &str,
) -> (
    String,
    Option<u64>,
    bool,
    Option<u32>,
    Option<u32>,
    Option<u64>,
) {
    let up = model_id.to_uppercase();
    // Split on typical separators but NOT on '.' so that "1.5B" stays intact.
    let tokens: Vec<&str> = up.split(['-', '/', '_', ' ', ':']).collect();

    // MoE pattern "8X7B": N experts × M params each.
    for tok in &tokens {
        if let Some(x_pos) = tok.find('X') {
            let (left, right) = tok.split_at(x_pos);
            let right = &right[1..]; // skip 'X'
            if let (Ok(n_exp), Some(per_exp)) = (left.parse::<u32>(), parse_param_str(right))
                && (2..=512).contains(&n_exp)
            {
                let total = per_exp.saturating_mul(n_exp as u64);
                let active_exp = 2u32.min(n_exp);
                let active = per_exp.saturating_mul(active_exp as u64);
                let pb = per_exp / 1_000_000_000;
                let s = if pb > 0 {
                    format!("{}x{}B", n_exp, pb)
                } else {
                    format!("{}x{}M", n_exp, per_exp / 1_000_000)
                };
                return (
                    s,
                    Some(total),
                    true,
                    Some(n_exp),
                    Some(active_exp),
                    Some(active),
                );
            }
        }
    }

    // MoE pattern "17B-16E": per-expert params + expert count.
    for window in tokens.windows(2) {
        if let (Some(pb), Some(ne)) = (parse_param_str(window[0]), parse_expert_suffix(window[1]))
            && (2..=512).contains(&ne)
        {
            let total = pb.saturating_mul(ne as u64);
            let ae = 2u32.min(ne);
            let active = pb.saturating_mul(ae as u64);
            let s = format!("{}B", pb / 1_000_000_000);
            return (s, Some(total), true, Some(ne), Some(ae), Some(active));
        }
    }

    // Standard dense model: first matching NB / NM token wins.
    for tok in &tokens {
        if let Some(raw) = parse_param_str(tok) {
            let b = raw / 1_000_000_000;
            let frac = (raw % 1_000_000_000) / 100_000_000;
            let s = if b > 0 {
                if frac > 0 {
                    format!("{}.{}B", b, frac)
                } else {
                    format!("{}B", b)
                }
            } else {
                format!("{}M", raw / 1_000_000)
            };
            return (s, Some(raw), false, None, None, None);
        }
    }

    // Could not parse param count from the model name — mark as unknown rather
    // than silently guessing 7B, which would produce misleading RAM estimates.
    ("Unknown".to_string(), None, false, None, None, None)
}

// ── Use-case inference ────────────────────────────────────────────────────────

fn infer_use_case(model_id: &str, tags: &[String]) -> String {
    let lower = format!(
        "{} {}",
        model_id.to_lowercase(),
        tags.join(" ").to_lowercase()
    );
    if lower.contains("embed") || lower.contains("bge") || lower.contains("-e5-") {
        "Embedding".to_string()
    } else if lower.contains("code") || lower.contains("starcoder") || lower.contains("coder") {
        "Code generation".to_string()
    } else if lower.contains("vision")
        || lower.contains("-vl")
        || lower.contains("llava")
        || lower.contains("multimodal")
    {
        "Vision & Language".to_string()
    } else if lower.contains("-r1")
        || lower.contains("reasoning")
        || lower.contains("thinking")
        || lower.contains("qwq")
    {
        "Reasoning & chain-of-thought".to_string()
    } else if lower.contains("instruct") || lower.contains("chat") || lower.contains("assistant") {
        "Chat & instruction following".to_string()
    } else {
        "General text generation".to_string()
    }
}

// ── Context-length inference ──────────────────────────────────────────────────

fn infer_context_length(model_id: &str, params_raw: Option<u64>) -> u32 {
    let low = model_id.to_lowercase();
    for (kw, ctx) in &[
        ("1m", 1_048_576u32),
        ("128k", 131_072),
        ("64k", 65_536),
        ("32k", 32_768),
        ("16k", 16_384),
        ("8k", 8_192),
    ] {
        if low.contains(kw) {
            return *ctx;
        }
    }
    if low.contains("llama-4") || low.contains("llama4") {
        return 1_048_576;
    }
    if low.contains("llama-3") || low.contains("llama3") {
        return 131_072;
    }
    if low.contains("qwen2") || low.contains("qwen3") || low.contains("mistral") {
        return 32_768;
    }
    if low.contains("gemma") {
        return 8_192;
    }
    match params_raw {
        Some(p) if p >= 70_000_000_000 => 32_768,
        Some(p) if p >= 13_000_000_000 => 16_384,
        Some(p) if p >= 3_000_000_000 => 8_192,
        _ => 4_096,
    }
}

// ── RAM estimation ────────────────────────────────────────────────────────────

fn estimate_ram(
    params_raw: u64,
    _is_moe: bool,
    _active_params: Option<u64>,
) -> (f64, f64, Option<f64>) {
    let total_b = params_raw as f64 / 1_000_000_000.0;
    // Q2_K bpp ≈ 0.37 → absolute floor
    let min_ram = (total_b * 0.37 + 0.5).max(1.0);
    // Q4_K_M bpp ≈ 0.58 → comfortable default
    let rec_ram = (total_b * 0.58 + 1.0).max(2.0);
    // min_vram uses total params — all MoE experts must be loaded into memory.
    // Active-params optimization only applies to throughput estimation (fit.rs).
    let min_vram = (total_b * 0.58 + 0.5).max(1.0);
    (min_ram, rec_ram, Some(min_vram))
}

// ── HF config.json fetching ───────────────────────────────────────────────────

/// Subset of fields we read from a model's `config.json`. Everything is
/// optional because configs vary wildly across architectures.
#[derive(Deserialize, Debug, Default)]
struct HfConfig {
    #[serde(default)]
    num_hidden_layers: Option<u32>,
    #[serde(default)]
    num_attention_heads: Option<u32>,
    #[serde(default)]
    num_key_value_heads: Option<u32>,
    #[serde(default)]
    head_dim: Option<u32>,
    #[serde(default)]
    hidden_size: Option<u32>,
    // MoE architecture fields for bandwidth decomposition
    #[serde(default)]
    vocab_size: Option<u32>,
    #[serde(default)]
    moe_intermediate_size: Option<u32>,
    #[serde(default)]
    intermediate_size: Option<u32>,
    #[serde(default)]
    shared_expert_intermediate_size: Option<u32>,
    #[serde(default)]
    n_shared_experts: Option<u32>,
    // Expert count naming variants
    #[serde(default)]
    n_routed_experts: Option<u32>,
    #[serde(default)]
    num_local_experts: Option<u32>,
    // Nested config (Qwen3.5 vision+text models store LLM params under text_config)
    #[serde(default)]
    text_config: Option<Box<HfConfig>>,
}

/// Fetch a model's `config.json` from the HuggingFace resolve endpoint.
/// Returns `None` on any failure (missing config, network error, parse
/// error). This is best-effort metadata enrichment, not load-bearing.
fn fetch_hf_config(repo_id: &str, token: Option<&str>) -> Option<HfConfig> {
    let url = format!(
        "https://huggingface.co/{}/resolve/main/config.json",
        repo_id
    );
    let req = if let Some(t) = token {
        ureq::get(&url)
            .header("Authorization", &format!("Bearer {}", t))
            .config()
            .timeout_global(Some(std::time::Duration::from_secs(10)))
            .build()
    } else {
        ureq::get(&url)
            .config()
            .timeout_global(Some(std::time::Duration::from_secs(10)))
            .build()
    };
    let resp = req.call().ok()?;
    resp.into_body().read_json::<HfConfig>().ok()
}

/// Resolve a `head_dim` value from a config. Prefers the explicit field,
/// otherwise computes `hidden_size / num_attention_heads` when both are
/// available.
fn resolve_head_dim(cfg: &HfConfig) -> Option<u32> {
    if let Some(h) = cfg.head_dim {
        return Some(h);
    }
    let hidden = cfg.hidden_size?;
    let heads = cfg.num_attention_heads?;
    if heads == 0 {
        return None;
    }
    Some(hidden / heads)
}

// ── HF API fetching ───────────────────────────────────────────────────────────

fn hf_get_list(sort: &str, limit: usize, token: Option<&str>) -> Result<Vec<HfApiModel>, String> {
    let url = format!(
        "{}?pipeline_tag=text-generation&sort={}&limit={}",
        HF_API, sort, limit
    );
    let resp = if let Some(t) = token {
        ureq::get(&url)
            .header("Authorization", &format!("Bearer {}", t))
            .config()
            .timeout_global(Some(std::time::Duration::from_secs(30)))
            .build()
            .call()
    } else {
        ureq::get(&url)
            .config()
            .timeout_global(Some(std::time::Duration::from_secs(30)))
            .build()
            .call()
    };
    match resp {
        Ok(r) => r
            .into_body()
            .read_json::<Vec<HfApiModel>>()
            .map_err(|e| format!("Failed to parse HuggingFace API response: {e}")),
        Err(e) => {
            let msg = e.to_string();
            Err(if msg.contains("401") || msg.contains("Unauthorized") {
                "HTTP 401 Unauthorized — is HF_TOKEN set and valid?".to_string()
            } else if msg.contains("403") || msg.contains("Forbidden") {
                "HTTP 403 Forbidden — token may lack read permission".to_string()
            } else if msg.contains("429") || msg.contains("Too Many") {
                "HTTP 429 Rate limited — wait a moment and retry".to_string()
            } else {
                format!("HuggingFace API error: {e}")
            })
        }
    }
}

/// Convert a raw HF API entry into an `LlmModel`.
/// Returns `None` for models that cannot be characterised as text-generation.
///
/// When `token` is provided, also fetches `config.json` to populate the
/// architecture metadata used by the precise KV cache formula
/// (`num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`,
/// `head_dim`). The fetch is best-effort and silently degrades to `None`.
fn map_to_llm_model(hf: HfApiModel, token: Option<&str>) -> Option<LlmModel> {
    let is_tg = hf.pipeline_tag.as_deref() == Some("text-generation")
        || hf.tags.iter().any(|t| t == "text-generation");
    if !is_tg {
        return None;
    }

    // Use safetensors for an exact parameter count when available, but always
    // run name-based parsing for MoE architecture hints — safetensors only
    // reports total parameters and would cause MoE models (e.g. Mixtral) to
    // lose their MoE classification and receive inaccurate VRAM estimates.
    let (param_str, params_raw, is_moe, num_experts, active_experts, active_params) =
        if let Some(total) = hf.safetensors.as_ref().and_then(|s| s.total) {
            let (_, _, is_moe, num_experts, active_experts, active_params) =
                extract_model_params(&hf.id);
            let b = total / 1_000_000_000;
            let frac = (total % 1_000_000_000) / 100_000_000;
            let s = if b > 0 {
                if frac > 0 {
                    format!("{}.{}B", b, frac)
                } else {
                    format!("{}B", b)
                }
            } else {
                format!("{}M", total / 1_000_000)
            };
            (
                s,
                Some(total),
                is_moe,
                num_experts,
                active_experts,
                active_params,
            )
        } else {
            extract_model_params(&hf.id)
        };

    let raw = params_raw.unwrap_or(7_000_000_000);
    let use_case = infer_use_case(&hf.id, &hf.tags);
    let context_length = infer_context_length(&hf.id, params_raw);
    let (min_ram, rec_ram, min_vram) = estimate_ram(raw, is_moe, active_params);

    let provider = hf
        .author
        .as_deref()
        .or_else(|| hf.id.split('/').next())
        .unwrap_or("Unknown")
        .to_string();

    let release_date = hf
        .created_at
        .as_deref()
        .map(|s| s.get(..10).unwrap_or(s).to_string());

    let license = hf.license.or_else(|| {
        hf.tags
            .iter()
            .find_map(|t| t.strip_prefix("license:").map(|l| l.to_string()))
    });

    // Best-effort architecture metadata. We always try to fetch even without
    // a token because most public models allow anonymous config.json access;
    // the fetch returns None on failure and the precise KV formula falls
    // back to the linear approximation in that case.
    let cfg = fetch_hf_config(&hf.id, token);
    let (
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        hidden_size,
        vocab_size,
        moe_intermediate_size,
        shared_expert_intermediate_size,
    ) = if let Some(c) = cfg.as_ref() {
        // Flatten nested text_config if present (Qwen3.5 vision+text models)
        let flat = c.text_config.as_deref().unwrap_or(c);
        let hidden = flat.hidden_size.or(c.hidden_size);
        let vocab = flat.vocab_size.or(c.vocab_size);
        // moe_intermediate_size: explicit field wins, else fall back to intermediate_size
        let moe_inter = flat
            .moe_intermediate_size
            .or(c.moe_intermediate_size)
            .or(flat.intermediate_size)
            .or(c.intermediate_size);
        // shared_expert_intermediate_size: explicit field wins,
        // else derive from n_shared_experts × intermediate_size (DeepSeek-V2/V3)
        let shared_inter = flat
            .shared_expert_intermediate_size
            .or(c.shared_expert_intermediate_size)
            .or_else(|| {
                let n_shared = flat.n_shared_experts.or(c.n_shared_experts)?;
                let inter = flat.intermediate_size.or(c.intermediate_size)?;
                Some(n_shared * inter)
            });
        (
            flat.num_hidden_layers.or(c.num_hidden_layers),
            flat.num_attention_heads.or(c.num_attention_heads),
            flat.num_key_value_heads
                .or(c.num_key_value_heads)
                .or(flat.num_attention_heads)
                .or(c.num_attention_heads),
            resolve_head_dim(flat),
            hidden,
            vocab,
            moe_inter,
            shared_inter,
        )
    } else {
        (None, None, None, None, None, None, None, None)
    };

    Some(LlmModel {
        name: hf.id.clone(),
        provider,
        parameter_count: param_str,
        parameters_raw: params_raw,
        min_ram_gb: min_ram,
        recommended_ram_gb: rec_ram,
        min_vram_gb: min_vram,
        // Q4_K_M is used as a conservative approximation for all fetched models.
        // Actual available quantizations depend on the GGUF files published for
        // each model.  RAM/VRAM estimates downstream reflect this assumption.
        quantization: "Q4_K_M".to_string(),
        context_length,
        use_case,
        is_moe,
        num_experts,
        active_experts,
        active_parameters: active_params,
        release_date,
        gguf_sources: vec![],
        capabilities: vec![],
        format: ModelFormat::default(),
        num_attention_heads,
        num_key_value_heads,
        num_hidden_layers,
        head_dim,
        attention_layout: crate::models::infer_attention_layout_from_name(&hf.id),
        license,
        hidden_size,
        moe_intermediate_size,
        vocab_size,
        shared_expert_intermediate_size,
    })
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Options controlling the update behaviour.
#[derive(Debug, Clone)]
pub struct UpdateOptions {
    /// Number of trending models to fetch (0 = skip).
    pub trending_limit: usize,
    /// Number of top-downloaded models to fetch (0 = skip).
    pub downloads_limit: usize,
    /// Optional HuggingFace API token (raises the anonymous rate limit).
    pub token: Option<String>,
}

impl Default for UpdateOptions {
    fn default() -> Self {
        Self {
            trending_limit: 100,
            downloads_limit: 50,
            token: None,
        }
    }
}

/// Fetch new models from HuggingFace and merge them into the local cache.
///
/// Returns `Ok((new_count, total_cached))` on success, where `new_count` is
/// the number of models added in this run and `total_cached` is the size of the
/// cache file after the update.
///
/// `progress` receives human-readable status strings suitable for printing to
/// stdout or displaying in a TUI.
pub fn update_model_cache(
    opts: &UpdateOptions,
    progress: impl Fn(&str),
) -> Result<(usize, usize), String> {
    use crate::models::ModelDatabase;

    // Names already embedded in the binary — never add these to the cache.
    // Use canonical_slug for the same normalization applied in ModelDatabase::new().
    let embedded_names: HashSet<String> = ModelDatabase::embedded()
        .get_all_models()
        .iter()
        .map(|m| crate::models::canonical_slug(&m.name))
        .collect();

    // Load the existing cache so we can append to it.
    let mut cached = load_cache();
    let already_cached: HashSet<String> = cached
        .iter()
        .map(|m| crate::models::canonical_slug(&m.name))
        .collect();

    let token = opts.token.as_deref();
    let mut all_hf: Vec<HfApiModel> = Vec::new();

    if opts.trending_limit > 0 {
        progress(&format!(
            "Fetching {} trending models from HuggingFace...",
            opts.trending_limit
        ));
        match hf_get_list("trendingScore", opts.trending_limit, token) {
            Ok(list) => {
                progress(&format!("  Received {} trending models", list.len()));
                all_hf.extend(list);
            }
            Err(e) => progress(&format!("  Warning: trending fetch failed — {e}")),
        }
    }

    if opts.downloads_limit > 0 {
        progress(&format!(
            "Fetching {} top-downloaded models...",
            opts.downloads_limit
        ));
        match hf_get_list("downloads", opts.downloads_limit, token) {
            Ok(list) => {
                progress(&format!("  Received {} download-ranked models", list.len()));
                all_hf.extend(list);
            }
            Err(e) => progress(&format!("  Warning: downloads fetch failed — {e}")),
        }
    }

    if all_hf.is_empty() {
        return Err("No models fetched — check your internet connection".to_string());
    }

    // Deduplicate by ID (trending and downloads lists can overlap).
    let mut seen: HashSet<String> = HashSet::new();
    all_hf.retain(|m| seen.insert(m.id.clone()));

    progress(&format!(
        "Processing {} unique models (fetching config.json for KV cache metadata)...",
        all_hf.len()
    ));

    let mut new_count = 0usize;
    for hf in all_hf {
        let id_slug = crate::models::canonical_slug(&hf.id);
        if embedded_names.contains(&id_slug) || already_cached.contains(&id_slug) {
            continue;
        }
        if let Some(model) = map_to_llm_model(hf, token) {
            cached.push(model);
            new_count += 1;
        }
    }

    let total = cached.len();
    progress(&format!(
        "Saving {} cached models ({} new)...",
        total, new_count
    ));
    save_cache(&cached)?;

    Ok((new_count, total))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_param_str_billions() {
        assert_eq!(parse_param_str("7B"), Some(7_000_000_000));
        assert_eq!(parse_param_str("70B"), Some(70_000_000_000));
        assert_eq!(parse_param_str("1.5B"), Some(1_500_000_000));
        assert_eq!(parse_param_str("405B"), Some(405_000_000_000));
    }

    #[test]
    fn test_parse_param_str_millions() {
        assert_eq!(parse_param_str("500M"), Some(500_000_000));
        assert_eq!(parse_param_str("135M"), Some(135_000_000));
    }

    #[test]
    fn test_parse_param_str_invalid() {
        assert_eq!(parse_param_str("INSTRUCT"), None);
        assert_eq!(parse_param_str(""), None);
        assert_eq!(parse_param_str("0.001B"), None); // below 0.05 threshold
    }

    #[test]
    fn test_extract_dense_model() {
        let (s, raw, moe, ne, ae, ap) = extract_model_params("meta-llama/Llama-3.1-8B-Instruct");
        assert_eq!(s, "8B");
        assert_eq!(raw, Some(8_000_000_000));
        assert!(!moe);
        assert!(ne.is_none());
        assert!(ae.is_none());
        assert!(ap.is_none());
    }

    #[test]
    fn test_extract_moe_nx_model() {
        let (s, raw, moe, ne, ae, _ap) = extract_model_params("mistralai/Mixtral-8x7B-v0.1");
        assert!(s.contains("8x"), "Expected MoE label, got: {}", s);
        assert!(raw.is_some());
        assert!(moe);
        assert_eq!(ne, Some(8));
        assert_eq!(ae, Some(2));
    }

    #[test]
    fn test_extract_fractional_param() {
        let (s, raw, ..) = extract_model_params("Qwen/Qwen2.5-1.5B-Instruct");
        assert!(s.contains("1.5") || s.contains("1"), "got: {}", s);
        assert!(raw.is_some());
    }

    #[test]
    fn test_infer_use_case_coding() {
        let uc = infer_use_case("deepseek-ai/DeepSeek-Coder-6.7B", &[]);
        assert!(uc.to_lowercase().contains("code"), "got: {}", uc);
    }

    #[test]
    fn test_infer_use_case_embedding() {
        let uc = infer_use_case("BAAI/bge-large-en-v1.5", &[]);
        assert!(uc.to_lowercase().contains("embed"), "got: {}", uc);
    }

    #[test]
    fn test_infer_context_length_keywords() {
        assert_eq!(infer_context_length("model-128k", None), 131_072);
        assert_eq!(infer_context_length("model-32k", None), 32_768);
    }

    #[test]
    fn test_infer_context_length_llama3() {
        // Llama-3 family defaults to 128 k context
        assert_eq!(
            infer_context_length("meta-llama/Llama-3.1-8B", None),
            131_072
        );
    }

    #[test]
    fn test_estimate_ram_dense() {
        let (min_r, rec_r, vram) = estimate_ram(7_000_000_000, false, None);
        assert!(min_r > 2.0 && min_r < 5.0, "min_ram={}", min_r);
        assert!(rec_r > min_r, "rec_ram should exceed min_ram");
        assert!(vram.is_some());
    }

    #[test]
    fn test_estimate_ram_moe() {
        // MoE: total=56B, active=14B → min_vram uses total params (all experts loaded)
        let (_, _, vram_moe) = estimate_ram(56_000_000_000, true, Some(14_000_000_000));
        let (_, _, vram_dense) = estimate_ram(56_000_000_000, false, None);
        // min_vram must reflect all parameters since all experts are loaded into memory
        assert!(
            (vram_moe.unwrap() - vram_dense.unwrap()).abs() < 0.01,
            "MoE min_vram ({}) should equal dense ({}) — all experts must be loaded",
            vram_moe.unwrap(),
            vram_dense.unwrap()
        );
    }

    // ────────────────────────────────────────────────────────────────────
    // HfConfig parsing tests — validates config.json extraction for
    // MoE bandwidth decomposition fields
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn test_hfconfig_parses_moe_fields() {
        // OLMoE-style config: uses intermediate_size for per-expert FFN
        let json = r#"{
            "hidden_size": 2048,
            "intermediate_size": 1024,
            "vocab_size": 50304,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "num_experts": 64,
            "num_experts_per_tok": 8
        }"#;
        let cfg: HfConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.hidden_size, Some(2048));
        assert_eq!(cfg.vocab_size, Some(50304));
        // moe_intermediate_size not present → should be None (fallback handled in extraction)
        assert_eq!(cfg.moe_intermediate_size, None);
        // But intermediate_size IS present as fallback
        assert_eq!(cfg.intermediate_size, Some(1024));
        assert_eq!(cfg.shared_expert_intermediate_size, None);
    }

    #[test]
    fn test_hfconfig_parses_shared_experts() {
        // Qwen1.5-MoE style: has both moe_intermediate_size and shared_expert_intermediate_size
        let json = r#"{
            "hidden_size": 2048,
            "moe_intermediate_size": 1408,
            "shared_expert_intermediate_size": 5632,
            "vocab_size": 151936,
            "num_hidden_layers": 24,
            "num_experts": 60,
            "num_experts_per_tok": 4
        }"#;
        let cfg: HfConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.moe_intermediate_size, Some(1408));
        assert_eq!(cfg.shared_expert_intermediate_size, Some(5632));
        assert_eq!(cfg.vocab_size, Some(151936));
    }

    #[test]
    fn test_hfconfig_parses_deepseek_shared_expert_derivation() {
        // DeepSeek-V2 style: has n_shared_experts but no shared_expert_intermediate_size
        let json = r#"{
            "hidden_size": 2048,
            "moe_intermediate_size": 1408,
            "intermediate_size": 10944,
            "n_shared_experts": 2,
            "vocab_size": 102400,
            "num_hidden_layers": 27,
            "n_routed_experts": 64,
            "num_experts_per_tok": 6
        }"#;
        let cfg: HfConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.n_shared_experts, Some(2));
        assert_eq!(cfg.intermediate_size, Some(10944));
        // shared_expert_intermediate_size should be derived: 2 × 10944 = 21888
        // (this derivation happens in the extraction block, not in serde)
        assert_eq!(cfg.shared_expert_intermediate_size, None); // not in JSON
        // Verify derivation logic:
        let derived = cfg
            .n_shared_experts
            .filter(|&n| n > 0)
            .and_then(|n| cfg.intermediate_size.map(|inter| n * inter));
        assert_eq!(derived, Some(2 * 10944));
    }

    #[test]
    fn test_hfconfig_parses_nested_text_config() {
        // Qwen3.5 style: architecture params nested under text_config
        let json = r#"{
            "model_type": "qwen3_5_moe",
            "text_config": {
                "hidden_size": 2048,
                "moe_intermediate_size": 512,
                "shared_expert_intermediate_size": 512,
                "vocab_size": 248320,
                "num_hidden_layers": 40,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "num_experts": 256,
                "num_experts_per_tok": 8
            }
        }"#;
        let cfg: HfConfig = serde_json::from_str(json).unwrap();
        // Top-level fields should be None
        assert_eq!(cfg.hidden_size, None);
        assert_eq!(cfg.vocab_size, None);
        // text_config should be populated
        let tc = cfg.text_config.as_ref().unwrap();
        assert_eq!(tc.hidden_size, Some(2048));
        assert_eq!(tc.moe_intermediate_size, Some(512));
        assert_eq!(tc.shared_expert_intermediate_size, Some(512));
        assert_eq!(tc.vocab_size, Some(248320));
        assert_eq!(tc.num_hidden_layers, Some(40));
        assert_eq!(tc.num_key_value_heads, Some(2));
    }
}
