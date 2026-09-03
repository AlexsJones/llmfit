//! LLM inference benchmarking against Ollama and OpenAI-compatible endpoints.
//!
//! Measures time-to-first-token (TTFT), tokens per second (TPS),
//! and total latency using real inference requests.

use std::time::{Duration, Instant};

use crate::providers::{OpenAiEndpointIdentity, fetch_openai_model_list, openai_model_ids};

/// Results from a single benchmark run.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BenchRun {
    /// Time to first token in milliseconds, if measurable.
    /// - Ollama: measured from `eval_duration` (accurate).
    /// - vLLM/MLX: `None` — would require streaming to measure; only wall-clock
    ///   total is available.
    pub ttft_ms: Option<f64>,
    /// Output tokens per second.
    pub tps: f64,
    /// Total request latency in milliseconds.
    pub total_ms: f64,
    /// Number of prompt tokens processed.
    pub prompt_tokens: u32,
    /// Number of output tokens generated.
    pub output_tokens: u32,
}

/// Aggregated benchmark results across multiple runs.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BenchResult {
    pub model: String,
    pub provider: String,
    pub runs: Vec<BenchRun>,
    pub summary: BenchSummary,
}

/// Statistical summary of benchmark runs.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BenchSummary {
    pub num_runs: usize,
    pub avg_ttft_ms: Option<f64>,
    pub avg_tps: f64,
    pub min_tps: f64,
    pub max_tps: f64,
    pub avg_total_ms: f64,
    pub avg_output_tokens: f64,
}

impl BenchSummary {
    fn from_runs(runs: &[BenchRun]) -> Self {
        let n = runs.len() as f64;
        if runs.is_empty() {
            return BenchSummary {
                num_runs: 0,
                avg_ttft_ms: None,
                avg_tps: 0.0,
                min_tps: 0.0,
                max_tps: 0.0,
                avg_total_ms: 0.0,
                avg_output_tokens: 0.0,
            };
        }
        // Only compute avg TTFT if any run has a measured value
        let ttft_values: Vec<f64> = runs.iter().filter_map(|r| r.ttft_ms).collect();
        let avg_ttft_ms = if ttft_values.is_empty() {
            None
        } else {
            Some(ttft_values.iter().sum::<f64>() / ttft_values.len() as f64)
        };
        BenchSummary {
            num_runs: runs.len(),
            avg_ttft_ms,
            avg_tps: runs.iter().map(|r| r.tps).sum::<f64>() / n,
            min_tps: runs.iter().map(|r| r.tps).fold(f64::INFINITY, f64::min),
            max_tps: runs.iter().map(|r| r.tps).fold(0.0_f64, f64::max),
            avg_total_ms: runs.iter().map(|r| r.total_ms).sum::<f64>() / n,
            avg_output_tokens: runs.iter().map(|r| r.output_tokens as f64).sum::<f64>() / n,
        }
    }
}

/// Test prompts of varying lengths for benchmarking.
const BENCH_PROMPTS: &[&str] = &[
    "Explain what a hash table is in 2 sentences.",
    "Write a Python function that checks if a string is a palindrome. Include a docstring.",
    "Compare and contrast TCP and UDP protocols. Cover reliability, ordering, speed, and common use cases. Be concise.",
    "You are a senior software engineer. Review this code and suggest improvements:\n\n```python\ndef fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)\n```",
];

// ── Ollama benchmarking ────────────────────────────────────────────

/// Ollama /api/generate response fields we care about.
/// Shared with `quality.rs` — both modules talk to the same endpoints.
#[derive(serde::Deserialize, Default)]
#[allow(dead_code)]
pub(crate) struct OllamaGenResponse {
    #[serde(default)]
    pub(crate) response: String,
    #[serde(default)]
    pub(crate) eval_count: Option<u64>,
    #[serde(default)]
    pub(crate) eval_duration: Option<u64>, // nanoseconds
    #[serde(default)]
    pub(crate) prompt_eval_count: Option<u64>,
    #[serde(default)]
    pub(crate) prompt_eval_duration: Option<u64>, // nanoseconds
    #[serde(default)]
    pub(crate) total_duration: Option<u64>, // nanoseconds
}

/// Benchmark a model via Ollama's /api/generate endpoint.
pub fn bench_ollama(
    base_url: &str,
    model: &str,
    num_runs: usize,
    on_progress: &dyn Fn(usize, usize),
) -> Result<BenchResult, String> {
    let url = format!("{}/api/generate", base_url.trim_end_matches('/'));
    let mut runs = Vec::with_capacity(num_runs);

    // Warmup request (don't count it)
    on_progress(0, num_runs);
    if let Err(e) = ollama_generate(&url, model, "Say hello.", 300) {
        return Err(format!(
            "Warmup request failed (is the model loaded?): {}",
            e
        ));
    }

    for i in 0..num_runs {
        on_progress(i + 1, num_runs);
        let prompt = BENCH_PROMPTS[i % BENCH_PROMPTS.len()];
        let run = ollama_generate(&url, model, prompt, 300)?;
        runs.push(run);
    }

    let summary = BenchSummary::from_runs(&runs);
    Ok(BenchResult {
        model: model.to_string(),
        provider: "ollama".to_string(),
        runs,
        summary,
    })
}

fn ollama_generate(
    url: &str,
    model: &str,
    prompt: &str,
    max_tokens: u32,
) -> Result<BenchRun, String> {
    let body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "stream": false,
        "options": {
            "num_predict": max_tokens,
        }
    });

    let start = Instant::now();
    let resp = ureq::post(url)
        .config()
        .timeout_global(Some(Duration::from_secs(300)))
        .build()
        .send_json(&body)
        .map_err(|e| format!("Ollama request failed: {}", e))?;

    let total_wall = start.elapsed();

    let resp_body: OllamaGenResponse = resp
        .into_body()
        .read_json()
        .map_err(|e| format!("Ollama JSON parse error: {}", e))?;

    // Ollama provides native timing in nanoseconds
    let prompt_tokens = resp_body.prompt_eval_count.unwrap_or(0) as u32;
    let output_tokens = resp_body.eval_count.unwrap_or(0) as u32;

    let ttft_ms = resp_body
        .prompt_eval_duration
        .map(|ns| ns as f64 / 1_000_000.0);

    let tps = if let (Some(eval_count), Some(eval_dur)) =
        (resp_body.eval_count, resp_body.eval_duration)
    {
        if eval_dur > 0 {
            eval_count as f64 / (eval_dur as f64 / 1_000_000_000.0)
        } else {
            0.0
        }
    } else if output_tokens > 0 {
        // Fallback to wall-clock
        output_tokens as f64 / total_wall.as_secs_f64()
    } else {
        0.0
    };

    let total_ms = resp_body
        .total_duration
        .map(|ns| ns as f64 / 1_000_000.0)
        .unwrap_or(total_wall.as_secs_f64() * 1000.0);

    Ok(BenchRun {
        ttft_ms,
        tps,
        total_ms,
        prompt_tokens,
        output_tokens,
    })
}

// ── OpenAI-compatible benchmarking (vLLM, MLX) ────────────────────

/// OpenAI-compatible chat completion response fields we care about.
/// Shared with `quality.rs` — both modules talk to the same endpoints.
#[derive(serde::Deserialize)]
#[allow(dead_code)]
pub(crate) struct ChatCompletionResponse {
    #[serde(default)]
    pub(crate) choices: Vec<ChatChoice>,
    #[serde(default)]
    pub(crate) usage: Option<ChatUsage>,
}

#[derive(serde::Deserialize)]
#[allow(dead_code)]
pub(crate) struct ChatChoice {
    #[serde(default)]
    pub(crate) message: Option<ChatMessage>,
}

#[derive(serde::Deserialize)]
#[allow(dead_code)]
pub(crate) struct ChatMessage {
    #[serde(default)]
    pub(crate) content: Option<String>,
}

#[derive(serde::Deserialize)]
pub(crate) struct ChatUsage {
    #[serde(default)]
    pub(crate) prompt_tokens: u32,
    #[serde(default)]
    pub(crate) completion_tokens: u32,
}

/// Benchmark a model via OpenAI-compatible /v1/chat/completions.
pub fn bench_openai_compat(
    base_url: &str,
    model: &str,
    provider_name: &str,
    num_runs: usize,
    on_progress: &dyn Fn(usize, usize),
) -> Result<BenchResult, String> {
    let url = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));
    let mut runs = Vec::with_capacity(num_runs);

    // Warmup
    on_progress(0, num_runs);
    if let Err(e) = openai_chat(&url, model, "Say hello.", 100) {
        return Err(format!(
            "Warmup request failed (is the endpoint reachable?): {}",
            e
        ));
    }

    for i in 0..num_runs {
        on_progress(i + 1, num_runs);
        let prompt = BENCH_PROMPTS[i % BENCH_PROMPTS.len()];
        let run = openai_chat(&url, model, prompt, 300)?;
        runs.push(run);
    }

    let summary = BenchSummary::from_runs(&runs);
    Ok(BenchResult {
        model: model.to_string(),
        provider: provider_name.to_string(),
        runs,
        summary,
    })
}

fn openai_chat(url: &str, model: &str, prompt: &str, max_tokens: u32) -> Result<BenchRun, String> {
    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": false,
    });

    let start = Instant::now();

    // TTFT is estimated from wall clock: prompt_tokens / total_tokens * total_time.
    // This is a rough heuristic — actual TTFT requires streaming (not implemented).
    let resp = ureq::post(url)
        .config()
        .timeout_global(Some(Duration::from_secs(300)))
        .build()
        .send_json(&body)
        .map_err(|e| format!("{} request failed: {}", url, e))?;

    let total_wall = start.elapsed();

    let completion: ChatCompletionResponse = resp
        .into_body()
        .read_json()
        .map_err(|e| format!("JSON parse error: {}", e))?;

    let usage = completion.usage.unwrap_or(ChatUsage {
        prompt_tokens: 0,
        completion_tokens: 0,
    });

    let output_tokens = usage.completion_tokens;
    let prompt_tokens = usage.prompt_tokens;

    // TTFT cannot be measured without streaming — set to None.
    let total_ms = total_wall.as_secs_f64() * 1000.0;

    let tps = if output_tokens > 0 && total_wall.as_secs_f64() > 0.0 {
        output_tokens as f64 / total_wall.as_secs_f64()
    } else {
        0.0
    };

    Ok(BenchRun {
        ttft_ms: None,
        tps,
        total_ms,
        prompt_tokens,
        output_tokens,
    })
}

// ── Auto-detect and benchmark ──────────────────────────────────────

/// Which provider to benchmark against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BenchTarget {
    Ollama { url: String, model: String },
    VLlm { url: String, model: String },
    Ferrum { url: String, model: String },
    Mlx { url: String, model: String },
    LlamaCpp { url: String, model: String },
}

/// Benchmark a discovered target while preserving its provider attribution.
pub fn benchmark_target(
    target: &BenchTarget,
    num_runs: usize,
    on_progress: &dyn Fn(usize, usize),
) -> Result<BenchResult, String> {
    match target {
        BenchTarget::Ollama { url, model } => bench_ollama(url, model, num_runs, on_progress),
        BenchTarget::VLlm { url, model } => {
            bench_openai_compat(url, model, "vllm", num_runs, on_progress)
        }
        BenchTarget::Ferrum { url, model } => {
            bench_openai_compat(url, model, "ferrum", num_runs, on_progress)
        }
        BenchTarget::Mlx { url, model } => {
            bench_openai_compat(url, model, "mlx", num_runs, on_progress)
        }
        BenchTarget::LlamaCpp { url, model } => {
            bench_openai_compat(url, model, "llamacpp", num_runs, on_progress)
        }
    }
}

/// Base URL for a running Ferrum server.
/// `FERRUM_HOST` is a full URL and defaults to Ferrum's default listen port.
pub fn ferrum_url() -> String {
    std::env::var("FERRUM_HOST")
        .ok()
        .filter(|host| !host.trim().is_empty())
        .map(|host| host.trim().trim_end_matches('/').to_string())
        .unwrap_or_else(|| "http://localhost:8000".to_string())
}

/// Base URL for a running llama-server instance.
/// `LLAMA_SERVER_HOST` (full URL) wins; otherwise localhost with
/// `LLAMA_SERVER_PORT` (default 8080, llama-server's own default).
pub fn llamacpp_url() -> String {
    if let Ok(host) = std::env::var("LLAMA_SERVER_HOST")
        && !host.trim().is_empty()
    {
        return host.trim().trim_end_matches('/').to_string();
    }
    let port = std::env::var("LLAMA_SERVER_PORT").unwrap_or_else(|_| "8080".to_string());
    format!("http://localhost:{}", port)
}

/// Positively identify a llama-server instance via its `/props` endpoint.
/// llama.cpp serves it; MLX and vLLM return 404, which disambiguates
/// llama-server from `mlx_lm.server` on the shared default port 8080.
pub fn probe_llamacpp(base_url: &str) -> bool {
    ureq::get(&format!("{}/props", base_url.trim_end_matches('/')))
        .config()
        .timeout_global(Some(Duration::from_secs(2)))
        .build()
        .call()
        .is_ok()
}

fn vllm_url() -> String {
    let port = std::env::var("VLLM_PORT").unwrap_or_else(|_| "8000".to_string());
    format!("http://localhost:{}", port)
}

fn openai_bench_urls() -> Vec<String> {
    let mut urls = vec![vllm_url()];
    let ferrum = ferrum_url();
    if !urls.contains(&ferrum) {
        urls.push(ferrum);
    }
    urls
}

fn identified_openai_models(
    base_url: &str,
    timeout: Duration,
) -> Result<(OpenAiEndpointIdentity, Vec<String>), String> {
    let (list, identity) = fetch_openai_model_list(base_url, timeout)
        .ok_or_else(|| format!("Cannot read {}/v1/models", base_url.trim_end_matches('/')))?;
    let models = openai_model_ids(&list).map(str::to_owned).collect();
    Ok((identity, models))
}

fn choose_model(models: &[String], hint: Option<&str>) -> Result<String, String> {
    if models.is_empty() {
        return Err("No models loaded".to_string());
    }

    if let Some(hint) = hint {
        let hint_lower = hint.to_lowercase();
        if let Some(model) = models
            .iter()
            .find(|model| model.to_lowercase().contains(&hint_lower))
        {
            return Ok(model.clone());
        }
    }

    Ok(models[0].clone())
}

fn target_for_identity(
    identity: OpenAiEndpointIdentity,
    url: String,
    model: String,
) -> Option<BenchTarget> {
    match identity {
        OpenAiEndpointIdentity::Vllm => Some(BenchTarget::VLlm { url, model }),
        OpenAiEndpointIdentity::Ferrum => Some(BenchTarget::Ferrum { url, model }),
        _ => None,
    }
}

fn auto_detect_openai_target(urls: &[String], hint: Option<&str>) -> Option<BenchTarget> {
    urls.iter().find_map(|url| {
        let (identity, models) = identified_openai_models(url, Duration::from_secs(5)).ok()?;
        let model = choose_model(&models, hint).ok()?;
        target_for_identity(identity, url.clone(), model)
    })
}

fn discover_openai_targets(urls: &[String]) -> Vec<BenchTarget> {
    let mut targets = Vec::new();
    for url in urls {
        let Ok((identity, models)) = identified_openai_models(url, Duration::from_secs(3)) else {
            continue;
        };
        for model in models {
            if let Some(target) = target_for_identity(identity, url.clone(), model) {
                targets.push(target);
            }
        }
    }
    targets
}

/// Auto-detect available providers and pick the best one to benchmark.
pub fn auto_detect_target(model_hint: Option<&str>) -> Result<BenchTarget, String> {
    // vLLM and Ferrum share port 8000 by default, so classify the endpoint
    // from positive `owned_by` evidence instead of assigning it by port.
    if let Some(target) = auto_detect_openai_target(&openai_bench_urls(), model_hint) {
        return Ok(target);
    }

    // Check Ollama
    let ollama_url =
        std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string());
    if ureq::get(&format!("{}/api/tags", ollama_url))
        .config()
        .timeout_global(Some(Duration::from_secs(2)))
        .build()
        .call()
        .is_ok()
        && let Ok(model_name) = detect_ollama_model(&ollama_url, model_hint)
    {
        return Ok(BenchTarget::Ollama {
            url: ollama_url,
            model: model_name,
        });
    }

    // Check llama-server before MLX: both default to port 8080, but only
    // llama.cpp answers /props, so it can be identified positively.
    let llama_url = llamacpp_url();
    if probe_llamacpp(&llama_url)
        && let Ok(model_name) = detect_llamacpp_model(&llama_url, model_hint)
    {
        return Ok(BenchTarget::LlamaCpp {
            url: llama_url,
            model: model_name,
        });
    }

    // Check MLX
    let mlx_url =
        std::env::var("MLX_LM_HOST").unwrap_or_else(|_| "http://localhost:8080".to_string());
    if ureq::get(&format!("{}/v1/models", mlx_url))
        .config()
        .timeout_global(Some(Duration::from_secs(2)))
        .build()
        .call()
        .is_ok()
        && let Ok(model_name) = detect_openai_model(&mlx_url, model_hint)
    {
        return Ok(BenchTarget::Mlx {
            url: mlx_url,
            model: model_name,
        });
    }

    Err(
        "No inference provider found. Start Ollama, vLLM, Ferrum, MLX, or llama-server first."
            .to_string(),
    )
}

/// Discover all available models across all providers.
pub fn discover_all_targets() -> Vec<BenchTarget> {
    let ollama_url =
        std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string());
    let llama_url = llamacpp_url();
    let mlx_url =
        std::env::var("MLX_LM_HOST").unwrap_or_else(|_| "http://localhost:8080".to_string());
    discover_all_targets_at(&openai_bench_urls(), &ollama_url, &llama_url, &mlx_url)
}

fn identified_openai_claims_url(targets: &[BenchTarget], candidate_url: &str) -> bool {
    let candidate_url = candidate_url.trim_end_matches('/');
    targets.iter().any(|target| match target {
        BenchTarget::VLlm { url, .. } | BenchTarget::Ferrum { url, .. } => {
            url.trim_end_matches('/') == candidate_url
        }
        _ => false,
    })
}

fn discover_all_targets_at(
    openai_urls: &[String],
    ollama_url: &str,
    llama_url: &str,
    mlx_url: &str,
) -> Vec<BenchTarget> {
    let mut targets = discover_openai_targets(openai_urls);

    // Check Ollama
    if let Ok(models) = list_ollama_models(ollama_url) {
        for model in models {
            targets.push(BenchTarget::Ollama {
                url: ollama_url.to_string(),
                model,
            });
        }
    }

    // Check llama-server before MLX: both default to port 8080, but only
    // llama.cpp answers /props, so it can be identified positively. Do not
    // probe a URL already claimed by vLLM or Ferrum.
    let llamacpp_found =
        !identified_openai_claims_url(&targets, llama_url) && probe_llamacpp(llama_url);
    if llamacpp_found && let Ok(models) = list_llamacpp_models(llama_url) {
        for model in models {
            targets.push(BenchTarget::LlamaCpp {
                url: llama_url.to_string(),
                model,
            });
        }
    }

    // Check MLX (skip if the llama-server probe already claimed this URL,
    // e.g. both on the default port 8080)
    if !identified_openai_claims_url(&targets, mlx_url)
        && !(llamacpp_found && mlx_url.trim_end_matches('/') == llama_url.trim_end_matches('/'))
        && let Ok(models) = list_openai_models(mlx_url)
    {
        for model in models {
            targets.push(BenchTarget::Mlx {
                url: mlx_url.to_string(),
                model,
            });
        }
    }

    targets
}

fn list_openai_models(base_url: &str) -> Result<Vec<String>, String> {
    let url = format!("{}/v1/models", base_url);
    let resp = ureq::get(&url)
        .config()
        .timeout_global(Some(Duration::from_secs(3)))
        .build()
        .call()
        .map_err(|e| format!("{}", e))?;

    let body: serde_json::Value = resp.into_body().read_json().map_err(|e| format!("{}", e))?;
    let models = body
        .get("data")
        .and_then(|d: &serde_json::Value| d.as_array())
        .ok_or("no data")?;

    Ok(models
        .iter()
        .filter_map(|m| {
            m.get("id")
                .and_then(|i: &serde_json::Value| i.as_str())
                .map(|s| s.to_string())
        })
        .collect())
}

/// List models reported by llama.cpp, removing local GGUF paths from IDs.
fn list_llamacpp_models(base_url: &str) -> Result<Vec<String>, String> {
    list_openai_models(base_url).map(|models| {
        models
            .into_iter()
            .map(|model| normalize_llamacpp_model_id(&model))
            .collect()
    })
}

fn list_ollama_models(base_url: &str) -> Result<Vec<String>, String> {
    let url = format!("{}/api/tags", base_url);
    let resp = ureq::get(&url)
        .config()
        .timeout_global(Some(Duration::from_secs(3)))
        .build()
        .call()
        .map_err(|e| format!("{}", e))?;

    #[derive(serde::Deserialize)]
    struct Tags {
        models: Vec<M>,
    }
    #[derive(serde::Deserialize)]
    struct M {
        name: String,
    }

    let tags: Tags = resp.into_body().read_json().map_err(|e| format!("{}", e))?;
    Ok(tags.models.into_iter().map(|m| m.name).collect())
}

/// Detect model from a given base URL (OpenAI-compatible /v1/models).
pub fn detect_model_from_url(base_url: &str, hint: Option<&str>) -> Result<String, String> {
    detect_openai_model(base_url, hint)
}

/// Detect a model only after the endpoint positively identifies as vLLM.
pub fn detect_vllm_model(base_url: &str, hint: Option<&str>) -> Result<String, String> {
    detect_identified_openai_model(base_url, hint, OpenAiEndpointIdentity::Vllm, "vLLM")
}

/// Detect a model only after the endpoint positively identifies as Ferrum.
pub fn detect_ferrum_model(base_url: &str, hint: Option<&str>) -> Result<String, String> {
    detect_identified_openai_model(base_url, hint, OpenAiEndpointIdentity::Ferrum, "Ferrum")
}

fn detect_identified_openai_model(
    base_url: &str,
    hint: Option<&str>,
    expected: OpenAiEndpointIdentity,
    provider: &str,
) -> Result<String, String> {
    let (identity, models) = identified_openai_models(base_url, Duration::from_secs(5))?;
    if identity != expected {
        return Err(format!(
            "Endpoint at {} did not identify as {}",
            base_url.trim_end_matches('/'),
            provider
        ));
    }
    choose_model(&models, hint)
}

fn detect_llamacpp_model(base_url: &str, hint: Option<&str>) -> Result<String, String> {
    detect_openai_model(base_url, hint).map(|model| normalize_llamacpp_model_id(&model))
}

/// llama-server reports the value passed to `--model` as its OpenAI model ID.
/// When that value is a local GGUF path, retain only its filename so benchmark
/// results do not expose local filesystem details or fragment model grouping.
fn normalize_llamacpp_model_id(model_id: &str) -> String {
    let is_gguf = model_id.to_ascii_lowercase().ends_with(".gguf");
    let has_separator = model_id.contains(std::path::MAIN_SEPARATOR)
        || (std::path::MAIN_SEPARATOR != '/' && model_id.contains('/'))
        || (std::path::MAIN_SEPARATOR != '\\' && model_id.contains('\\'));

    if is_gguf && has_separator {
        model_id
            .rsplit(['/', '\\'])
            .next()
            .unwrap_or(model_id)
            .to_string()
    } else {
        model_id.to_string()
    }
}

fn detect_openai_model(base_url: &str, hint: Option<&str>) -> Result<String, String> {
    let url = format!("{}/v1/models", base_url);
    let resp = ureq::get(&url)
        .config()
        .timeout_global(Some(Duration::from_secs(5)))
        .build()
        .call()
        .map_err(|e| format!("Cannot reach {}: {}", url, e))?;

    let body: serde_json::Value = resp
        .into_body()
        .read_json()
        .map_err(|e| format!("JSON error: {}", e))?;

    let models = body
        .get("data")
        .and_then(|d: &serde_json::Value| d.as_array())
        .ok_or("No models found")?;

    if models.is_empty() {
        return Err("No models loaded".to_string());
    }

    // If hint provided, find matching model
    if let Some(hint) = hint {
        let hint_lower = hint.to_lowercase();
        for m in models {
            if let Some(id) = m.get("id").and_then(|i: &serde_json::Value| i.as_str())
                && id.to_lowercase().contains(&hint_lower)
            {
                return Ok(id.to_string());
            }
        }
    }

    // Return first model
    models[0]
        .get("id")
        .and_then(|i: &serde_json::Value| i.as_str())
        .map(|s| s.to_string())
        .ok_or("Model has no id".to_string())
}

fn detect_ollama_model(base_url: &str, hint: Option<&str>) -> Result<String, String> {
    let url = format!("{}/api/tags", base_url);
    let resp = ureq::get(&url)
        .config()
        .timeout_global(Some(Duration::from_secs(5)))
        .build()
        .call()
        .map_err(|e| format!("Cannot reach Ollama: {}", e))?;

    #[derive(serde::Deserialize)]
    struct Tags {
        models: Vec<Model>,
    }
    #[derive(serde::Deserialize)]
    struct Model {
        name: String,
    }

    let tags: Tags = resp
        .into_body()
        .read_json()
        .map_err(|e| format!("JSON error: {}", e))?;

    if tags.models.is_empty() {
        return Err("No models installed in Ollama".to_string());
    }

    if let Some(hint) = hint {
        let hint_lower = hint.to_lowercase();
        for m in &tags.models {
            if m.name.to_lowercase().contains(&hint_lower) {
                return Ok(m.name.clone());
            }
        }
    }

    Ok(tags.models[0].name.clone())
}

// ── Display helpers ────────────────────────────────────────────────

impl BenchResult {
    pub fn display(&self) {
        println!();
        println!("  === Benchmark Results ===");
        println!("  Model:    {}", self.model);
        println!("  Provider: {}", self.provider);
        println!("  Runs:     {}", self.summary.num_runs);
        println!();
        println!(
            "  TPS:      {:.1} avg  ({:.1} min / {:.1} max)",
            self.summary.avg_tps, self.summary.min_tps, self.summary.max_tps
        );
        if let Some(ttft) = self.summary.avg_ttft_ms {
            println!("  TTFT:     {:.0} ms avg", ttft);
        } else {
            println!("  TTFT:     n/a (streaming required)");
        }
        println!("  Latency:  {:.0} ms avg", self.summary.avg_total_ms);
        println!(
            "  Output:   {:.0} tokens avg",
            self.summary.avg_output_tokens
        );
        println!();

        // Per-run breakdown
        println!("  Run  TPS      TTFT     Latency  Tokens");
        println!("  ───  ───────  ───────  ───────  ──────");
        for (i, run) in self.runs.iter().enumerate() {
            println!(
                "  {:>3}  {:>6.1}   {:>5}ms  {:>5.0}ms  {:>5}",
                i + 1,
                run.tps,
                run.ttft_ms
                    .map(|t| format!("{:.0}", t))
                    .unwrap_or_else(|| "n/a".to_string()),
                run.total_ms,
                run.output_tokens
            );
        }
        println!();
    }

    pub fn display_json(&self) {
        let json = serde_json::json!({
            "benchmark": {
                "model": self.model,
                "provider": self.provider,
                "summary": self.summary,
                "runs": self.runs,
            }
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).expect("JSON serialization failed")
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FERRUM_MODELS_FIXTURE: &str = r#"{"data":[{"id":"ferrum","owned_by":"ferrum"}]}"#;
    const VLLM_MODELS_FIXTURE: &str = r#"{"object":"list","data":[{"id":"facebook/opt-125m","object":"model","created":1788290125,"owned_by":"vllm","root":"facebook/opt-125m","parent":null,"max_model_len":512}]}"#;
    const UNKNOWN_MODELS_FIXTURE: &str = r#"{"data":[{"id":"foreign-model"}]}"#;
    const CHAT_COMPLETION_FIXTURE: &str = r#"{"choices":[{"message":{"content":"hello"}}],"usage":{"prompt_tokens":3,"completion_tokens":2}}"#;

    /// Serve one JSON response on an ephemeral loopback port.
    fn serve_fixture(body: &'static str) -> String {
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind test listener");
        let addr = listener.local_addr().expect("test listener addr");
        std::thread::spawn(move || {
            while let Ok((mut stream, _)) = listener.accept() {
                let mut request = [0u8; 2048];
                let _ = stream.read(&mut request);
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes());
            }
        });
        format!("http://{}", addr)
    }

    #[test]
    fn auto_detects_ferrum_from_positive_identity() {
        let url = serve_fixture(FERRUM_MODELS_FIXTURE);
        let target = auto_detect_openai_target(std::slice::from_ref(&url), None)
            .expect("Ferrum endpoint should be detected");
        assert_eq!(
            target,
            BenchTarget::Ferrum {
                url,
                model: "ferrum".to_string(),
            }
        );
    }

    #[test]
    fn discovery_includes_only_identified_vllm_and_ferrum_targets() {
        let ferrum_url = serve_fixture(FERRUM_MODELS_FIXTURE);
        let vllm_url = serve_fixture(VLLM_MODELS_FIXTURE);
        let unknown_url = serve_fixture(UNKNOWN_MODELS_FIXTURE);

        let targets =
            discover_openai_targets(&[ferrum_url.clone(), vllm_url.clone(), unknown_url.clone()]);

        assert_eq!(
            targets,
            vec![
                BenchTarget::Ferrum {
                    url: ferrum_url,
                    model: "ferrum".to_string(),
                },
                BenchTarget::VLlm {
                    url: vllm_url,
                    model: "facebook/opt-125m".to_string(),
                },
            ]
        );
        assert!(auto_detect_openai_target(&[unknown_url], None).is_none());
    }

    #[test]
    fn discover_all_does_not_reclassify_a_claimed_ferrum_url() {
        let ferrum_url = serve_fixture(FERRUM_MODELS_FIXTURE);
        let targets = discover_all_targets_at(
            std::slice::from_ref(&ferrum_url),
            &ferrum_url,
            &ferrum_url,
            &ferrum_url,
        );

        assert_eq!(
            targets,
            vec![BenchTarget::Ferrum {
                url: ferrum_url,
                model: "ferrum".to_string(),
            }]
        );
    }

    #[test]
    fn explicit_vllm_and_ferrum_detection_rejects_foreign_endpoints() {
        let vllm_url = serve_fixture(VLLM_MODELS_FIXTURE);
        let ferrum_url = serve_fixture(FERRUM_MODELS_FIXTURE);
        let unknown_url = serve_fixture(UNKNOWN_MODELS_FIXTURE);

        assert_eq!(
            detect_vllm_model(&vllm_url, None).expect("vLLM identity should match"),
            "facebook/opt-125m"
        );
        assert_eq!(
            detect_ferrum_model(&ferrum_url, None).expect("Ferrum identity should match"),
            "ferrum"
        );
        assert!(detect_vllm_model(&ferrum_url, Some("ferrum")).is_err());
        assert!(detect_ferrum_model(&vllm_url, Some("opt-125m")).is_err());
        assert!(detect_vllm_model(&unknown_url, Some("foreign-model")).is_err());
    }

    #[test]
    fn ferrum_target_keeps_json_provider_attribution() {
        let url = serve_fixture(CHAT_COMPLETION_FIXTURE);
        let result = benchmark_target(
            &BenchTarget::Ferrum {
                url,
                model: "ferrum".to_string(),
            },
            1,
            &|_, _| {},
        )
        .expect("Ferrum OpenAI-compatible benchmark should succeed");

        assert_eq!(result.provider, "ferrum");
        let json = serde_json::json!({ "result": result });
        assert_eq!(json["result"]["provider"], "ferrum");
    }

    #[test]
    fn normalizes_llamacpp_gguf_paths_to_filenames() {
        assert_eq!(
            normalize_llamacpp_model_id("/home/llmfit/gguf/arcee-ai_AFM-4.5B-Q4_K_M.gguf"),
            "arcee-ai_AFM-4.5B-Q4_K_M.gguf"
        );
        assert_eq!(
            normalize_llamacpp_model_id(r"C:\\models\\arcee-ai_AFM-4.5B-Q4_K_M.gguf"),
            "arcee-ai_AFM-4.5B-Q4_K_M.gguf"
        );
    }

    #[test]
    fn preserves_non_path_or_non_gguf_llamacpp_ids() {
        assert_eq!(normalize_llamacpp_model_id("llama-3.2:3b"), "llama-3.2:3b");
        assert_eq!(
            normalize_llamacpp_model_id("/models/config.json"),
            "/models/config.json"
        );
    }

    fn make_run(ttft_ms: f64, tps: f64, total_ms: f64, output_tokens: u32) -> BenchRun {
        BenchRun {
            ttft_ms: Some(ttft_ms),
            tps,
            total_ms,
            prompt_tokens: 10,
            output_tokens,
        }
    }

    // ──────────────────────────────────────────────────────────────────
    // BenchSummary::from_runs
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_summary_multiple_runs() {
        let runs = vec![
            make_run(100.0, 20.0, 500.0, 50),
            make_run(150.0, 30.0, 600.0, 60),
            make_run(200.0, 10.0, 700.0, 70),
        ];
        let s = BenchSummary::from_runs(&runs);

        assert_eq!(s.num_runs, 3);
        assert!((s.avg_ttft_ms.unwrap() - 150.0).abs() < 0.01);
        assert!((s.avg_tps - 20.0).abs() < 0.01);
        assert!((s.min_tps - 10.0).abs() < 0.01);
        assert!((s.max_tps - 30.0).abs() < 0.01);
        assert!((s.avg_total_ms - 600.0).abs() < 0.01);
        assert!((s.avg_output_tokens - 60.0).abs() < 0.01);
    }

    #[test]
    fn test_summary_single_run() {
        let runs = vec![make_run(100.0, 25.0, 500.0, 50)];
        let s = BenchSummary::from_runs(&runs);

        assert_eq!(s.num_runs, 1);
        assert!((s.avg_ttft_ms.unwrap() - 100.0).abs() < 0.01);
        assert!((s.avg_tps - 25.0).abs() < 0.01);
        // min == max == avg for a single run
        assert!((s.min_tps - 25.0).abs() < 0.01);
        assert!((s.max_tps - 25.0).abs() < 0.01);
        assert!((s.avg_total_ms - 500.0).abs() < 0.01);
        assert!((s.avg_output_tokens - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_summary_empty_runs() {
        let runs: Vec<BenchRun> = vec![];
        let s = BenchSummary::from_runs(&runs);

        assert_eq!(s.num_runs, 0);
        assert_eq!(s.avg_tps, 0.0);
        assert_eq!(s.min_tps, 0.0);
        assert_eq!(s.max_tps, 0.0);
        assert_eq!(s.avg_ttft_ms, None);
        assert_eq!(s.avg_total_ms, 0.0);
        assert_eq!(s.avg_output_tokens, 0.0);
    }

    #[test]
    fn test_summary_min_max_correctness() {
        let runs = vec![
            make_run(50.0, 5.0, 200.0, 20),
            make_run(60.0, 50.0, 300.0, 30),
            make_run(70.0, 25.0, 400.0, 40),
            make_run(80.0, 100.0, 500.0, 50),
            make_run(90.0, 1.0, 600.0, 60),
        ];
        let s = BenchSummary::from_runs(&runs);

        assert_eq!(s.num_runs, 5);
        assert!((s.min_tps - 1.0).abs() < 0.01);
        assert!((s.max_tps - 100.0).abs() < 0.01);
        // avg_tps = (5+50+25+100+1)/5 = 36.2
        assert!((s.avg_tps - 36.2).abs() < 0.01);
    }

    #[test]
    fn test_summary_identical_runs() {
        let runs = vec![
            make_run(100.0, 20.0, 500.0, 50),
            make_run(100.0, 20.0, 500.0, 50),
            make_run(100.0, 20.0, 500.0, 50),
        ];
        let s = BenchSummary::from_runs(&runs);

        assert_eq!(s.num_runs, 3);
        assert!((s.avg_tps - 20.0).abs() < 0.01);
        assert!((s.min_tps - 20.0).abs() < 0.01);
        assert!((s.max_tps - 20.0).abs() < 0.01);
        assert!((s.avg_ttft_ms.unwrap() - 100.0).abs() < 0.01);
    }
}
